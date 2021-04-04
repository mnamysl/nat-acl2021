import copy
import logging
from pathlib import Path
from typing import List, Union
import time
import datetime
import sys
import inspect

import torch
from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

try:
    from apex import amp
except ImportError:
    amp = None

import flair
import flair.nn
from flair.data import MultiCorpus, Corpus
from flair.datasets import DataLoader
from flair.optim import ExpAnnealLR
from flair.training_utils import (
    init_output_file,
    WeightExtractor,
    log_line,
    add_file_handler,
    Result,
    store_embeddings,
    AnnealOnPlateau,
)
import random

from flair.trainers import ModelTrainer

from robust_ner.noise import (
    make_char_vocab,
)

from robust_ner.confusion_matrix import (
    load_confusion_matrix,
    filter_cmx,
    make_vocab_from_lut,
)

from robust_ner.typos import load_typos

from robust_ner.enums import (
    TrainingMode,
    MisspellingMode,
    EvalMode,
)

from flair_ext.models import NATSequenceTagger

from pysia.nmt import init_translator

log = logging.getLogger("flair")


class ParameterizedModelTrainer(ModelTrainer):
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
        optimizer: torch.optim.Optimizer = SGD,
        epoch: int = 0,
        use_tensorboard: bool = False,
    ):
        """
        Initialize a model trainer
        :param model: The model that you want to train. The model should inherit from flair.nn.Model
        :param corpus: The dataset used to train the model, should be of type Corpus
        :param optimizer: The optimizer to use (typically SGD or Adam)
        :param epoch: The starting epoch (normally 0 but could be higher if you continue training model)
        :param use_tensorboard: If True, writes out tensorboard information
        """

        super(ParameterizedModelTrainer, self).__init__(model, corpus, optimizer, epoch, use_tensorboard)

    def train(
        self,
        base_path: Union[Path, str],
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        mini_batch_chunk_size: int = None,
        max_epochs: int = 100,
        scheduler = AnnealOnPlateau,
        anneal_factor: float = 0.5,
        patience: int = 3,
        initial_extra_patience = 0,
        min_learning_rate: float = 0.0001,
        train_with_dev: bool = False,
        monitor_train: bool = False,
        monitor_test: bool = False,
        embeddings_storage_mode: str = "cpu",
        checkpoint: bool = False,
        save_final_model: bool = True,
        anneal_with_restarts: bool = False,
        anneal_with_prestarts: bool = False,
        batch_growth_annealing: bool = False,
        shuffle: bool = True,
        param_selection_mode: bool = False,
        num_workers: int = 6,
        sampler=None,
        use_amp: bool = False,
        amp_opt_level: str = "O1",
        eval_on_train_fraction = 0.,
        eval_on_train_shuffle = False,
        valid_with_misspellings: bool = True,
        corpus_name: str = "",
        **kwargs,
    ) -> dict:
        """
        Trains any class that implements the flair.nn.Model interface.
        :param base_path: Main path to which all output during training is logged and models are saved
        :param learning_rate: Initial learning rate
        :param mini_batch_size: Size of mini-batches during training
        :param mini_batch_chunk_size: If mini-batches are larger than this number, they get broken down into chunks of this size for processing purposes
        :param max_epochs: Maximum number of epochs to train. Terminates training if this number is surpassed.
        :param anneal_factor: The factor by which the learning rate is annealed
        :param patience: Patience is the number of epochs with no improvement the Trainer waits until annealing the learning rate
        :param min_learning_rate: If the learning rate falls below this threshold, training terminates
        :param train_with_dev: If True, training is performed using both train+dev data
        :param monitor_train: If True, training data is evaluated at end of each epoch
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param embeddings_storage_mode: One of 'none' (all embeddings are deleted and freshly recomputed),
                                        'cpu' (embeddings are stored on CPU) or 'gpu' (embeddings are stored on GPU)
        :param checkpoint: If True, a full checkpoint is saved at end of each epoch
        :param save_final_model: If True, final model is saved
        :param anneal_with_restarts: If True, the last best model is restored when annealing the learning rate
        :param shuffle: If True, data is shuffled during training
        :param param_selection_mode: If True, testing is performed against dev data. Use this mode when doing parameter selection.
        :param num_workers: Number of workers in your data loader.
        :param sampler: You can pass a data sampler here for special sampling of data.
        :param eval_on_train_fraction: the fraction of train data to do the evaluation on,
                                        if 0. the evaluation is not performed on fraction of training data,
                                        if 'dev' the size is determined from dev set size
        :param eval_on_train_shuffle: if True the train data fraction is determined on the start of training
                                        and kept fixed during training, otherwise it's sampled at beginning of each epoch
        :param valid_with_misspellings: use a combination of the original loss and the loss computed using the misspelled sentences for validation
        :param kwargs: Other arguments for the Optimizer
        :return:
        """

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter()
            except:
                log_line(log)
                log.warning(
                    "ATTENTION! PyTorch >= 1.1.0 and pillow are required for TensorBoard support!"
                )
                log_line(log)
                self.use_tensorboard = False
                pass

        if use_amp:
            if sys.version_info < (3, 0):
                raise RuntimeError("Apex currently only supports Python 3. Aborting.")
            if amp is None:
                raise RuntimeError(
                    "Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
                    "to enable mixed-precision training."
                )

        if mini_batch_chunk_size is None:
            mini_batch_chunk_size = mini_batch_size
        if learning_rate < min_learning_rate:
            min_learning_rate = learning_rate / 10

        initial_learning_rate = learning_rate

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        log_handler = add_file_handler(log, base_path / "training.log")

        log_line(log)
        log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{learning_rate}"')
        log.info(f' - mini_batch_size: "{mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log.info(f' - train_with_dev: "{train_with_dev}"')
        log.info(f' - batch_growth_annealing: "{batch_growth_annealing}"')
        log.info(f' - mixed precision training: "{use_amp}"')
        log.info(f' - valid_with_misspellings: "{valid_with_misspellings}"')
        log.info("Model:")
        log.info(f' - hidden_size: "{self.model.hidden_size}"')
        log.info(f' - train_mode: "{self.model.train_mode}"')
        log.info(f' - misspell_mode: "{self.model.misspell_mode}"')        
        log.info(f' - alpha: "{self.model.alpha}"')
        log.info(f' - beta: "{self.model.beta}"')

        if self.model.misspell_mode == MisspellingMode.Seq2Seq:
            log.info(f' - errgen_model: "{self.model.errgen_model_train}"')
            log.info(f' - errgen_mode: "{self.model.errgen_mode_train}"')

            from pysia.utils import is_generation_mode, is_correction_mode
            
            if is_generation_mode(self.model.errgen_mode_train):
                log.info(f' - errgen_temp: "{self.model.errgen_temp_train}"')
                log.info(f' - errgen_topk: "{self.model.errgen_topk_train}"')
            elif is_correction_mode(self.model.errgen_mode_train):
                log.info(f' - errgen_nbest: "{self.model.errgen_nbest_train}"')
                log.info(f' - errgen_beam_size: "{self.model.errgen_beam_size_train}"')

        elif self.model.misspell_mode in [MisspellingMode.Random]:
            log.info(f' - misspelling_rate: "{self.model.misspelling_rate_train}"')
        elif self.model.misspell_mode in [MisspellingMode.ConfusionMatrixBased]:
            log.info(f' - cmx_file: "{self.model.cmx_file_train}"')
        elif self.model.misspell_mode in [MisspellingMode.Typos]:
            log.info(f' - typos_file: "{self.model.typos_file_train}"')
            log.info(f' - misspelling_rate: "{self.model.misspelling_rate_train}"')

        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Device: {flair.device}")
        log_line(log)
        log.info(f"Embeddings storage mode: {embeddings_storage_mode}")

        # determine what splits (train, dev, test) to evaluate and log
        log_train = True if monitor_train else False
        log_test = (
            True
            if (not param_selection_mode and self.corpus.test and monitor_test)
            else False
        )
        log_dev = True if not train_with_dev else False
        log_train_part = (
            True
            if (eval_on_train_fraction == "dev" or eval_on_train_fraction > 0.0)
            else False
        )

        if log_train_part:
            train_part_size = (
                len(self.corpus.dev)
                if eval_on_train_fraction == "dev"
                else int(len(self.corpus.train) * eval_on_train_fraction)
            )
            assert train_part_size > 0
            if not eval_on_train_shuffle:
                train_part_indices = list(range(train_part_size))
                train_part = torch.utils.data.dataset.Subset(
                    self.corpus.train, train_part_indices
                )

        log_test = not log_dev
        eval_misspelling_rate = 0.05
        eval_misspelling_mode = MisspellingMode.Random

        log_suffix = lambda prefix, rate, cm, mode: f"{prefix} (misspell: cmx={cm})" if mode == MisspellingMode.ConfusionMatrixBased else f"{prefix} (misspell: rate={rate})"

        # prepare loss logging file and set up header
        loss_txt = init_output_file(base_path, "loss.tsv")

        weight_extractor = WeightExtractor(base_path)

        optimizer: torch.optim.Optimizer = self.optimizer(
            self.model.parameters(), lr=learning_rate, **kwargs
        )
        if use_amp:
            self.model, optimizer = amp.initialize(
                self.model, optimizer, opt_level=amp_opt_level
            )

        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev else "max"

        lr_scheduler = scheduler(
            optimizer,
            factor=anneal_factor,
            patience=patience,
            initial_extra_patience=initial_extra_patience,
            mode=anneal_mode,
            verbose=True,
        )

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data = ConcatDataset([self.corpus.train, self.corpus.dev])

        # initialize sampler if provided
        if sampler is not None:
            # init with default values if only class is provided
            if inspect.isclass(sampler):
                sampler = sampler()
            # set dataset to sample from
            sampler.set_dataset(train_data)
            shuffle = False

        dev_clean_score_history = []
        dev_noisy_score_history = []
        dev_clean_loss_history = []
        dev_noisy_loss_history = []
        train_loss_history = []

        micro_batch_size = mini_batch_chunk_size

        complete_data = ConcatDataset([self.corpus.train, self.corpus.dev, self.corpus.test])
        char_vocab = make_char_vocab(complete_data)
        log.info(f"Vocabulary of the corpus (#{len(char_vocab)}): {char_vocab}")

        cmx, lut, typos = None, {}, {}
        if self.model.misspell_mode == MisspellingMode.ConfusionMatrixBased:
            cmx, lut = load_confusion_matrix(self.model.cmx_file_train)
            cmx, lut = filter_cmx(cmx, lut, char_vocab)
        elif self.model.misspell_mode == MisspellingMode.Typos:
            typos = load_typos(self.model.typos_file_train, char_vocab, False)

        if self.model.misspell_mode == MisspellingMode.Seq2Seq:
            translator, opt = init_translator(self.model.errgen_model_train, self.model.errgen_mode_train, log, 
                temp=self.model.errgen_temp_train, topk=self.model.errgen_topk_train, nbest=self.model.errgen_nbest_train, 
                beam_size=self.model.errgen_beam_size_train, shard_size=20000, batch_size=256, verbose=True)
        else:
            translator, opt = None, None

        loss_params = {}
        loss_params["verbose"] = False
        loss_params["char_vocab"] = char_vocab
        loss_params["cmx"] = cmx
        loss_params["lut"] = lut
        loss_params["typos"] = typos
        loss_params["translator"] = translator
        loss_params["opt"] = opt
        loss_params["translation_mode"] = self.model.errgen_mode_train
        loss_params["embeddings_storage_mode"] = embeddings_storage_mode

        if self.model.train_mode == TrainingMode.Combined and self.model.beta > 0.0:
            
            batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    sampler=sampler,
                )

            sum_sent_len, cnt_sent = 0, 0
            for batch_no, batch in enumerate(batch_loader):
                for sent in batch:
                    sum_sent_len += len(sent)
                cnt_sent += len(batch)
        
            mean_tokens_per_batch = float(sum_sent_len) / float(cnt_sent)
            loss_params["mean_tokens_per_batch"] = mean_tokens_per_batch
            log.info(f"mean_tokens_per_batch = {mean_tokens_per_batch:.4f}")
    
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = learning_rate

            for self.epoch in range(self.epoch + 1, max_epochs + 1):
                log_line(log)

                if anneal_with_prestarts:
                    last_epoch_model_state_dict = copy.deepcopy(self.model.state_dict())

                if eval_on_train_shuffle:
                    train_part_indices = list(range(self.corpus.train))
                    random.shuffle(train_part_indices)
                    train_part_indices = train_part_indices[:train_part_size]
                    train_part = torch.utils.data.dataset.Subset(
                        self.corpus.train, train_part_indices
                    )

                # get new learning rate
                for group in optimizer.param_groups:
                    learning_rate = group["lr"]

                if learning_rate != previous_learning_rate and batch_growth_annealing:
                    mini_batch_size *= 2

                # reload last best model if annealing with restarts is enabled
                if (
                    (anneal_with_restarts or anneal_with_prestarts)
                    and learning_rate != previous_learning_rate
                    and (base_path / "best-model.pt").exists()
                ):
                    if anneal_with_restarts:
                        log.info("resetting to best model")
                        self.model.load_state_dict(
                            self.model.load(base_path / "best-model.pt").state_dict()
                        )
                    if anneal_with_prestarts:
                        log.info("resetting to pre-best model")
                        self.model.load_state_dict(
                            self.model.load(base_path / "pre-best-model.pt").state_dict()
                        )

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < min_learning_rate:
                    log_line(log)
                    log.info("learning rate too small - quitting training!")
                    log_line(log)
                    break

                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    sampler=sampler,
                )

                self.model.train()

                train_loss: float = 0
                train_auxilary_losses = {}
                seen_batches = 0
                total_number_of_batches = len(batch_loader)

                modulo = max(1, int(total_number_of_batches / 10))                

                # process mini-batches
                batch_time = 0
                for batch_no, batch in enumerate(batch_loader):
                    start_time = time.time()

                    # zero the gradients on the model and optimizer
                    self.model.zero_grad()
                    optimizer.zero_grad()

                    # if necessary, make batch_steps
                    batch_steps = [batch]
                    if len(batch) > micro_batch_size:
                        batch_steps = [
                            batch[x : x + micro_batch_size]
                            for x in range(0, len(batch), micro_batch_size)
                        ]

                    # forward and backward for batch
                    for batch_step in batch_steps:

                        # forward pass
                        loss, auxilary_losses = self.model.forward_loss(batch_step, params=loss_params)

                        # Backward
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                    # do the optimizer step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    seen_batches += 1
                    train_loss += loss.item()

                    for k,v in auxilary_losses.items():
                        train_auxilary_losses[k] = train_auxilary_losses.get(k, 0) + v

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(batch, embeddings_storage_mode)

                    batch_time += time.time() - start_time
                    if seen_batches % modulo == 0:
                        msg = f"epoch {self.epoch} - iter {seen_batches}/{total_number_of_batches} - loss {train_loss / seen_batches:.6f} - samples/sec: {mini_batch_size * modulo / batch_time:.2f}"
                        # note: this is the loss accumulated in the current epoch divided by the number of already seen batches
                        if len(train_auxilary_losses) > 0:
                            accuracies = [(key, value) for (key, value) in train_auxilary_losses.items() if key.startswith("acc_")]
                            counts = [(key, value) for (key, value) in train_auxilary_losses.items() if key.startswith("cnt_") or key.startswith("sum_")]
                            losses = [(key, value) for (key, value) in train_auxilary_losses.items() if key.startswith("loss_")]
                            aux_losses_str = ""
                            if len(losses) > 0:
                                aux_losses_str = " ".join([f"{key}={value / seen_batches:.6f}" for (key, value) in losses])
                            if len(accuracies) > 0:
                                if len(aux_losses_str) > 0:
                                    aux_losses_str += " "
                                aux_losses_str += " ".join([f"{key}={value / seen_batches:.2f}%" for (key, value) in accuracies])
                            if len(counts) > 0:
                                if len(aux_losses_str) > 0:
                                    aux_losses_str += " "
                                aux_losses_str += " ".join([f"{key}={value / seen_batches:.2f}" for (key, value) in counts])
                            msg += f" ({aux_losses_str})"
                        
                        log.info(msg)
                        batch_time = 0
                        iteration = self.epoch * total_number_of_batches + batch_no
                        if not param_selection_mode:
                            weight_extractor.extract_weights(
                                self.model.state_dict(), iteration
                            )

                train_loss /= seen_batches
                for k,v in auxilary_losses.items():
                    train_auxilary_losses[k] /= seen_batches

                self.model.eval()

                log_line(log)
                log.info(
                    f"EPOCH {self.epoch} done: loss {train_loss:.4f} - lr {learning_rate:.4f}"
                )

                if self.use_tensorboard:
                    writer.add_scalar("train_loss", train_loss, self.epoch)

                # anneal against train loss if training with dev, otherwise anneal against dev score
                current_score = train_loss

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_train:
                    train_eval_result, train_loss = self.model.evaluate(
                        DataLoader(
                            self.corpus.train,
                            batch_size=mini_batch_chunk_size,
                            num_workers=num_workers,
                        ),
                        embeddings_storage_mode=embeddings_storage_mode,
                        eval_dict_name=corpus_name,
                    )
                    result_line += f"\t{train_eval_result.log_line}"

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.train, embeddings_storage_mode)

                if log_train_part:
                    train_part_eval_result, train_part_loss = self.model.evaluate(
                        DataLoader(
                            train_part,
                            batch_size=mini_batch_chunk_size,
                            num_workers=num_workers,
                        ),
                        embeddings_storage_mode=embeddings_storage_mode,
                        eval_dict_name=corpus_name,
                    )
                    result_line += (
                        f"\t{train_part_loss}\t{train_part_eval_result.log_line}"
                    )
                    log.info(
                        f"TRAIN_SPLIT : loss {train_part_loss} - score {round(train_part_eval_result.main_score, 4)}"
                    )

                if log_dev:
                    dev_eval_result_clean, dev_loss_clean = self.model.evaluate(
                        DataLoader(
                            self.corpus.dev,
                            batch_size=mini_batch_chunk_size,
                            num_workers=num_workers,
                        ),
                        embeddings_storage_mode=embeddings_storage_mode,
                        eval_dict_name=corpus_name,
                    )
                    result_line += f"\t{dev_loss_clean}\t{dev_eval_result_clean.log_line}"
                    log.info(
                        f"DEV : loss {dev_loss_clean} - score {round(dev_eval_result_clean.main_score, 4)}"
                    )
                    # calculate scores using dev data if available
                    # append dev score to score history
                    dev_clean_score_history.append(dev_eval_result_clean.main_score)
                    dev_clean_loss_history.append(dev_loss_clean.item())     

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.dev, embeddings_storage_mode)

                    if self.use_tensorboard:
                        writer.add_scalar("dev_clean_loss", dev_loss_clean, self.epoch)
                        writer.add_scalar(
                            "dev_clean_score", dev_eval_result_clean.main_score, self.epoch
                        )
                    
                    # evaluate on misspellings 
                    if valid_with_misspellings:
                        dev_eval_result_noisy, dev_loss_noisy = self.model.evaluate(
                            DataLoader(
                                self.corpus.dev,
                                batch_size=mini_batch_chunk_size,
                                num_workers=num_workers,
                            ),
                            embeddings_storage_mode=embeddings_storage_mode,
                            eval_mode=EvalMode.Misspellings,
                            misspell_mode=eval_misspelling_mode,
                            char_vocab=char_vocab,
                            cmx=cmx,
                            lut=lut,
                            typos=typos,
                            misspelling_rate=eval_misspelling_rate,
                            eval_dict_name=corpus_name,
                        )                            

                        result_line += f"\t{dev_loss_noisy}\t{dev_eval_result_noisy.log_line}"
                        log.info(                                
                            f"{log_suffix('DEV', eval_misspelling_rate, '', eval_misspelling_mode)}"
                            + f" : loss {dev_loss_noisy} - score {round(dev_eval_result_noisy.main_score, 4)}"
                        )                        

                        # calculate scores using dev data if available
                        # append dev score to score history
                        dev_noisy_score_history.append(dev_eval_result_noisy)
                        dev_noisy_loss_history.append(dev_loss_noisy.item())

                        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                        store_embeddings(self.corpus.dev, embeddings_storage_mode)

                        if self.use_tensorboard:
                            writer.add_scalar("dev_noisy_loss", dev_loss_noisy, self.epoch)
                            writer.add_scalar(
                                "dev_noisy_score", dev_eval_result_noisy.main_score, self.epoch
                            )
                        
                    if valid_with_misspellings:
                        current_score = (dev_eval_result_clean.main_score + dev_eval_result_noisy.main_score) / 2.0
                        dev_loss = (dev_loss_clean + dev_loss_noisy) / 2.0
                    else:
                        current_score = dev_eval_result_clean.main_score
                        dev_loss = dev_loss_clean
                    # else: current_score = train_loss
                    
                if log_test:
                    test_eval_result_clean, test_loss_clean = self.model.evaluate(
                        DataLoader(
                            self.corpus.test,
                            batch_size=mini_batch_chunk_size,
                            num_workers=num_workers,
                        ),
                        base_path / "test.tsv",
                        embeddings_storage_mode=embeddings_storage_mode,
                        eval_dict_name=corpus_name,
                    )
                    result_line += f"\t{test_loss_clean}\t{test_eval_result_clean.log_line}"
                    log.info(
                        f"TEST : loss {test_loss_clean} - score {round(test_eval_result_clean.main_score, 4)}"
                    )

                    # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                    store_embeddings(self.corpus.test, embeddings_storage_mode)

                    if self.use_tensorboard:
                        writer.add_scalar("test_clean_loss", test_loss_clean, self.epoch)
                        writer.add_scalar(
                            "test_clean_score", test_eval_result_clean.main_score, self.epoch
                        ) 

                    if valid_with_misspellings:               
                        # evaluate on misspellings                       
                        test_eval_result_noisy, test_loss_noisy = self.model.evaluate(
                            DataLoader(
                                self.corpus.test,
                                batch_size=mini_batch_chunk_size,
                                num_workers=num_workers,
                            ),                            
                            base_path / f"test.tsv",                            
                            embeddings_storage_mode=embeddings_storage_mode,
                            eval_mode=EvalMode.Misspellings,
                            misspell_mode=eval_misspelling_mode,
                            char_vocab=char_vocab,
                            cmx=cmx,
                            lut=lut,
                            typos=typos,
                            misspelling_rate=eval_misspelling_rate,
                            eval_dict_name=corpus_name,
                        )
                        
                        result_line += f"\t{test_loss_noisy}\t{test_eval_result_noisy.log_line}"
                        log.info(
                            f"{log_suffix('TEST', eval_misspelling_rate, '', eval_misspelling_mode)}"
                            + f" : loss {test_loss_noisy} - score {round(test_eval_result_noisy.main_score, 4)}"
                        )       

                        # depending on memory mode, embeddings are moved to CPU, GPU or deleted
                        store_embeddings(self.corpus.test, embeddings_storage_mode)

                        if self.use_tensorboard:
                            writer.add_scalar("test_noisy_loss", test_loss_noisy, self.epoch)
                            writer.add_scalar(
                                "test_noisy_score", test_eval_result_noisy.main_score, self.epoch
                            )                                                           

                # determine learning rate annealing through scheduler. Use auxiliary metric for AnnealOnPlateau
                if not train_with_dev and isinstance(lr_scheduler, AnnealOnPlateau):
                    lr_scheduler.step(current_score, dev_loss)
                else:
                    lr_scheduler.step(current_score)

                train_loss_history.append(train_loss)

                # determine bad epoch number
                try:
                    bad_epochs = lr_scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    new_learning_rate = group["lr"]
                if new_learning_rate != previous_learning_rate:
                    bad_epochs = patience + 1
                    if previous_learning_rate == initial_learning_rate: bad_epochs += initial_extra_patience

                # log bad epochs
                log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

                # output log file
                with open(loss_txt, "a") as f:

                    # make headers on first epoch
                    if self.epoch == 1:
                        f.write(
                            f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
                        )

                        if log_train:
                            f.write(
                                "\tTRAIN_"
                                + "\tTRAIN_".join(
                                    train_eval_result.log_header.split("\t")
                                )
                            )
                        if log_train_part:
                            f.write(
                                "\tTRAIN_PART_LOSS\tTRAIN_PART_"
                                + "\tTRAIN_PART_".join(
                                    train_part_eval_result.log_header.split("\t")
                                )
                            )
                        if log_dev:
                            f.write(
                                "\tDEV_LOSS\tDEV_"
                                + "\tDEV_".join(dev_eval_result_clean.log_header.split("\t"))
                            )
                        if log_test:
                            f.write(
                                "\tTEST_LOSS\tTEST_"
                                + "\tTEST_".join(
                                    test_eval_result_clean.log_header.split("\t")
                                )
                            )

                    f.write(
                        f"\n{self.epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
                    )
                    f.write(result_line)

                # if checkpoint is enabled, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.save_checkpoint(base_path / "checkpoint.pt")

                # if we use dev data, remember best model based on dev evaluation score
                if (
                    (not train_with_dev or anneal_with_restarts or anneal_with_prestarts)
                    and not param_selection_mode
                    and current_score == lr_scheduler.best
                    and bad_epochs == 0
                ):
                    log.info("saving best model")
                    self.model.save(base_path / "best-model.pt")

                    if anneal_with_prestarts:
                        current_state_dict = self.model.state_dict()
                        self.model.load_state_dict(last_epoch_model_state_dict)
                        self.model.save(base_path / "pre-best-model.pt")
                        self.model.load_state_dict(current_state_dict)

            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt")

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")

            if self.use_tensorboard:
                writer.close()

            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt")
                log.info("Done.")

        # test best model if test data is present
        if self.corpus.test:
            final_score_clean = self.final_test(base_path, mini_batch_chunk_size, num_workers, embeddings_storage_mode, corpus_name=corpus_name)
            final_score_noisy = self.final_test(base_path, mini_batch_chunk_size, num_workers, embeddings_storage_mode,
                eval_mode=EvalMode.Misspellings, misspell_mode=eval_misspelling_mode,
                misspelling_rate=eval_misspelling_rate, char_vocab=char_vocab, cmx=cmx, lut=lut, typos=typos, corpus_name=corpus_name)

        else:
            final_score_clean, final_score_noisy = 0, 0
            log.info("Test data not provided setting final score to 0")

        log.removeHandler(log_handler)

        if self.use_tensorboard:
            writer.close()

        return {
            "test_score_clean": final_score_clean,
            "test_score_noisy": final_score_noisy,
            "dev_clean_score_history": dev_clean_score_history,
            "dev_noisy_score_history": dev_noisy_score_history,
            "train_loss_history": train_loss_history,
            "dev_clean_loss_history": dev_clean_loss_history,
            "dev_noisy_loss_history": dev_noisy_loss_history,
        }

    def save_checkpoint(self, model_file: Union[str, Path]):
        corpus = self.corpus
        self.corpus = None
        torch.save(self, str(model_file), pickle_protocol=4)
        self.corpus = corpus

    @classmethod
    def load_checkpoint(cls, checkpoint: Union[Path, str], corpus: Corpus):
        model: ParameterizedModelTrainer = torch.load(checkpoint, map_location=flair.device)
        model.corpus = corpus
        return model

    def final_test(
        self,
        base_path: Path,
        eval_mini_batch_size: int,
        num_workers: int = 8,
        embeddings_storage_mode: str = "cpu",
        eval_mode: EvalMode = EvalMode.Standard,
        misspell_mode: MisspellingMode = MisspellingMode.Random,
        misspelling_rate: float = 0.0,
        char_vocab: set = {},
        cmx = None,
        lut = {},
        typos = {},
        corpus_name = "",
    ):

        log_line(log)
        log.info("Testing using best model ...")

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model = self.model.load(base_path / "best-model.pt")

        test_results, test_loss = self.model.evaluate(
            DataLoader(
                self.corpus.test,
                batch_size=eval_mini_batch_size,
                num_workers=num_workers,
            ),
            out_path=base_path / "test.tsv",
            embeddings_storage_mode=embeddings_storage_mode,
            eval_mode=eval_mode,
            misspell_mode=misspell_mode,
            misspelling_rate=misspelling_rate,
            char_vocab=char_vocab,
            cmx=cmx,
            lut=lut,
            typos=typos,
            eval_dict_name=corpus_name,
        )

        test_results: Result = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)

        # if we are training over multiple datasets, do evaluation for each
        if type(self.corpus) is MultiCorpus:
            for subcorpus in self.corpus.corpora:
                log_line(log)
                self.model.evaluate(
                    DataLoader(
                        subcorpus.test,
                        batch_size=eval_mini_batch_size,
                        num_workers=num_workers,
                    ),
                    out_path=base_path / f"{subcorpus.name}-test.tsv",
                    embeddings_storage_mode=embeddings_storage_mode,
                    eval_mode=eval_mode,
                    misspell_mode=misspell_mode,
                    misspelling_rate=misspelling_rate,
                    char_vocab=char_vocab,
                    cmx=cmx,
                    lut=lut,
                    typos=typos,
                    eval_dict_name=corpus_name,
                )

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score

#import logging
#from pathlib import Path
#from typing import List, Union
