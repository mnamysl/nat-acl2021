import logging
import random
from pathlib import Path
from typing import List, Union, Optional, Callable
from collections import Counter

import math

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
#from flair.datasets import DataLoader

from tqdm import tqdm

import flair.nn
from flair.data import Dictionary, Sentence, Token, Label, space_tokenizer
from flair.datasets import SentenceDataset, StringDataset
from flair.embeddings import TokenEmbeddings
from flair.file_utils import cached_path
from flair.training_utils import Metric, EvaluationMetric, Result, store_embeddings

from enum import Enum

from flair.models import SequenceTagger

from robust_ner.enums import TrainingMode, EvalMode, MisspellingMode, CorrectionMode
from robust_ner.noise import noise_sentences
from robust_ner.embeddings import check_embeddings, clear_embeddings
from robust_ner.seq2seq import Seq2SeqMode

from flair_ext.nn import ParameterizedModel


log = logging.getLogger("flair")

def get_masked_sum(loss_unreduced, lengths):

    loss_sum = 0
    for batch_idx, length in enumerate(lengths):
        loss_sum += loss_unreduced[batch_idx][:length].sum()
    
    return loss_sum

def get_per_token_mean(loss_sum, lengths):
    return loss_sum / float(sum(lengths))

def get_per_batch_mean(loss_sum, lengths):
    return loss_sum / float(len(lengths))


class NATSequenceTagger(SequenceTagger, ParameterizedModel):
    def __init__(
        self,
        hidden_size: int,
        embeddings: TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        use_crf: bool = True,
        use_rnn: bool = True,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        reproject_to: int = None,
        train_initial_hidden_state: bool = False,
        rnn_type: str = "LSTM",
        pickle_module: str = "pickle",
        train_mode: TrainingMode = TrainingMode.Combined,
        alpha: float = 0.0,
        beta: float = 0.0,
        misspelling_rate_train: float = 0.0,
        cmx_file = "",
        typos_file = "",
        errgen_model: str = "",
        errgen_mode: Seq2SeqMode = Seq2SeqMode.ErrorGenerationTok, 
        errgen_temp: float = 1.0, 
        errgen_topk: int = -1, 
        errgen_nbest: int = 5, 
        errgen_beam_size: int = 10,
    ):
        """
        Initializes a SequenceTagger
        :param hidden_size: number of hidden states in RNN
        :param embeddings: word embeddings used in tagger
        :param tag_dictionary: dictionary of tags you want to predict
        :param tag_type: string identifier for tag type
        :param use_crf: if True use CRF decoder, else project directly to tag space
        :param use_rnn: if True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param reproject_to: set this to control the dimensionality of the reprojection layer
        :param locked_dropout: locked dropout probability
        :param rnn_type: type of RNN layers: LSTM, GRU
        :param train_mode: training mode to use (combined)
        :param alpha: weight of the augmentation objective
        :param beta: weight of the stability objective
        :param misspelling_rate_train: error rate to use for training
        :param cmx_file: confusion matrix file to use for training
        :param typos_file: typos file to use for training
        :param errgen_model: seq2seq error generation model path
        :param errgen_mode: seq2seq generation mode
        :param errgen_temp: seq2seq sampling temperature
        :param errgen_topk: seq2seq sampling top-k candidates
        :param errgen_nbest: seq2seq beam search n-best paths
        :param errgen_beam_size: seq2seq beam size
        :param train_initial_hidden_state: if True, trains initial hidden state of RNN
        """

        super(NATSequenceTagger, self).__init__(hidden_size=hidden_size, embeddings=embeddings, tag_dictionary=tag_dictionary, tag_type=tag_type,
              use_crf=use_crf, use_rnn=use_rnn, rnn_layers=rnn_layers, dropout=dropout, word_dropout=word_dropout, locked_dropout=locked_dropout, 
              reproject_to=reproject_to, train_initial_hidden_state=train_initial_hidden_state, rnn_type=rnn_type, pickle_module=pickle_module)

        self.set_training_params(train_mode=train_mode, alpha=alpha, beta=beta, misspelling_rate_train=misspelling_rate_train, 
            cmx_file=cmx_file, typos_file=typos_file, errgen_model=errgen_model, errgen_mode=errgen_mode, errgen_temp=errgen_temp, 
            errgen_topk=errgen_topk, errgen_nbest=errgen_nbest, errgen_beam_size=errgen_beam_size)

        rnn_input_dim: int = self.embeddings.embedding_length
        
        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(rnn_input_dim, rnn_input_dim)

        if self.use_rnn and self.rnn_type in ["LSTM", "GRU"]:
            
            self.rnn = getattr(torch.nn, self.rnn_type)(
                rnn_input_dim,
                hidden_size,
                num_layers=self.nlayers,
                dropout=0.0 if self.nlayers == 1 else 0.5,
                bidirectional=True,
                batch_first=True,
            )        
        
        self.to(flair.device)

    def set_training_params(self, train_mode: TrainingMode, alpha: float = 0.0, beta: float = 0.0, 
        misspelling_rate_train: float = 0.0, cmx_file = "", typos_file = "",
        errgen_model="", errgen_mode=Seq2SeqMode.ErrorGenerationTok, errgen_temp=1.0, errgen_topk=-1, 
        errgen_nbest=5, errgen_beam_size=10):

        self.train_mode = train_mode
        self.alpha = alpha
        self.beta = beta
        self.misspelling_rate_train = misspelling_rate_train
        self.cmx_file_train = cmx_file
        self.typos_file_train = typos_file
        self.errgen_model_train = errgen_model
        self.errgen_mode_train = errgen_mode
        self.errgen_temp_train = errgen_temp
        self.errgen_topk_train = errgen_topk
        self.errgen_nbest_train = errgen_nbest
        self.errgen_beam_size_train = errgen_beam_size

        if self.cmx_file_train:
            self.misspell_mode = MisspellingMode.ConfusionMatrixBased
        elif self.typos_file_train:
            self.misspell_mode = MisspellingMode.Typos
        elif self.errgen_model_train:
            self.misspell_mode = MisspellingMode.Seq2Seq
        else:
            self.misspell_mode = MisspellingMode.Random

    def _get_state_dict(self):
        model_state = super(NATSequenceTagger, self)._get_state_dict()
        model_state["train_mode"] = self.train_mode
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):

        rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
        use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
        use_word_dropout = (
            0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
        )
        use_locked_dropout = (
            0.0
            if "use_locked_dropout" not in state.keys()
            else state["use_locked_dropout"]
        )
        train_initial_hidden_state = (
            False
            if "train_initial_hidden_state" not in state.keys()
            else state["train_initial_hidden_state"]
        )
        train_mode = TrainingMode.Combined if "train_mode" not in state.keys() else state["train_mode"]

        model = NATSequenceTagger(
            hidden_size=state["hidden_size"],
            embeddings=state["embeddings"],
            tag_dictionary=state["tag_dictionary"],
            tag_type=state["tag_type"],
            use_crf=state["use_crf"],
            use_rnn=state["use_rnn"],
            rnn_layers=state["rnn_layers"],
            dropout=use_dropout,
            word_dropout=use_word_dropout,
            locked_dropout=use_locked_dropout,
            train_initial_hidden_state=train_initial_hidden_state,
            rnn_type=rnn_type,
            train_mode=train_mode,
        )
        model.load_state_dict(state["state_dict"])
        return model

    def predict(
        self,
        sentences: Union[List[Sentence], Sentence, List[str], str],
        mini_batch_size=32,
        embeddings_storage_mode="none",
        all_tag_prob: bool = False,
        verbose: bool = False,
        use_tokenizer: Union[bool, Callable[[str], List[Token]]] = space_tokenizer,
        eval_mode: EvalMode = EvalMode.Standard,
        misspell_mode: MisspellingMode = MisspellingMode.Random,
        misspelling_rate: float = 0.0,
        char_vocab: set = {}, 
        lut: dict = {},
        cmx: np.array = None,
        typos: dict = {},
        correction_mode: CorrectionMode = CorrectionMode.NotSpecified,
    ) -> List[Sentence]:
        """
        Predict sequence tags for Named Entity Recognition task
        :param sentences: a Sentence or a string or a List of Sentence or a List of string.
        :param mini_batch_size: size of the minibatch, usually bigger is more rapid but consume more memory,
        up to a point when it has no more effect.
        :param embeddings_storage_mode: 'none' for the minimum memory footprint, 'cpu' to store embeddings in Ram,
        'gpu' to store embeddings in GPU memory.
        :param all_tag_prob: True to compute the score for each tag on each token,
        otherwise only the score of the best tag is returned
        :param verbose: set to True to display a progress bar
        :param use_tokenizer: a custom tokenizer when string are provided (default is space based tokenizer).
        :return: List of Sentence enriched by the predicted tags
        """

        predict_params = {}
        predict_params["eval_mode"] = eval_mode
        predict_params["misspelling_rate"] = misspelling_rate
        predict_params["misspell_mode"] = misspell_mode
        predict_params["char_vocab"] = char_vocab
        predict_params["lut"] = lut
        predict_params["cmx"] = cmx
        predict_params["typos"] = typos
        predict_params["correction_mode"] = correction_mode

        with torch.no_grad():
            if not sentences:
                return sentences

            if isinstance(sentences, Sentence) or isinstance(sentences, str):
                sentences = [sentences]

            if (flair.device.type == "cuda") and embeddings_storage_mode == "cpu":
                log.warning(
                    "You are inferring on GPU with parameter 'embeddings_storage_mode' set to 'cpu'."
                    "This option will slow down your inference, usually 'none' (default value) "
                    "is a better choice."
                )

            # reverse sort all sequences by their length
            rev_order_len_index = sorted(
                range(len(sentences)), key=lambda k: len(sentences[k]), reverse=True
            )
            original_order_index = sorted(
                range(len(rev_order_len_index)), key=lambda k: rev_order_len_index[k]
            )

            reordered_sentences: List[Union[Sentence, str]] = [
                sentences[index] for index in rev_order_len_index
            ]

            if isinstance(sentences[0], Sentence):
                # remove previous embeddings
                store_embeddings(reordered_sentences, "none")
                dataset = SentenceDataset(reordered_sentences)
            else:
                dataset = StringDataset(
                    reordered_sentences, use_tokenizer=use_tokenizer
                )
            dataloader = DataLoader(
                dataset=dataset, batch_size=mini_batch_size, collate_fn=lambda x: x
            )

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            # progress bar for verbosity
            if verbose:
                dataloader = tqdm(dataloader)

            results: List[Sentence] = []
            for i, batch in enumerate(dataloader):

                if verbose:
                    dataloader.set_description(f"Inferencing on batch {i}")
                results += batch
                batch = self._filter_empty_sentences(batch)
                # stop if all sentences are empty
                if not batch:
                    continue

                feature: torch.Tensor = self.forward(batch, predict_params)
                tags, all_tags = self._obtain_labels(
                    feature=feature,
                    batch_sentences=batch,
                    transitions=transitions,
                    get_all_tags=all_tag_prob,
                )

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token.add_tag_label(self.tag_type, tag)

                # all_tags will be empty if all_tag_prob is set to False, so the for loop will be avoided
                for (sentence, sent_all_tags) in zip(batch, all_tags):
                    for (token, token_all_tags) in zip(sentence.tokens, sent_all_tags):
                        token.add_tags_proba_dist(self.tag_type, token_all_tags)

                # clearing token embeddings to save memory
                store_embeddings(batch, storage_mode=embeddings_storage_mode)

            results: List[Union[Sentence, str]] = [
                results[index] for index in original_order_index
            ]
            assert len(sentences) == len(results)
            return results

    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embeddings_storage_mode: str = "none",
        eval_mode: EvalMode = EvalMode.Standard,
        misspell_mode: MisspellingMode = MisspellingMode.Random,
        misspelling_rate: float = 0.0,
        char_vocab: set = {}, 
        lut: dict = {},
        cmx: np.array = None,
        typos: dict = {},
        correction_mode: CorrectionMode = CorrectionMode.NotSpecified,
        eval_dict_name = None,
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
    ) -> (Result, float):

        if type(out_path) == str:
            out_path = Path(out_path)

        from robust_ner.spellcheck import load_correction_dict, get_lang_from_corpus_name
        
        if correction_mode == CorrectionMode.NotSpecified:
            eval_dict = None
        else:
            eval_dict = load_correction_dict(eval_dict_name, log)
            # note: use 'save_correction_dict' to re-generate a dictionary
        
        lang = get_lang_from_corpus_name(eval_dict_name)

        eval_params = {}
        eval_params["eval_mode"] = eval_mode
        eval_params["misspelling_rate"] = misspelling_rate
        eval_params["misspell_mode"] = misspell_mode
        eval_params["char_vocab"] = char_vocab
        eval_params["lut"] = lut
        eval_params["cmx"] = cmx
        eval_params["typos"] = typos
        eval_params["correction_mode"] = correction_mode
        eval_params["lang"] = lang
        eval_params["dictionary"] = eval_dict

        with torch.no_grad():
            eval_loss = 0

            batch_no: int = 0

            metric = Metric("Evaluation")

            lines: List[str] = []

            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None

            for batch in data_loader:
                batch_no += 1

                with torch.no_grad():
                    features = self.forward(batch, eval_params)
                    loss = self._calculate_loss(features, batch)
                    tags, _ = self._obtain_labels(
                        feature=features,
                        batch_sentences=batch,
                        transitions=transitions,
                        get_all_tags=False,
                    )

                eval_loss += loss

                for (sentence, sent_tags) in zip(batch, tags):
                    for (token, tag) in zip(sentence.tokens, sent_tags):
                        token: Token = token
                        token.add_tag("predicted", tag.value, tag.score)

                        # append both to file for evaluation
                        eval_line = "{} {} {} {}\n".format(
                            token.text,
                            token.get_tag(self.tag_type).value,
                            tag.value,
                            tag.score,
                        )
                        lines.append(eval_line)
                    lines.append("\n")

                for sentence in batch:
                    # make list of gold tags
                    gold_tags = [
                        (tag.tag, tag.text) for tag in sentence.get_spans(self.tag_type)
                    ]
                    # make list of predicted tags
                    predicted_tags = [
                        (tag.tag, tag.text) for tag in sentence.get_spans("predicted")
                    ]

                    # check for true positives, false positives and false negatives
                    for tag, prediction in predicted_tags:
                        if (tag, prediction) in gold_tags:
                            metric.add_tp(tag)
                        else:
                            metric.add_fp(tag)

                    for tag, gold in gold_tags:
                        if (tag, gold) not in predicted_tags:
                            metric.add_fn(tag)
                        else:
                            metric.add_tn(tag)

                store_embeddings(batch, embeddings_storage_mode)

            eval_loss /= batch_no

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy():.4f} - f1-score {metric.micro_avg_f_score():.4f}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy():.4f} - f1-score {metric.macro_avg_f_score():.4f}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            if evaluation_metric == EvaluationMetric.MICRO_F1_SCORE:
                main_score = metric.micro_avg_f_score()
            elif evaluation_metric == EvaluationMetric.MACRO_F1_SCORE:
                main_score = metric.macro_avg_f_score()
            elif evaluation_metric == EvaluationMetric.MICRO_ACCURACY:
                main_score = metric.micro_avg_accuracy()
            elif evaluation_metric == EvaluationMetric.MACRO_ACCURACY:
                main_score = metric.macro_avg_accuracy()
            elif evaluation_metric == EvaluationMetric.MEAN_SQUARED_ERROR:
                main_score = metric.mean_squared_error()
            else:
                log.error(f"unknown evaluation metric: {evaluation_metric}")

            result = Result(
                main_score=main_score,
                log_line=f"{metric.precision():.4f}\t{metric.recall():.4f}\t{main_score:.4f}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            return result, eval_loss

    def forward_loss(
        self, data_points: Union[List[Sentence], Sentence], sort=True, params: dict = {}
    ) -> (torch.tensor, dict):

        verbose = params.get("verbose", False)        
        char_vocab = params.get("char_vocab", {})
        cmx = params.get("cmx", {})
        lut = params.get("lut", {})
        typos = params.get("typos", {})
        translator = params.get("translator", None)
        opt = params.get("opt", None)
        translation_mode = params.get("translation_mode", Seq2SeqMode.ErrorGenerationTok)
        mean_tokens_per_batch = params.get("mean_tokens_per_batch", 1.0)
        perturbed_train_data = params.get("perturbed_train_data", None)

        train_info = dict()

        self.zero_grad()
      
        if self.train_mode == TrainingMode.Combined:
            loss, train_info = self._forward_loss_combined(data_points, cmx=cmx, lut=lut,
                typos=typos, char_vocab=char_vocab, mean_tokens_per_batch=mean_tokens_per_batch,
                translator=translator, opt=opt, translation_mode=translation_mode,
                perturbed_train_data=perturbed_train_data, verbose=verbose)
        else:
            raise Exception("Training mode '{}' is not supported!".format(self.train_mode))
        
        return loss, train_info

    def _forward_loss_combined(
        self, sentences: Union[List[Sentence], Sentence], char_vocab: dict, lut: dict = {}, cmx: np.array = None,
        typos: dict = {}, translator = None, opt = None, translation_mode = Seq2SeqMode.ErrorGenerationTok, 
        mean_tokens_per_batch = 1.0, perturbed_train_data = None, verbose: bool = False
    ) -> (torch.tensor, dict):
        """
        Implements the standard, the stability (alpha> 0) or the data augmentation (beta > 0) objective.        
        * Data augmentation objective - returns the auxiliary loss as the sum of standard objectives 
        calculated on the original and the perturbed samples.
        * Stability objective: L_stab(x,x') = -sum_j(P(yj|x)*log(P(yj|x')))
        The output loss is the sum of the standard loss and the similarity objective.
        """

        losses = list()
        train_info = dict()

        # calculate embeddings for the clean (original) sentences
        embeddings_clean, lengths_clean = self._embed_sentences(sentences)
        
        # generate misspelled sentences and calculate embeddings for them
        if self.alpha > 0.0 or self.beta > 0.0:

            if perturbed_train_data != None:
                
                misspelled_sentences = list()
                for sentence in sentences:
                    
                    candidates = perturbed_train_data[sentence.to_tokenized_string()]
                    if len(candidates) > 0:
                        if verbose and len(candidates) > 1:
                            log.warning(f"sentence: '{sentence.to_tokenized_string()}' has {len(candidates)} noisy candidates!")

                        random_candidate = random.choice(candidates)
                        misspelled_sentences.append(random_candidate)
                        
                    else:
                        log.error(f"len(candidates) == 0")
                        exit(-1)
            else:                
                misspelled_sentences, _ = noise_sentences(sentences, self.misspell_mode, log, noise_level=self.misspelling_rate_train, 
                    char_vocab=char_vocab, cmx=cmx, lut=lut, typos=typos, translator=translator, opt=opt,
                    translation_mode=translation_mode, verbose=verbose)

            clear_embeddings(misspelled_sentences)

            embeddings_misspell, lengths_misspell = self._embed_sentences(misspelled_sentences)
                            
            if not check_embeddings(sentences, misspelled_sentences, embeddings_clean, embeddings_misspell):
                log.warning("WARNING: embedding of the misspelled text may be invalid!")
        
        # BASE target objective
        outputs_clean, features_clean = self._forward(embeddings_clean, lengths_clean)
        loss_base = self._calculate_loss(outputs_clean, sentences)        
        train_info.update({ 'loss_base': loss_base })
        losses.append(loss_base)

        # only for data augmentation and stability objectives
        if self.alpha > 0.0 or self.beta > 0.0:
            outputs_misspell, features_misspell = self._forward(embeddings_misspell, lengths_misspell)

        # DATA AUGMENTATION objective
        if self.alpha > 0.0:
            loss_misspell = self.alpha * self._calculate_loss(outputs_misspell, misspelled_sentences)    
            train_info.update({ 'loss_misspell': loss_misspell })
            losses.append(loss_misspell)
        
        # clean-up
        if self.alpha > 0.0 or self.beta > 0.0:
            del misspelled_sentences

        # STABILITY objective
        if self.beta > 0.0:
            # https://pytorch.org/docs/master/nn.html?highlight=kldiv#torch.nn.KLDivLoss
            # the input given is expected to contain log-probabilities, the targets are given as probabilities 
            # (i.e. without taking the logarithm). Input: (N,C)(N,C)(N,C) where C = number of classes, or 
            # (N,C,d1,d2,...,dK) with K≥1, K in the case of K-dimensional loss.
            target_distrib = F.softmax(outputs_clean, dim=2).transpose(1, 2).detach()
            input_log_distrib = F.log_softmax(outputs_misspell, dim=2).transpose(1, 2)#.detach()

            loss_stability = F.kl_div(input_log_distrib, target_distrib, reduction='none').transpose(2, 1)            
            loss_sum = get_masked_sum(loss_stability, lengths_clean)
            # loss_mean = self.beta * get_per_batch_mean(loss_sum, lengths)            
            loss_mean = self.beta * mean_tokens_per_batch * get_per_token_mean(loss_sum, lengths_clean)
                
            train_info.update({ 'loss_kldiv': loss_mean })            

            losses.append(loss_mean)     
            
        return sum(losses), train_info

    def _embed_sentences(self, sentences: List[Sentence]) -> (torch.tensor, List[int]):
                        
        self.embeddings.embed(sentences)        

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                ]
                all_embs.append(t)

        embeddings = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        return embeddings, lengths

    def _forward(self, embeddings: torch.tensor, lengths: List[int]):

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        if self.use_dropout > 0.0:
            embeddings = self.dropout(embeddings)
        if self.use_word_dropout > 0.0:
            embeddings = self.word_dropout(embeddings)
        if self.use_locked_dropout > 0.0:
            embeddings = self.locked_dropout(embeddings)

        if self.relearn_embeddings:
            embeddings = self.embedding2nn(embeddings)

        if self.use_rnn:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, enforce_sorted=False, batch_first=True
            )

            # if initial hidden state is trainable, use this state
            if self.train_initial_hidden_state:
                initial_hidden_state = [
                    self.lstm_init_h.unsqueeze(1).repeat(1, len(lengths), 1),
                    self.lstm_init_c.unsqueeze(1).repeat(1, len(lengths), 1),
                ]
                rnn_output, hidden = self.rnn(packed, initial_hidden_state)
            else:
                rnn_output, hidden = self.rnn(packed)

            sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )

            if self.use_dropout > 0.0:
                sentence_tensor = self.dropout(sentence_tensor)
            # word dropout only before LSTM - TODO: more experimentation needed
            # if self.use_word_dropout > 0.0:
            #     sentence_tensor = self.word_dropout(sentence_tensor)
            if self.use_locked_dropout > 0.0:
                sentence_tensor = self.locked_dropout(sentence_tensor)
        
        outputs = self.linear(sentence_tensor)
        
        return outputs, sentence_tensor

    def forward(self, sentences: List[Sentence], params: dict = {}):
        
        verbose = params.get("verbose", False)                
        eval_mode = params.get("eval_mode", EvalMode.Standard)
        misspell_mode = params.get("misspell_mode", MisspellingMode.Random)
        misspelling_rate = params.get("misspelling_rate", 0.0)
        char_vocab = params.get("char_vocab", {})
        lut = params.get("lut", {})
        cmx = params.get("cmx", {})
        typos = params.get("typos", {})
        correction_mode = params.get("correction_mode", None)
        dictionary = params.get("dictionary", None)
        lang = params.get("lang", None)

        from robust_ner.hunspell import init_hunspell
        spell_check = init_hunspell(lang, dictionary=dictionary) if correction_mode == CorrectionMode.Hunspell else None
        
        self.zero_grad()

        if eval_mode is EvalMode.Standard:
            outputs = self._forward_standard(sentences, correction_mode, spell_check, dictionary)
        elif eval_mode is EvalMode.Misspellings:
            outputs = self._forward_misspelled(sentences, misspelling_rate=misspelling_rate, misspell_mode=misspell_mode,
                char_vocab=char_vocab, lut=lut, cmx=cmx, typos=typos, correction_mode=correction_mode, spell_check=spell_check, 
                dictionary=dictionary, verbose=verbose)
        else:
            raise Exception("Evaluation mode '{}' is not supported!".format(eval_mode))
                
        return outputs

    def _forward_standard(self, sentences: List[Sentence], correction_mode=CorrectionMode.NotSpecified, spell_check=None, dictionary=None):
    
        if correction_mode != CorrectionMode.NotSpecified:
            
            from robust_ner.spellcheck import correct_sentences
            corrected_sentences = correct_sentences(correction_mode, sentences, spell_check, dictionary)
            clear_embeddings(corrected_sentences)
             
            embeddings, lengths = self._embed_sentences(corrected_sentences)
        else:
            embeddings, lengths = self._embed_sentences(sentences)
        
        outputs, _ = self._forward(embeddings, lengths)        
        
        return outputs

    def _forward_misspelled(
        self, sentences: Union[List[Sentence], Sentence], misspelling_rate: float, misspell_mode: MisspellingMode, char_vocab: set, 
        cmx: np.array, lut: dict, typos:dict, correction_mode=CorrectionMode.NotSpecified, spell_check=None, dictionary=None, verbose: bool=False
    ) -> (torch.tensor, dict):
        
        misspelled_sentences, _ = noise_sentences(sentences, misspell_mode, log, noise_level=misspelling_rate, 
            char_vocab=char_vocab, cmx=cmx, lut=lut, typos=typos, verbose=verbose)
        clear_embeddings(misspelled_sentences)

        outputs_misspell = self._forward_standard(misspelled_sentences, correction_mode, spell_check, dictionary)
        
        return outputs_misspell    

    