import torch
from torch.optim.sgd import SGD
import os
import sys, csv, random, logging
import numpy as np
import pickle

from robust_ner.enums import Seq2SeqMode, CorrectionMode

FIXED_RANDOM_SEEDS = False # default: False

if FIXED_RANDOM_SEEDS:
    random.seed(0)
    np.random.seed(0)

EXIT_SUCCESS=0
EXIT_FAILURE=-1

def _make_lm_dict(corpus_dir, mappings_path):

    # make an empty character dictionary
    from flair.data import Dictionary
    char_dictionary: Dictionary = Dictionary()

    # counter object
    import collections
    counter = collections.Counter()

    processed = 0

    from pathlib import Path
    files = list(Path(corpus_dir).rglob('*.txt'))

    log.info(f"files: {files}")
    for file in files:
        log.info(file)

        with open(file, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                processed += 1            
                chars = list(line)
                tokens += len(chars)

                # Add chars to the dictionary
                counter.update(chars)

                # comment this line in to speed things up (if the corpus is too large)
                # if tokens > 50000000: break

        # break

    total_count = 0
    for letter, count in counter.most_common():
        total_count += count

    log.info(f"total_count: {total_count}")
    log.info(f"processed: {processed}")

    sum = 0
    idx = 0
    percentile_thresh = 1.0 - 1e-6
    
    for letter, count in counter.most_common():
        sum += count
        percentile = (sum / total_count)

        # comment this line in to use only top X percentile of chars, otherwise filter later
        if percentile > percentile_thresh:
            break

        char_dictionary.add_item(letter)
        idx += 1
        log.info('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))

    log.info(char_dictionary.item2idx)

    import pickle
    with open(mappings_path, 'wb') as f:
        mappings = {
            'idx2item': char_dictionary.idx2item,
            'item2idx': char_dictionary.item2idx
        }
        pickle.dump(mappings, f)

    log.info(f"Generated mappings: {mappings_path}.")

def train_lm(corpus_name, lm_model_dir, lm_type="", hidden_size=1024, mini_batch_size=100, max_epochs=10000,
    learning_rate=20, patience=50, anneal_factor=0.25, sequence_length=250, num_layers=1):

    if lm_type == 'forward':
        is_forward_lm = True
    elif lm_type == 'backward':
        is_forward_lm = False
    else:
        log.error("Unknown language model type")
        exit(EXIT_FAILURE)

    from flair.data import Dictionary
    from flair.models import LanguageModel
    from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

    data_dir = f'resources/corpora/'
    corpus_dir = f"{data_dir}{corpus_name}"

    if not os.path.exists(corpus_dir):
        log.error(f"Data directory '{corpus_dir}' does not exists!")
        exit(EXIT_FAILURE)

    mappings_path = os.path.join(corpus_dir, "chars.bin")

    if not os.path.exists(mappings_path):
        _make_lm_dict(corpus_dir, mappings_path)

    log.info(f"Loading mappings from '{mappings_path}'")
    dictionary = Dictionary.load_from_file(mappings_path)

    ## load the default character dictionary - len(dictionary) == 275 chars
    #dictionary: Dictionary = Dictionary.load('chars')
    
    # get your corpus, process forward and at the character level
    # alanakbik used 1/500 of the data of the corpus for test and validation for German
    corpus = TextCorpus(corpus_dir, dictionary, is_forward_lm, character_level=True, random_case_flip=True, document_delimiter='\n')

    # instantiate your language model, set hidden size and number of layers
    # hidden_size: 1024 or 2048
    # embedding_size: character embedding size (see language_model.py line:40)
    language_model = LanguageModel(dictionary, is_forward_lm, hidden_size=hidden_size, nlayers=num_layers, embedding_size=100)

    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus)

    # patience: set as a half of the number of training splits
    trainer.train(lm_model_dir, sequence_length=sequence_length, mini_batch_size=mini_batch_size, max_epochs=max_epochs,
        learning_rate=learning_rate, anneal_factor=anneal_factor, patience=patience, clip=0.25, checkpoint=False, grow_to_sequence_length=0, 
        num_workers=2, use_amp=False, amp_opt_level="O1")

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

def get_storage_mode(embeddings_storage_mode, embeddings_in_memory = False, device = None):
    """
    Determines the embeddings storage mode.
    """
    mode = embeddings_storage_mode

    if embeddings_storage_mode == "auto":        
        if embeddings_in_memory:
            if device.startswith("cpu"):
                mode = "cpu"
            elif device.startswith("cuda"):
                mode = "gpu"
            else:
                mode = "none"
        else:
            mode = "none"

    log.info(f"storage_mode: {mode}")
    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))
    
    return mode

def get_eval_metric(tag_type):
    """
    Gets a metric used for evaluation.
    """

    from flair.training_utils import EvaluationMetric

    if tag_type in ["ner"]:
        return EvaluationMetric.MICRO_F1_SCORE
    elif tag_type in ["pos", "upos"]:
        return EvaluationMetric.MICRO_ACCURACY
    else:
        log.error(f"cannot decide which evaluation metric to choose for the following tag type: {tag_type}.")
        exit(EXIT_FAILURE)

def evaluate(model_path, corpus, tag_type, mini_batch_size=32, misspelling_rate=0.0, 
             cmx_file="", typos_file="", correction_mode=CorrectionMode.NotSpecified, 
             dict_name=None, device = None, embeddings_storage_mode="auto"):
    """
    Evaluates the model on the test set of the given corpus. 
    Appends the results to the eval.txt file in the model's directory.

    Parameters:    
        model_path (str): path to the model to be evaluated
        corpus (ColumnCorpus): loaded corpus
        mini_batch_size (int): size of batches used by the evaluation function
        misspelling_rate (float): misspelling rate (used in case of 'random' misspelling mode)
        cmx_file (str): confusion matrix file (used in case of 'confusion matrix' misspelling mode)
        typos_file (str): file with typos (used in case of 'typos' misspelling mode)
        correction_mode (CorrectionMode): text correction mode
        dict_name (str): dictionary name for correction.
        device (str): name of a device to compute on.
        embedding_storage_mode (str): storage mode.
    """
    from robust_ner.enums import EvalMode, MisspellingMode

    if cmx_file:
        eval_mode = EvalMode.Misspellings
        misspell_mode = MisspellingMode.ConfusionMatrixBased
    elif typos_file:
        eval_mode = EvalMode.Misspellings
        misspell_mode = MisspellingMode.Typos
    elif misspelling_rate > 0.0:
        eval_mode = EvalMode.Misspellings
        misspell_mode = MisspellingMode.Random    
    else:
        eval_mode = EvalMode.Standard
        misspell_mode = MisspellingMode.Random

    # load the tagger model 
    from flair_ext.models import NATSequenceTagger
    tagger = NATSequenceTagger.load(model_path)
  
    eval_data = corpus.test

    from robust_ner.noise import make_char_vocab
    from robust_ner.confusion_matrix import load_confusion_matrix, filter_cmx
    from robust_ner.typos import load_typos
    
    char_vocab = make_char_vocab(eval_data)

    cmx, lut, typos = None, {}, {}

    # initialize resources used for evaluation
    if misspell_mode == MisspellingMode.ConfusionMatrixBased:
        cmx, lut = load_confusion_matrix(cmx_file)
        cmx, lut = filter_cmx(cmx, lut, char_vocab)
    elif misspell_mode == MisspellingMode.Typos:
        typos = load_typos(typos_file, char_vocab, False)        

    # fixed parameters
    num_workers = 8

    from flair.datasets import DataLoader

    embeddings_in_memory = True
    embeddings_storage_mode = get_storage_mode(embeddings_storage_mode, embeddings_in_memory, device)
    evaluation_metric = get_eval_metric(tag_type)

    # evaluate the model
    result, loss = tagger.evaluate(
        DataLoader(
            eval_data,
            batch_size=mini_batch_size,
            num_workers=num_workers,
        ),
        embeddings_storage_mode=embeddings_storage_mode,                 
        eval_mode=eval_mode, misspell_mode=misspell_mode, misspelling_rate=misspelling_rate,  
        char_vocab=char_vocab, cmx=cmx, lut=lut, typos=typos, correction_mode=correction_mode, 
        eval_dict_name=dict_name, evaluation_metric=evaluation_metric)

    # append the evaluation results to a file
    model_dir = os.path.dirname(model_path)
    eval_txt = os.path.join(model_dir, "eval.txt")

    with open(eval_txt, "a") as f:
        
        f.write(f"eval_mode: {eval_mode}\n")
        f.write(f"correction_mode: {correction_mode}\n")

        if eval_mode == EvalMode.Misspellings:
            f.write(f"misspell_mode: {misspell_mode}\n")            
            if misspell_mode == MisspellingMode.Random:
                f.write(f"misspelling_rate: {misspelling_rate}\n")
            elif misspell_mode == MisspellingMode.ConfusionMatrixBased:
                f.write(f"cmx_file: {cmx_file}\n")
            elif misspell_mode == MisspellingMode.Typos:
                f.write(f"typos_file: {typos_file}\n")

        f.write(f"Loss: {loss:.6} {result.detailed_results}\n")
        f.write("-" * 100 + "\n")

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

def train_tagger(model_dir, corpus, corpus_name, tag_type, embedding_type, train_mode, alpha=0.0, beta=0.0,
    misspelling_rate=0.0, cmx_file="", typos_file="", num_hidden=256, learning_rate=0.1, mini_batch_size=32, device=None,
    max_epochs=100, train_with_dev=False, save_checkpoint=False, valid_with_misspellings=True,
    embeddings_storage_mode="auto", use_amp=False, num_layers=1, errgen_model="", errgen_mode=Seq2SeqMode.ErrorGenerationTok,
    errgen_temp=1.0, errgen_topk=-1, errgen_nbest=5, errgen_beam_size=10):
    """
    Trains a tagger model from scratch.

    Parameters:
        model_dir (str): output model path
        corpus (ColumnCorpus): loaded corpus
        corpus_name (str): name of the corpus used to load proper embeddings
        tag_type (str): type of the tag to train
        embedding_type (str): type of embeddings (e.g. flair, elmo, bert, word+char)
        train_mode (TrainingMode): training mode
        alpha (float): primary auxiliary loss weighting factor
        beta (float): secondary auxiliary loss weighting factor
        misspelling_rate (float): misspelling rate (used in case of 'random' misspelling mode)    
        cmx_file (float): a confusion matrix file (used in case of 'confusion matrix' misspelling mode)
        typos_file (float): a typos file (used in case of 'typos' misspelling mode)
        num_hidden (int): number of hidden layers of the tagger's LSTM 
        learning_rate (float): initial learning rate
        mini_batch_size (int): the size of batches used by the evaluation function
        device (str): name of the device to compute on
        max_epochs (int): maximum number of epochs to run
        train_with_dev (bool): train using the development set
        save_checkpoint (bool): save checkpoint files
        valid_with_misspellings (bool): use validation with misspelling as additional measure
        embedding_storage_mode (str): storage mode
        use_amp (bool): use mixed-precision training
        num_layers (int): number of RNN layers
        errgen_model (str): model path for seq2seq error generation
        errgen_mode (Seq2SeqMode): seq2seq error generation mode
        errgen_temp (float): temperature for seq2seq error generation
        errgen_topk (int): number of sampling candidates for seq2seq error generation
        errgen_nbest (int): number of beams for seq2seq error generation
        errgen_beam_size (int): beam size for seq2seq error generation
    """
    
    # load embeddings
    embeddings, embeddings_in_memory = init_embeddings(corpus_name, embedding_type=embedding_type)
        
    # fixed parameters
    use_crf = True
    dropout, word_dropout, locked_dropout = 0.0, 0.05, 0.5
    optimizer = SGD

    # create the tagger model
    from flair_ext.models import NATSequenceTagger
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    tagger: NATSequenceTagger = NATSequenceTagger(hidden_size=num_hidden, embeddings=embeddings,
        tag_dictionary=tag_dictionary, tag_type=tag_type, use_crf=use_crf, use_rnn=num_layers>0, 
        rnn_layers=num_layers, dropout=dropout, word_dropout=word_dropout, locked_dropout=locked_dropout,
        train_mode=train_mode, alpha=alpha, beta=beta, misspelling_rate_train=misspelling_rate, 
        cmx_file=cmx_file, typos_file=typos_file, errgen_model=errgen_model, errgen_mode=errgen_mode, 
        errgen_temp=errgen_temp, errgen_topk=errgen_topk, errgen_nbest=errgen_nbest, errgen_beam_size=errgen_beam_size)

    from robust_ner.enums import TrainingMode

    # fixed parameters
    anneal_factor = 0.5
    patience = 3
    anneal_with_restarts = False
    num_workers = 8

    embeddings_storage_mode = get_storage_mode(embeddings_storage_mode, embeddings_in_memory, device)
    
    # train the model
    from flair_ext.trainers import ParameterizedModelTrainer
    trainer: ParameterizedModelTrainer = ParameterizedModelTrainer(model=tagger, corpus=corpus, optimizer=optimizer, epoch=0, use_tensorboard=False)
    trainer.train(model_dir, learning_rate=learning_rate, mini_batch_size=mini_batch_size, max_epochs=max_epochs,
        anneal_factor=anneal_factor, patience=patience, min_learning_rate=0.0001, train_with_dev=train_with_dev,
        monitor_train=False, monitor_test=False, embeddings_storage_mode=embeddings_storage_mode, checkpoint=save_checkpoint, 
        save_final_model=True, anneal_with_restarts=anneal_with_restarts, batch_growth_annealing=False, shuffle=True, 
        param_selection_mode=False, num_workers=num_workers, sampler=None, use_amp=use_amp, amp_opt_level="O1",
        eval_on_train_fraction=0.0, eval_on_train_shuffle=False,
        valid_with_misspellings=valid_with_misspellings, corpus_name=corpus_name)
    
    plot_training_curves(model_dir)

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

def fine_tune(model_dir, corpus, checkpoint_name, train_mode, alpha=0.0, beta=0.0, misspelling_rate = 0.0, 
    cmx_file="", typos_file="", learning_rate=0.01, mini_batch_size=32, max_epochs=100, 
    train_with_dev=False, save_checkpoint=True, valid_with_misspellings=True, 
    use_amp=False, device=None, embeddings_storage_mode="auto"):
    """
    Fine-tunes an existing tagger model.

    Parameters:
        model_dir (str): output model path
        corpus (str): loaded corpus
        checkpoint_name (str): name of the checkpoint file
        train_mode (TrainingMode): training mode
        alpha (float): primary auxiliary loss weighting factor
        beta (float): secondary auxiliary loss weighting factor
        misspelling_rate (float): misspelling rate (used in case of 'random' misspelling mode)    
        cmx_file (str): a confusion matrix file (used in case of 'confusion matrix' misspelling mode)    
        typos_file (float): a typos file (used in case of 'typos' misspelling mode)
        learning_rate (float): initial learning rate
        mini_batch_size (int): the size of batches used by the evaluation function
        max_epochs (int): maximum number of epochs to run
        train_with_dev (bool): train using the development set
        save_checkpoint (bool): save checkpoint files
        valid_with_misspellings (bool): use validation with misspelling as additional measure
        use_amp (bool): use mixed-precision training
        device (str): name of the device to compute on
        embeddings_storage_mode (str): storage mode
    """
    
    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    
    from flair_ext.models import NATSequenceTagger

    # fixed parameters
    optimizer = SGD
    anneal_factor = 0.5
    patience = 3
    anneal_with_restarts = False
    num_workers = 8
    
    from flair_ext.trainers import ParameterizedModelTrainer        

    # load checkpoint file
    trainer: ParameterizedModelTrainer = ParameterizedModelTrainer.load_checkpoint(checkpoint_path, corpus)
    trainer.model.set_training_params(train_mode=train_mode, alpha=alpha, beta=beta, misspelling_rate_train=misspelling_rate, 
        cmx_file=cmx_file, typos_file=typos_file)
    trainer.optimizer = optimizer

    embeddings_in_memory = True # assuming that there is no character embeddings
    embeddings_storage_mode = embeddings_storage_mode = get_storage_mode(embeddings_storage_mode, embeddings_in_memory, device)

    # train the model
    trainer.train(model_dir, learning_rate=learning_rate, mini_batch_size=mini_batch_size, max_epochs=max_epochs,
        anneal_factor=anneal_factor, patience=patience, train_with_dev=train_with_dev, monitor_train=False, 
        embeddings_storage_mode=embeddings_storage_mode, checkpoint=save_checkpoint, anneal_with_restarts=anneal_with_restarts, 
        shuffle=True, param_selection_mode=False, num_workers=num_workers,
        valid_with_misspellings=valid_with_misspellings, use_amp=use_amp)

    plot_training_curves(model_dir)

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

def init_embeddings(corpus_name, embedding_type):
    """
    Initializes embeddings for a given corpus.

    Parameters:
        corpus_name (str): name of the corpus used to load proper embeddings
        embedding_type (str): type of embeddings (e.g. flair, elmo, bert, word+char)
    
    Returns:
        tuple(StackedEmbeddings, bool): loaded embeddings
    """
    
    from typing import List    
    from flair.embeddings import TokenEmbeddings, StackedEmbeddings
    from flair.embeddings import WordEmbeddings, CharacterEmbeddings
    from flair.embeddings import FlairEmbeddings, BertEmbeddings, ELMoEmbeddings    

    embedding_types: List[TokenEmbeddings] = []
    embeddings_in_memory = False
    
    if corpus_name.startswith('conll03_en'):
        if embedding_type == 'flair+glove':
            embedding_types.append(WordEmbeddings('glove'))
            embedding_types.append(FlairEmbeddings('news-forward', fine_tune=False))
            embedding_types.append(FlairEmbeddings('news-backward', fine_tune=False))
            embeddings_in_memory = True 
        elif embedding_type == 'flair':
            embedding_types.append(FlairEmbeddings('news-forward', fine_tune=False))
            embedding_types.append(FlairEmbeddings('news-backward', fine_tune=False))        
        elif embedding_type == 'bert':            
            embedding_types.append(BertEmbeddings(bert_model_or_path='bert-base-cased'))            
            #embedding_types.append(BertEmbeddings(bert_model_or_path='bert-large-cased'))
            embeddings_in_memory = True
        elif embedding_type == 'elmo':
            embedding_types.append(ELMoEmbeddings())
            embeddings_in_memory = True
        elif embedding_type == 'glove+char':
            # similar to Lample et al. (2016)
            embedding_types.append(WordEmbeddings('glove'))
            embedding_types.append(CharacterEmbeddings(char_embedding_dim=25, hidden_size_char=25))
            embeddings_in_memory = False # because it contains a char model (problem with deepcopy in noise_sentences)        
        elif embedding_type == 'myflair':
            embedding_types.append(FlairEmbeddings('resources/language_models/custom_forward/best-lm.pt', fine_tune=False))
            embedding_types.append(FlairEmbeddings('resources/language_models/custom_backward/best-lm.pt', fine_tune=False))
            embeddings_in_memory = True         
        elif embedding_type == 'myflair+glove':
            embedding_types.append(WordEmbeddings('glove'))
            embedding_types.append(FlairEmbeddings('resources/language_models/custom_forward/best-lm.pt', fine_tune=False))
            embedding_types.append(FlairEmbeddings('resources/language_models/custom_backward/best-lm.pt', fine_tune=False))
            embeddings_in_memory = True
        else:
            log.error(f"no settings for '{embedding_type}'!")
            exit(EXIT_FAILURE)

    elif corpus_name in ["conll03_de", "germeval"]:
        if embedding_type == 'flair+wiki':
            embedding_types.append(WordEmbeddings('de'))
            embedding_types.append(FlairEmbeddings('german-forward', fine_tune=False))
            embedding_types.append(FlairEmbeddings('german-backward', fine_tune=False))
            embeddings_in_memory = True
        elif embedding_type == 'wiki+char':
            # similar to Lample et al. (2016)
            embedding_types.append(WordEmbeddings('de'))
            embedding_types.append(CharacterEmbeddings(char_embedding_dim=25, hidden_size_char=25))
            embeddings_in_memory = False # because it contains a char model (problem with deepcopy in noise_sentences)
        else:
            log.error(f"no settings for '{embedding_type}'!")
            exit(EXIT_FAILURE)

    elif corpus_name.startswith("ud_en"):
        if embedding_type == 'flair+glove': # Yasunaga paper
            embedding_types.append(WordEmbeddings('glove'))
            embedding_types.append(FlairEmbeddings('news-forward', fine_tune=False))
            embedding_types.append(FlairEmbeddings('news-backward', fine_tune=False))
            embeddings_in_memory = True
        elif embedding_type == 'myflair':
            embedding_types.append(FlairEmbeddings('resources/language_models/custom_forward/best-lm.pt', fine_tune=False))
            embedding_types.append(FlairEmbeddings('resources/language_models/custom_backward/best-lm.pt', fine_tune=False))
            embeddings_in_memory = True        
        elif embedding_type == 'myflair+glove':
            embedding_types.append(WordEmbeddings('glove'))
            embedding_types.append(FlairEmbeddings('resources/language_models/custom_forward/best-lm.pt', fine_tune=False))
            embedding_types.append(FlairEmbeddings('resources/language_models/custom_backward/best-lm.pt', fine_tune=False))
            embeddings_in_memory = True
        else:
            log.error(f"no settings for '{embedding_type}'!")
            exit(EXIT_FAILURE)
    else:
        log.error(f"unknown corpus or embeddings '{corpus_name}'!")
        exit(EXIT_FAILURE)
        
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

    return embeddings, embeddings_in_memory

def load_corpus(corpus_name, col_idx, text_idx, downsample_perc=1.0, tag_type="",
                name_train=None, name_dev=None, name_test=None, verbose=False):
    """
    Loads a corpus with a given name.
    Optionally performs downsampling of the data.

    Parameters:
        corpus_name (str): name of the corpus used to load proper embeddings
        col_idx (int): index of the column's tag
        text_idx (int): index of the text's tag
        downsample_rate (float): downsample rate (1.0 = full corpus)
        name_train (str): name of a file containing the train set
        name_dev (str): name of a file containing the development set
        name_test (str): name of a file containing the test set
        verbose (bool): activates verbose messages
    
    Returns:
        ColumnCorpus: the loaded corpus
    """        
    
    from pathlib import Path

    data_dir = f'resources/tasks/'
        
    if corpus_name in ["conll03_en"]:
        from flair.datasets import CONLL_03
        if not tag_type:
            tag_type='ner'
        corpus = CONLL_03(base_path=Path(data_dir), tag_to_bioes=tag_type)
    elif corpus_name in ["conll03_de"]:
        from flair.datasets import CONLL_03_GERMAN
        if not tag_type:
            tag_type='ner'
        corpus = CONLL_03_GERMAN(base_path=Path(data_dir), tag_to_bioes=tag_type)
    elif corpus_name in ["germeval"]:
        from flair.datasets import GERMEVAL
        if not tag_type:
            tag_type='ner'
        corpus = GERMEVAL(base_path=Path(data_dir), tag_to_bioes=tag_type)
    elif corpus_name in ["ud_en"]:
        from flair.datasets import UD_ENGLISH
        if not tag_type:
            tag_type='upos'
        corpus = UD_ENGLISH(base_path=Path(data_dir))
    else:        
        corpus_dir = f"{data_dir}{corpus_name}"
        if not os.path.exists(corpus_dir):
            log.error(f"Data directory '{corpus_dir}' does not exists!")
            exit(EXIT_FAILURE)

        is_ud = corpus_name.startswith("ud_")
        
        if not tag_type:
            tag_type='upos' if is_ud else 'ner'

        tag_to_bioes = None if is_ud else tag_type

        columns = { text_idx: 'text', col_idx: tag_type }
        train_set = None if name_train is None else f'{name_train}'
        dev_set = None if name_dev is None else f'{name_dev}'
        test_set = None if name_test is None else f'{name_test}'

        from flair.datasets import ColumnCorpus        
        corpus: ColumnCorpus = ColumnCorpus(corpus_dir, columns, train_file=train_set, test_file=test_set, dev_file=dev_set,
            tag_to_bioes=tag_to_bioes)            

    if downsample_perc >= 0.0 and downsample_perc < 1.0:
        corpus.downsample(downsample_perc)

    if "_tess" in corpus_name:
        log.info(f"normalizing whitespaces")
        # replace all char(172) occurences with a whitespace
        from robust_ner.utils import _replace_chars_in_corpus
        _replace_chars_in_corpus(corpus, chr(172), ' ')
    
    if verbose:
        log.info(corpus.obtain_statistics(label_type=tag_type))
    
    log.info(f"tag_type: {tag_type}")
    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

    return corpus, tag_type

def plot_training_curves(model_dir):
    """
    Plots training curves given the model directory.

    Parameters:
        model_dir (str): model's directory
    """
    
    from flair_ext.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_training_curves('{}/loss.tsv'.format(model_dir))
    plotter.plot_weights('{}/weights.txt'.format(model_dir))

    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))

def extract_noisy_corpus(corpus_name):

    dir = f"results/generated/{corpus_name}/"
    input_path = os.path.join(dir, f"{corpus_name}_pairs_norm.txt")
    
    from robust_ner.seq2seq import extract_noisy_corpus
    extract_noisy_corpus(input_path, log, max_lines=-1, split_num_lines=int(3e5))

def onmt(onmt_path, corpus_name):
    """
    Allows to split the data, perform data pre-processing, training or inference for a seq2seq method.
    """
    
    # Note: adjust these flags and modes (see below) to set the seq2seq processing mode!
    do_split_data = True
    do_preprocess = True
    do_train = True

    # mode = Seq2SeqMode.ErrorGenerationCh
    mode = Seq2SeqMode.ErrorGenerationTok

    # mode = Seq2SeqMode.ErrorCorrectionTok

    max_lines_total = 1e7 # 1e2, 1e3, 1e4, 1e5, 1e6, 1e7

    from pysia.datasplit import split_train_valid
    from pysia.nmt import onmt_preprocess, onmt_train
    from pysia.utils import get_max_lines_alias

    max_lines_str = get_max_lines_alias(max_lines_total)
    
    dir = f"results/generated/{corpus_name}/"
    input_path = os.path.join(dir, f"{corpus_name}_pairs_norm.txt")
    data_path = os.path.join(dir, f"preprocessed_{mode.value}_{max_lines_str}", f"{corpus_name}")
    model_path = os.path.join(dir, f"model_{mode.value}_{max_lines_str}", f"{corpus_name}")

    if do_split_data:
        split_train_valid(input_path, mode, log, max_lines_total=int(max_lines_total))

    if do_preprocess:
        if mode in [Seq2SeqMode.ErrorGenerationTok, Seq2SeqMode.ErrorCorrectionTok]:
            shard_size = 2e5 # ~108MB, 1249 shards
            num_threads = 12
            max_seq_length = 1000
        elif mode in [Seq2SeqMode.ErrorGenerationCh]:
            shard_size = 2e4 # ~65MB files, 497 shards
            num_threads = 16
            max_seq_length = 1000

        onmt_preprocess(onmt_path, input_path, data_path, mode, log, shard_size=int(shard_size), num_threads=num_threads, 
                max_seq_len=max_seq_length, max_lines_total=int(max_lines_total))
        
    if do_train:
        if mode in [Seq2SeqMode.ErrorGenerationTok]:
            train_steps = 4e5 # 1e7 for 10M iter=3'907'466
            valid_steps = 5e4 # 5e5 for 10M 
            start_decay_steps = 5e4 # 5e5 for 10M
            decay_steps = 5e4 # 5e5 for 10M
        elif mode in [Seq2SeqMode.ErrorGenerationCh]:
            train_steps = 16e3 # 1e7 for 10M iter=156'421
            valid_steps = 2e3 # 2e4 for 10M
            start_decay_steps = 2e3 # 2e4 for 10M
            decay_steps = 2e3 # 2e4 for 10M
        elif mode in [Seq2SeqMode.ErrorCorrectionTok]:
            train_steps = 1e7 #1e5
            valid_steps = 1e5 #2e4
            start_decay_steps = 5e5 #2e4
            decay_steps = 5e5 #2e4

        onmt_train(onmt_path, data_path, model_path, mode, log, train_steps=int(train_steps), valid_steps=int(valid_steps), 
                start_decay_steps=int(start_decay_steps), decay_steps=int(decay_steps))
    
def _restore_noisy_dataset(dataset_name, clean_dataset, tag_type, dir):

    from pysia.algos import align_text
    from pysia.align import _bioes_to_bio

    input_file = os.path.join(dir, f"{dataset_name}_ops.txt")
    output_file = os.path.join(dir, f"{dataset_name}_restored.txt")

    tokens = [tok for sent in clean_dataset for tok in sent]
    
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        line_idx, token_idx = 0, 0
        input_line = f_in.readline()

        while input_line:

            input_line = input_line.strip()
            
            if not input_line:
                print("", file=f_out)
                input_line = f_in.readline()
                line_idx += 1
                continue

            elems = input_line.split(' ')

            if len(elems) < 1 or len(elems) > 2:
                log.error(f"[{line_idx}] invalid len(elems) ({len(elems)})")
                exit(EXIT_FAILURE)

            edit_ops = elems[0]
            changes = elems[1] if len(elems) == 2 else None

            clean_token = tokens[token_idx]

            tag_value = _bioes_to_bio(clean_token.get_tag(tag_type).value)

            clean_aligned = align_text(clean_token.text, edit_ops, "i", 172)
            noisy_token_text = ""

            idx_change = 0
            for i, (clean_ch, op) in enumerate(zip(clean_aligned, edit_ops)):
                if op == '-': # no-change
                    noisy_token_text += clean_ch
                elif op == 's' or op == 'i':
                    noisy_token_text += changes[idx_change]
                    idx_change += 1

            print(f"{noisy_token_text}\t{clean_token.text}\t{tag_value}", file=f_out)

            input_line = f_in.readline()
            line_idx += 1
            token_idx += 1

def restore_noisy_dataset(corpus_name, clean_corpus, tag_type):
    
    dir = f"resources/conversion/{corpus_name}/"

    _restore_noisy_dataset("train", clean_corpus.train, tag_type, dir)
    _restore_noisy_dataset("dev", clean_corpus.dev, tag_type, dir)
    _restore_noisy_dataset("test", clean_corpus.test, tag_type, dir)

def restore_noisy_datasets(corpus_name, clean_corpus, tag_type):

    restore_noisy_dataset(f"{corpus_name}_tess3_01", clean_corpus, tag_type)
    restore_noisy_dataset(f"{corpus_name}_tess4_01", clean_corpus, tag_type)
    restore_noisy_dataset(f"{corpus_name}_tess4_02", clean_corpus, tag_type)
    restore_noisy_dataset(f"{corpus_name}_tess4_03", clean_corpus, tag_type)
    restore_noisy_dataset(f"{corpus_name}_typos", clean_corpus, tag_type)

def _md5(path):

    import hashlib
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _check_noisy_dataset(res_path, ref_path):

    md5_res = _md5(res_path)

    with open(ref_path, 'r') as f:
        md5_ref = f.read().replace('\n', '')

    # log.info(f"{ref_path}: {md5_ref}")
    # log.info(f"{res_path}: {md5_res}")

    return md5_ref == md5_res

def check_noisy_dataset(dir):

    ok = True
    ok &= _check_noisy_dataset(os.path.join(dir, "train_restored.txt"), os.path.join(dir, "train.md5"))
    ok &= _check_noisy_dataset(os.path.join(dir, "dev_restored.txt"), os.path.join(dir, "dev.md5"))
    ok &= _check_noisy_dataset(os.path.join(dir, "test_restored.txt"), os.path.join(dir, "test.md5"))
    
    return ok

def check_noisy_datasets(corpus_name):

    dir = f"resources/conversion/"

    ok = check_noisy_dataset(os.path.join(dir, f"{corpus_name}_tess3_01"))
    log.info(f"tess3_01: {ok}")

    ok = check_noisy_dataset(os.path.join(dir, f"{corpus_name}_tess4_01"))
    log.info(f"tess4_01: {ok}")

    ok = check_noisy_dataset(os.path.join(dir, f"{corpus_name}_tess4_02"))
    log.info(f"tess4_02: {ok}")

    ok = check_noisy_dataset(os.path.join(dir, f"{corpus_name}_tess4_03"))
    log.info(f"tess4_03: {ok}")

    ok = check_noisy_dataset(os.path.join(dir, f"{corpus_name}_typos"))
    log.info(f"typos: {ok}")

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        parsed arguments
    """
        
    import argparse
    from robust_ner.enums import TrainingMode
    from robust_ner.seq2seq import Seq2SeqMode

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', dest='mode', type=str, help="execution mode",
        choices=['train', 'train_lm', 'tune', 'eval', 'sent_gen', 'sent_gen_txt', 'noisy_crp', 'onmt', 'ds_restore', 'ds_check'],
        default='', required=True)
    parser.add_argument('--corpus', dest='corpus', type=str, help="data set to use", default='', required=False)
    parser.add_argument('--text_corpus', dest='text_corpus', type=str, help="text corpus to use", default='', required=False)
    parser.add_argument('--type', dest='embedding_type', type=str, help="embedding type")
    parser.add_argument('--model', dest='model', type=str, help="model path", default='', required=False)
    parser.add_argument('--col_idx', dest='col_idx', type=int, help="ner tag column index", default=3)
    parser.add_argument('--text_idx', dest='text_idx', type=int, help="text tag column index", default=0)
    parser.add_argument('--device', dest='device', type=str, help="device to use", default='cuda:0')
    parser.add_argument('--downsample', dest='downsample', type=float, help="downsample rate", default='1.0')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, help="checkpoint file", default='best-model.pt')
    parser.add_argument('--alpha', dest='alpha', type=float, help="primary auxiliary loss weight factor", default=0.0)
    parser.add_argument('--beta', dest='beta', type=float, help="secondary auxiliary loss weight factor", default=0.0)
    parser.add_argument('--misspelling_rate', dest='misspelling_rate', type=float, 
        help="misspellings rate used during training", default=0.1)
    parser.add_argument('--train_mode', dest='train_mode', type=TrainingMode, help="training mode", 
        choices=list(TrainingMode), default=TrainingMode.Combined)
    parser.add_argument('--verbose', dest='verbose', action='store_true', help="print verbose messages", default=False)
    parser.add_argument('--num_hidden', dest='num_hidden', type=int, help="the number of hidden units of a tagger LSTM", 
        default=256)
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, help="max number of epochs to train", default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help="mini batch size", default=32)
    parser.add_argument('--lr', dest='learning_rate', type=float, help="initial learning rate", default=0.1)
    parser.add_argument('--train_with_dev', dest='train_with_dev', action='store_true', 
        help="train using development data set", default=False)
    parser.add_argument('--cmx_file', dest='cmx_file', type=str, help="confusion matrix file for training or evaluation", 
        default='')
    parser.add_argument('--typos_file', dest='typos_file', nargs=argparse.ZERO_OR_MORE, type=str, help="typos file for evaluation", default=[])        
    parser.add_argument('--correction_module', dest='correction_module', type=str, 
        help="name of a correction module", default='')
    parser.add_argument('--no_valid_misspell', dest='no_valid_with_misspellings', action='store_true', 
        help="turns off the validation component that uses perturbed data", default=False)
    parser.add_argument('--use_amp', dest='use_amp', action='store_true', 
        help="use mixed-precision training", default=False)
    parser.add_argument('--save_checkpoint', dest='save_checkpoint', action='store_true', 
        help="save a checkpoint during training", default=False)
    parser.add_argument('--storage_mode', dest='embeddings_storage_mode', type=str, 
        help="embedding storage mode", choices=['auto', 'gpu', 'cpu', 'none'], default='auto')
    parser.add_argument('--num_layers', dest='num_layers', type=int, help="The number of RNN layers", default=1)
    parser.add_argument('--seek_file', dest='seek_file', type=str, help="file name to start with when generating paired data", default="")
    parser.add_argument('--seek_line', dest='seek_line', type=int, help="line number to seek when generating paired data", default=0)
    parser.add_argument('--errgen_model', dest='errgen_model', type=str, help="error generator model name", default="")
    parser.add_argument('--errgen_mode', dest='errgen_mode', type=Seq2SeqMode, help="error generation mode", 
        choices=list(Seq2SeqMode), default=Seq2SeqMode.ErrorGenerationTok)
    parser.add_argument('--errgen_temp', dest='errgen_temp', type=float, help="error generation random sampling temperature", default=1.0)
    parser.add_argument('--errgen_topk', dest='errgen_topk', type=int, 
        help="error generation number of candidates for random sampling (-1 == all)", default=-1)
    parser.add_argument('--errgen_nbest', dest='errgen_nbest', type=int, help="error generation number of candidates for beam search", 
        default=5)
    parser.add_argument('--errgen_beam_size', dest='errgen_beam_size', type=int, help="error generation beam width", default=10)
        
    ## LM training
    parser.add_argument('--lm_type', dest='lm_type', type=str, choices=["forward","backward"], default='', help="language model type")
    parser.add_argument('--patience', dest='patience', type=int, help="Patience factor", default=50)
    parser.add_argument('--anneal_factor', dest='anneal_factor', type=float, help="Anneal factor", default=0.25)    
    parser.add_argument('--sequence_length', dest='sequence_length', type=int, help="Truncated BPTT window length", default=250)
        
    args = parser.parse_args()

    log.info(args)

    import flair 

    if FIXED_RANDOM_SEEDS:
        torch.manual_seed(0)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    
    flair.device = torch.device(args.device)

    if args.col_idx < 0:
        log.error("invalid args.col_idx: '{}'".format(args.col_idx))
        exit(EXIT_FAILURE)
    
    if not 0.0 < args.downsample <= 1.0:
        log.error("invalud args.downsample: '{}'".format(args.downsample))
        exit(EXIT_FAILURE)
        
    log.info("'{}' function finished!".format(sys._getframe().f_code.co_name))
            
    return args

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    """
    Adopted from onmt.utils.logging.init_logger(..)
    """
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger("flair")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1000000, backupCount=10)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":

    log = init_logger()

    current_directory = os.path.dirname(os.path.abspath(__file__))

    # add the current directory to the system path to use functions from the robust_ner library
    # sys.path.append(current_directory)

    # add sub-folder containint the pysia library to the system path
    sys.path.append(os.path.join(current_directory, "pysia"))

    flair_path = os.path.join(current_directory, "flair")
    trdg_path = os.path.join(current_directory, "trdg")
    onmt_path = os.path.join(current_directory, "onmt")
    natas_path = os.path.join(current_directory, "natas")

    # add sub-folder containint the library paths to the system path    
    sys.path.append(flair_path)
    sys.path.append(trdg_path)    
    sys.path.append(natas_path)
    sys.path.append(onmt_path)

    _environ = dict(os.environ)  # or os.environ.copy()
    try:
        # parse command-line arguments
        args = parse_args()

        # os.environ["PYTHONPATH"] = onmt_path + os.pathsep + os.environ["PATH"]
        os.environ["PYTHONPATH"] = onmt_path
        
        model_name = args.model

        # if the model_name is not an absolute path - assume it is placed in the 'resources/taggers' sub-directory

        if os.path.isabs(model_name):
            model_dir = model_name
        elif args.mode == "train_lm":
            model_dir = os.path.join("resources/language_models", model_name)
        else:
            model_dir = os.path.join("resources/taggers", model_name)
    
        if type(model_dir) is str and model_name:
            from pathlib import Path
            base_path = Path(model_dir)
    
        # join the full model path
        model_path = os.path.join(model_dir, args.checkpoint)

        # if the given path does not exists, check whether it could be a built-in model
        if not os.path.exists(model_path) and model_name in ['ner', 'de-ner']:
            model_path = model_name
    
        if args.mode not in ['train_lm', 'sent_gen_txt', 'noisy_crp', 'onmt', 'ds_prep', 'ds_check']:
            # load the corpus
            corpus, tag_type = load_corpus(args.corpus, args.col_idx, args.text_idx, args.downsample, verbose=args.verbose)
        
        correction_mode = CorrectionMode.NotSpecified

        # optionaly, initialize the spell checker
        if args.correction_module == "hunspell":
            correction_mode = CorrectionMode.Hunspell
        elif args.correction_module == "natas":
            correction_mode = CorrectionMode.Natas        

        log.info(f"Correction-mode: {correction_mode.value}")

        if args.mode == 'train_lm':
            train_lm(args.text_corpus, model_dir, lm_type=args.lm_type, hidden_size=args.num_hidden, mini_batch_size=args.batch_size,
                max_epochs=args.max_epochs, learning_rate=args.learning_rate, patience=args.patience, anneal_factor=args.anneal_factor,
                sequence_length=args.sequence_length, num_layers=args.num_layers)
        elif args.mode == 'train':
            train_tagger(model_dir, corpus, corpus_name=args.corpus, tag_type=tag_type, embedding_type=args.embedding_type,            
                train_mode=args.train_mode, alpha=args.alpha, beta=args.beta, misspelling_rate=args.misspelling_rate, 
                cmx_file=args.cmx_file, typos_file=args.typos_file, num_hidden=args.num_hidden, max_epochs=args.max_epochs, 
                device=args.device, learning_rate=args.learning_rate, train_with_dev=args.train_with_dev, mini_batch_size=args.batch_size,
                valid_with_misspellings=not args.no_valid_with_misspellings, use_amp=args.use_amp, save_checkpoint=args.save_checkpoint, 
                embeddings_storage_mode=args.embeddings_storage_mode, num_layers=args.num_layers, errgen_model=args.errgen_model, 
                errgen_mode=args.errgen_mode, errgen_temp=args.errgen_temp,  errgen_topk=args.errgen_topk, errgen_nbest=args.errgen_nbest, 
                errgen_beam_size=args.errgen_beam_size)
        elif args.mode == 'tune':
            fine_tune(model_dir, corpus, args.checkpoint, train_mode=args.train_mode, alpha=args.alpha, beta=args.beta,  
                misspelling_rate=args.misspelling_rate, max_epochs=args.max_epochs, cmx_file=args.cmx_file, typos_file=args.typos_file,
                learning_rate=args.learning_rate, train_with_dev=args.train_with_dev, mini_batch_size=args.batch_size,
                valid_with_misspellings=not args.no_valid_with_misspellings, use_amp=args.use_amp, save_checkpoint=args.save_checkpoint, 
                device=args.device, embeddings_storage_mode=args.embeddings_storage_mode)
        elif args.mode == 'eval':
            evaluate(model_path, corpus, tag_type, misspelling_rate=args.misspelling_rate, cmx_file=args.cmx_file, typos_file=args.typos_file, 
                correction_mode=correction_mode, device=args.device, dict_name=args.corpus, embeddings_storage_mode=args.embeddings_storage_mode)
        elif args.mode == 'sent_gen':
            from pysia.align import generate_paired_sentences_from_seq_lab_corpus
            generate_paired_sentences_from_seq_lab_corpus(corpus, args.corpus, log, tag_type=tag_type, save_img=False)
        elif args.mode == 'sent_gen_txt':
            from pysia.align import generate_paired_sentences_from_text_files
            generate_paired_sentences_from_text_files(args.text_corpus, log, start_from_file=args.seek_file, start_from_line=args.seek_line, 
                chunk_size=1000, save_img=False)
        elif args.mode == 'onmt':
            onmt(onmt_path, args.text_corpus)
        elif args.mode == 'noisy_crp':
            extract_noisy_corpus(args.text_corpus)
        elif args.mode == "ds_restore":
            restore_noisy_datasets(args.corpus, corpus, tag_type)
        elif args.mode == "ds_check":
            check_noisy_datasets(args.corpus)
        else:
            print("unknown mode")
    
    finally:
        # restore the default env variables
        os.environ.clear()
        os.environ.update(_environ)
