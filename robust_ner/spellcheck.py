import pickle
import os

from robust_ner.enums import CorrectionMode
from robust_ner.hunspell import correct_text_with_hunspell
from robust_ner.natas import correct_text_with_natas


def get_lang_from_corpus_name(corpus_name):
    """
    Determines the language of a corpus based on its name.
    """

    if corpus_name.startswith("conll03_en") or corpus_name.startswith("ud_en"):
        return "en"
    elif corpus_name in ["conll03_de", "germeval"]:
        return "de"
    else:
        return None

def load_correction_dict(corpus_name, log):
    """
    Load a previously saved dictionary of a corpus.
    """

    dict_path = f"resources/dictionaries/{corpus_name}.pickle"        
    vocab = None
    if os.path.exists(dict_path):        
        log.info(f"Loading corpus dictionary from '{dict_path}'.")
        with open(dict_path, 'rb') as handle:
            vocab = pickle.load(handle)
    
    return vocab

def save_correction_dict(corpus_name, dictionary_type, clean_corpus, log):
    """
    Saves a dictionary of a given corpus into a file.
    """

    dict_path = f"resources/dictionaries/{corpus_name}.pickle"  
    log.info(f"Saving corpus dictionary to '{dict_path}'.")

    if dictionary_type == "corpus":
        from torch.utils.data.dataset import ConcatDataset
        clean_corpus_concat = ConcatDataset([clean_corpus.train, clean_corpus.dev, clean_corpus.test])
        vocab = {tok.text for sent in clean_corpus_concat for tok in sent}
    elif dictionary_type == "test":
        vocab = {tok.text for sent in clean_corpus.test for tok in sent}
    else:
        vocab=None # defaults to wiktionary

    with open(dict_path, "wb") as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return vocab

def correct_text(mode, input, spellcheck_module=None, dictionary=None, verbose=False):
    """
    Corrects all tokens in the given sentences with a given correction mode
    Returns the corrected sentences
    """

    if mode == CorrectionMode.Hunspell:
        return correct_text_with_hunspell(input, spellcheck_module, dictionary, verbose=verbose)
    elif mode == CorrectionMode.Natas:
        return correct_text_with_natas(input, dictionary, verbose=verbose)
    else:
        return input

def correct_sentences(mode, sentences, spellcheck=None, dictionary=None, verbose=False):
    """
    Corrects all tokens in the given sentences with a given correction mode
    Returns the corrected sentences
    """

    from copy import deepcopy
    corrected_sentences = deepcopy(sentences)

    for sentence in corrected_sentences:            
        for token in sentence:
            token.text = correct_text(mode, token.text, spellcheck, dictionary, verbose=verbose)    

    return corrected_sentences
