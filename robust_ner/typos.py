import os.path
import logging
import random
import numpy as np


def load_typos(file_name, char_vocab = {}, filter_OOA_chars = False):
    """
    Loads typos from a file or a list of files.
    """

    if isinstance(file_name, str):
        typos = load_typos_file(file_name, char_vocab, filter_OOA_chars)

    elif isinstance(file_name, list) or isinstance(file_name, tuple):
        typos = dict()
        for fn in file_name:
            # load a single typos file
            typos_part = load_typos_file(fn, char_vocab, filter_OOA_chars)
            # merge dictionaries
            for key, value in typos_part.items():
                if key not in typos:
                    typos[key] = value
                else:
                    typos[key] += value
    else:
        print(f"unrecognized file_name type: {file_name}")
        exit(-1)

    return typos


def load_typos_file(file_name, char_vocab = {}, filter_OOA_chars = False):
    """
    Loads typos from a given file.
    Optionally, filters all entries that contain out-of-alphabet characters.
    """

    basename, ext = os.path.splitext(file_name)

    replacement_rules = list()

    if ext == ".tsv":
        typos = load_typos_moe(file_name)
    else:
        typos = load_typos_belinkov_bisk(file_name)

    if "extracted" in basename:
        print("> applying replacement rules..")
        replacement_rules.append((chr(172), ' '))

    typos = _normalize_typos(typos, replacement_rules)

    if filter_OOA_chars:
        typos = _filter_typos(typos, char_vocab)
    
    return typos

def load_typos_moe(file_name):
    """
    Loads and returns a typos dictionary from a given file.
    Designed for Misspelling Oblivious Word Embeddings (MOE):
    https://github.com/facebookresearch/moe
    """
    
    # log = logging.getLogger("robust_ner")

    file_path = os.path.join(f"resources/typos/", f"{file_name}")

    typos = dict()
    for line in open(file_path):
        line = line.strip().split()

        if len(line) != 2:
            #log.warning(f"len(line) = {len(line)} != 2 (line: {line})")
            continue

        value = line[0]
        key = line[1]

        #print(key, value)
        
        if key not in typos:
            typos[key] = list()

        typos[key].append(value)
    
    return typos

def load_typos_belinkov_bisk(file_name):
    """
    Loads and returns a typos dictionary from a given file
    Credit: https://github.com/ybisk/charNMT-noise/blob/master/scrambler.py
    """
    
    file_path = os.path.join(f"resources/typos/", f"{file_name}")

    typos = {}
    for line in open(file_path):
        line = line.strip().split()
        typos[line[0]] = line[1:]
    
    return typos

def _filter_typos(typos, char_vocab):
    """
    Filters typos that contain out of the alphabet symbols
    """   

    new_typos = dict()

    for key,values in typos.items():
        new_values = list()
        for v in values:            
            
            invalid_chars = [c for c in v if c not in char_vocab]
            if len(invalid_chars) > 0:
                continue

            new_values.append(v)

        if len(new_values) > 0:
            new_typos[key] = new_values      

    return new_typos

def _normalize_typos(typos, replacement_rules):
    """
    Applies all character replacement rules to the typos and returns a new
    dictionary of typos of all non-empty elements from normalized 'typos'.
    """

    if len(replacement_rules) > 0:
        
        typos_new = dict()
        
        for key, values in typos.items():        
            typos_new[key] = list()
        
            for item in values:        
                for orig, replacement in replacement_rules:
                    item = item.replace(orig, replacement)
        
                item = item.strip()
                if item:
                    typos_new[key].append(item)

        return typos_new
    
    else:

        return typos

def induce_noise_typos(input_token, typos : dict, prob = 1.0):
    """
    Induces a random typo into the input token with a given probability.
    Credit: https://github.com/ybisk/charNMT-noise/blob/master/scrambler.py
    """

    if input_token in typos and random.random() <= prob:
        typos_for_token = typos[input_token]
        typo_idx = random.randint(0, len(typos_for_token) - 1)
        typo = typos_for_token[typo_idx]
        return typo, typo != input_token
    else:
        return input_token, False

def noise_sentences_typos(sentences, typos : dict, prob = 1.0):
    """
    Induces noise on the given list of sentences using a LUT of typos.
    """

    from copy import deepcopy
    noised_sentences = deepcopy(sentences)
    
    cnt_noised_tokens = 0
    for sentence in noised_sentences:            
        for token in sentence:
            token.text, noised = induce_noise_typos(token.text, typos, prob)        
            if noised: 
                cnt_noised_tokens += 1

    return noised_sentences, cnt_noised_tokens
