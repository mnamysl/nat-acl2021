import os
import math
import logging
import random
import numpy as np

from robust_ner.confusion_matrix import noise_sentences_cmx
from robust_ner.vanilla_noise import noise_sentences_vanilla
from robust_ner.typos import noise_sentences_typos
from robust_ner.seq2seq import noise_sentences_seq2seq, Seq2SeqMode
from robust_ner.enums import MisspellingMode


def make_char_vocab(sentences):
    """
    Construct the character vocabulary from the given sentences
    """

    char_vocab = set()  

    for sentence in sentences:
        _update_char_vocab(sentence, char_vocab)
    
    return char_vocab

def _update_char_vocab(sentence, char_vocab: set):
    """
    Updates the character vocabulary using a single sentence
    """

    for token in sentence:     

        if len(token.text) > 0:
            char_vocab.update([s for s in set(token.text) if not s.isspace()])

def perturb_dataset(batch_loader, log, misspell_mode, noise_level, char_vocab, cmx, lut, typos, 
    translator, opt, translation_mode, verbose=False):
    """
    Pre-generates a perturbed version of a data set.
    """

    log.info(f"Generation of a perturbed training data set started.")

    clean_sentences = list()
            
    for batch in batch_loader:
        clean_sentences += [sentence for sentence in batch]

    misspelled_sentences, _ = noise_sentences(clean_sentences, misspell_mode, log, 
        noise_level=noise_level, char_vocab=char_vocab, cmx=cmx, lut=lut, typos=typos, 
        translator=translator, opt=opt, translation_mode=translation_mode, verbose=verbose)

    perturbed_data = dict()
    for clean_sent, misspelled_sent in zip(clean_sentences, misspelled_sentences):
        
        key = clean_sent.to_tokenized_string()
        value = perturbed_data.get(key, list())
        value.append(misspelled_sent)
        perturbed_data[key] = value

    del clean_sentences

    log.info(f"Generation of a perturbed training data set finished.")

    return perturbed_data

def noise_sentences(sentences, misspell_mode, log, noise_level = 0.0, char_vocab = {}, cmx = None, lut = {}, typos = {}, 
    translator = None, opt = None, translation_mode = Seq2SeqMode.ErrorGenerationTok, verbose: bool = False):
    """
    Induces noise on the given list of sentences
    """

    if misspell_mode == MisspellingMode.ConfusionMatrixBased:
        return noise_sentences_cmx(sentences, cmx, lut, verbose)
    elif misspell_mode == MisspellingMode.Typos:
        return noise_sentences_typos(sentences, typos, noise_level)
    elif misspell_mode == MisspellingMode.Seq2Seq:
        return noise_sentences_seq2seq(sentences, translator, opt, translation_mode, log, noise_level, verbose)
    else:
        return noise_sentences_vanilla(sentences, char_vocab, noise_level, verbose)

def _write_misrecognitions(file, lut, sort=True):
    """
    Writes misrecognitions to a file.
    """
    if sort:
        # get a version sorted by key
        items = sorted(lut.items())
    else:
        items = lut.items()    
    
    with open(file, 'w') as f:
        for clean_tok, noisy_toks in items:
            # remove all items that consist from whitespaces only (join & split)
            noisy_toks = ' '.join(noisy_toks).split()
            if len(noisy_toks) > 0:
                print(f"{clean_tok} {' '.join(noisy_toks)}", file=f)

def extract_misrecognitions(input_file, output_file, log, unique_tokens:set=None, save_every=-1, display_every=-1):
    """
    Parses the "X_pairs_norm.txt" file, extracts and aligns both the clean and the noisy sentence.
    Align both sentences at token level and writes an output LUT containing lists of
    misrecognitions for every encountered clean token.
    Additionally, enables to filter all out-of-vocabulary tokens (if unique_tokens is given).
    """

    if not os.path.exists(input_file):
        log.error(f"Input file: '{input_file}' does not exists!")
        return False

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    filename, fileext = os.path.splitext(os.path.basename(output_file))
    output_file_part = os.path.join(os.path.dirname(output_file), filename + "_part" + fileext)

    noisy_lut = dict()

    from robust_ner.align import _get_aligned_tokens

    with open(input_file, 'r') as f:
        
        line = f.readline()
        line_idx, original_text, recognized_text = 0, None, None

        while f:
            
            if line_idx % 3 == 0: # header line
                elems = line.split(';')
                if len(elems) != 3:
                    log.error(f"Line: '{line}' length != 3'")
                    exit(-1)
                if original_text != None:
                    log.error(f"original_text != None")
                    exit(-1)
                if recognized_text != None:
                    log.error(f"recognized_text != None")
                    exit(-1)

            elif line_idx % 3 == 1: # original text
                if original_text != None:
                    log.error(f"original_text != None")
                    exit(-1)
                if recognized_text != None:
                    log.error(f"recognized_text != None")
                    exit(-1)
                original_text = line

            elif line_idx % 3 == 2: # ocr-ed text
                if recognized_text != None:
                    log.error(f"recognized_text != None")
                    exit(-1)
                recognized_text = line

            if original_text != None and recognized_text != None:
                                
                aligned_tokens, status = _get_aligned_tokens(original_text, recognized_text, log)

                if not status:
                    log.error(f"_get_aligned_tokens failed!")
                else:
                    for clean_tok, noisy_tok in aligned_tokens:
                        clean_tok = clean_tok.strip()
                        noisy_tok = noisy_tok.strip()
                        if clean_tok and clean_tok != noisy_tok:
                            if unique_tokens == None or clean_tok in unique_tokens:
                                # replace whitespaces with chr(172)
                                noisy_tok_norm = noisy_tok.replace(" ", chr(172))
                                # values = noisy_lut.get(clean_tok, list())
                                # values.append(noisy_tok_norm)
                                values = noisy_lut.get(clean_tok, set())
                                values.add(noisy_tok_norm)
                                noisy_lut[clean_tok] = values

                original_text, recognized_text = None, None            

            line_idx += 1

            tuple_idx = line_idx / 3

            if display_every >= 0 and tuple_idx % display_every == 0:
                log.info(f"Processed {tuple_idx:.0f} tuples.")

            if save_every >= 0 and tuple_idx % save_every == 0:                
                _write_misrecognitions(output_file_part, noisy_lut)

            line = f.readline()
    
    from robust_ner.seq2seq import _delete_file
    _delete_file(output_file_part)

    _write_misrecognitions(output_file, noisy_lut)

    return True
