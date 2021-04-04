
import os
import random as rand
# import collections

from pysia.algos import iterative_levenshtein, align_text
from pysia.align import _transfer_tags
from pysia.nmt import _translate_lines
from pysia.utils import get_max_lines_alias, recreate_directory

from robust_ner.enums import Seq2SeqMode


def extract_noisy_corpus(input_path, log, max_lines=-1, split_num_lines=int(3e6)):

    from segtok.tokenizer import word_tokenizer
    from pysia.align import _split_contractions
    
    fname, fext = os.path.splitext(input_path)
    max_lines_str = get_max_lines_alias(max_lines)
    org_dir = f"{fname}_org_{max_lines_str}"
    rec_dir = f"{fname}_rec_{max_lines_str}"

    log.info(f"Starting corpus extraction:")
    log.info(f"Input data directory: {input_path}")
    log.info(f"Original data directory: {org_dir}")
    log.info(f"Noisy data directory: {rec_dir}")

    recreate_directory(org_dir)
    recreate_directory(rec_dir)

    org_file_idx, rec_file_idx = 0, 0
    org_line_idx, rec_line_idx = 0, 0
    #org_line_limit, rec_line_limit = split_num_lines / 10, split_num_lines / 10 # first split for validation
    org_line_limit, rec_line_limit = split_num_lines, split_num_lines

    org_file_path = os.path.join(org_dir, f"{org_file_idx:04d}_org.txt")
    rec_file_path  = os.path.join(rec_dir, f"{org_file_idx:04d}_rec.txt")
     
    log.info(f"opening '{org_file_path}' for writing..")
    org_file = open(org_file_path, "w")
    
    log.info(f"opening '{rec_file_path}' for writing..")
    rec_file = open(rec_file_path, "w")

    num_org_lines, num_rec_lines = 0, 0

    with open(input_path, "r") as input_file:

        line = input_file.readline()
        line_idx = 0
        
        while line:
            
            tokens = _split_contractions(word_tokenizer(line.strip()))
            line = ' '.join([tok.strip() for tok in tokens])
            
            if line_idx % 3 == 0: # header line
                elems = line.split(';')
                if len(elems) != 3:
                    log.error(f"Line: '{line}' length != 3'")
                    exit(-1)

            elif line_idx % 3 == 1: # original text
                print(line.strip(), file=org_file)
                org_line_idx += 1
                num_org_lines += 1
                if org_line_idx >= org_line_limit:
                    org_file.close()
                    org_file_idx += 1
                    org_file_path = os.path.join(org_dir, f"{org_file_idx:04d}_org.txt")
                    log.info(f"opening '{org_file_path}' for writing..")
                    org_file = open(org_file_path, "w")
                    org_line_idx = 0
                    org_line_limit = split_num_lines

            elif line_idx % 3 == 2: # recognized text  
                print(line.strip(), file=rec_file)
                rec_line_idx += 1
                num_rec_lines += 1
                if rec_line_idx >= rec_line_limit:
                    rec_file.close()
                    rec_file_idx += 1
                    rec_file_path = os.path.join(rec_dir, f"{rec_file_idx:04d}_rec.txt")
                    log.info(f"opening '{rec_file_path}' for writing..")
                    rec_file = open(rec_file_path, "w")
                    rec_line_idx = 0
                    rec_line_limit = split_num_lines

            if max_lines > 0:
                num_lines = min(num_org_lines, num_rec_lines)
                if num_lines >= max_lines:
                    break

            line_idx += 1
            line = input_file.readline()

    org_file.close()
    rec_file.close()

    log.info(f"Loaded {line_idx} lines.")

def induce_noise_seq2seq(sources, translator, opt, mode: Seq2SeqMode, verbose: bool = False):
    """
    Induces noise using a seq2seq translator model.
    """

    selected_predictions = ["" for i in range(len(sources))]

    all_scores, all_predictions = _translate_lines(sources, translator, opt, mode)

    # select noisy tokens among predictions    
    for idx, (scores, preds, source) in enumerate(zip(all_scores, all_predictions, sources)):
        for score, pred in zip(scores, preds):
            if len(selected_predictions[idx]) == 0:
                selected_predictions[idx] = pred # if no result set - take the best output
                if pred != source:
                    break # break if the best output is different that source
            elif pred != source:
                selected_predictions[idx] = pred
                break # break if the prediction is different than source

    return selected_predictions

def noise_sentences_seq2seq(data_points, translator, opt, mode: Seq2SeqMode, log, noise_level = 1.0, verbose:bool = False):
    """
    Induces noise using a seq2seq translator model.
    """

    from copy import deepcopy    

    if mode == Seq2SeqMode.ErrorGenerationTok:
        noisy_data_points = deepcopy(data_points)
        clean_strings = [token.text for sentence in data_points for token in sentence]

    elif mode == Seq2SeqMode.ErrorGenerationCh:
        noisy_data_points = list()
        clean_strings = [sentence.to_plain_string() for sentence in data_points]

    # induce noise
    noisy_strings = induce_noise_seq2seq(clean_strings, translator, opt, mode, verbose=verbose)

    cnt_noised_tokens = list()

    # transfer the noise from the clean data points
    if mode == Seq2SeqMode.ErrorGenerationTok:
        token_idx = 0    
        for clean_sentence, noisy_sentence in zip(data_points, noisy_data_points):
            for token in noisy_sentence:
                if noise_level >= 1.0 or rand.random() <= noise_level:
                    token.text = noisy_strings[token_idx]
                token_idx += 1

            cnt_noised_tokens.append(sum([1 if clean_tok.text != noisy_tok.text else 0 for clean_tok, noisy_tok in zip(clean_sentence, noisy_sentence)]))

    elif mode == Seq2SeqMode.ErrorGenerationCh:
        for sent_idx, clean_sentence in enumerate(data_points):
            _, edit_ops = iterative_levenshtein(clean_strings[sent_idx], noisy_strings[sent_idx])
            clean_string_aligned = align_text(clean_strings[sent_idx], edit_ops, "i", 172)
            noisy_string_aligned = align_text(noisy_strings[sent_idx], edit_ops, "d", 166)
            noisy_sentence, transfer_status = _transfer_tags(clean_string_aligned, noisy_string_aligned, edit_ops, data_points[sent_idx], None, log)                
            
            if not transfer_status:
                log.error("Transfer of tags failed!")
                exit(-1)
            
            noisy_data_points.append(noisy_sentence)

            cnt_noised_tokens.append(sum([1 if clean_tok.text != noisy_tok.text else 0 for clean_tok, noisy_tok in zip(clean_sentence, noisy_sentence)]))

    return noisy_data_points, cnt_noised_tokens
