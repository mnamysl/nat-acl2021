import os, re
from enum import Enum
from nltk.tokenize import word_tokenize

from robust_ner.enums import Seq2SeqMode


def recreate_directory(path):
    """
    Creates a directory tree if does not exists.
    """
    
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path, ignore_errors=False)

    os.makedirs(path, exist_ok=True)

def is_correction_mode(mode):
    return mode in [Seq2SeqMode.ErrorCorrectionTok]

def is_generation_mode(mode):
    return mode in [Seq2SeqMode.ErrorGenerationCh, Seq2SeqMode.ErrorGenerationTok]

def get_max_lines_alias(max_lines_total):
    if max_lines_total == int(1e7):
        return "10M"
    elif max_lines_total == int(1e6):
        return "1M"
    elif max_lines_total == int(1e5):
        return "100k"
    elif max_lines_total == int(1e4):
        return "10k"
    elif max_lines_total == int(1e3):
        return "1k"
    elif max_lines_total == int(1e2):
        return "100"
    elif max_lines_total < 0:
        return "full"
    else:
        print(f"unrecognized max_lines_total: {max_lines_total}")
        exit(-1)

def get_filenames_for_splits(input_path, mode, max_lines_total):
    """
    Gets the filenames for the training and the validation splits.
    """

    fname, fext = os.path.splitext(input_path)
    max_lines_str = get_max_lines_alias(max_lines_total)

    src_train_path = f"{fname}_{mode.value}_{max_lines_str}_src_train{fext}"
    src_valid_path = f"{fname}_{mode.value}_{max_lines_str}_src_valid{fext}"
    tgt_train_path = f"{fname}_{mode.value}_{max_lines_str}_tgt_train{fext}"
    tgt_valid_path = f"{fname}_{mode.value}_{max_lines_str}_tgt_valid{fext}"

    return src_train_path, src_valid_path, tgt_train_path, tgt_valid_path

def encode_line(line, space=chr(172)):
    """
    Encodes the line to be utilized for training/inference.
    """
    result = " ".join([c if c != ' ' else space for c in line])
    return result

def decode_line(line, space=chr(172)):
    """
    Decodes the line to the readable format.
    """
    result = re.sub('[\s+]', '', line) # remove whitespaces
    result = re.sub(space, ' ', result) # replace space character with a whitespace char    
    return result
