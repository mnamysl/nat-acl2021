from pysia.utils import (
    is_correction_mode,
    is_generation_mode,
    get_filenames_for_splits,
    encode_line,
)

from pysia.align import _get_aligned_tokens

from robust_ner.enums import Seq2SeqMode

def _write_lines(lines, src_path, tgt_path, mode, log):
    """
    Writes the output of the training and the validation splits.
    """

    with open(src_path, "w") as src_file, open(tgt_path, "w") as tgt_file:
                
        for i, (src_line, tgt_line) in enumerate(lines):

            if i % 1000 == 0:
                log.info(f"Writing line {i}..")

            # check whether to tokenize and align the sentences to save token-level output
            if mode == Seq2SeqMode.ErrorGenerationTok:
                aligned_tokens, _ = _get_aligned_tokens(src_line.strip(), tgt_line.strip(), log)
                for tok_pair in aligned_tokens:
                    src = encode_line(tok_pair[0])
                    tgt = encode_line(tok_pair[1])
                    if len(src) > 0 and len(tgt) > 0:
                        print(src, file=src_file)
                        print(tgt, file=tgt_file)

            elif mode == Seq2SeqMode.ErrorCorrectionTok:
                aligned_tokens, _ = _get_aligned_tokens(tgt_line.strip(), src_line.strip(), log)
                for tok_pair in aligned_tokens:
                    src = encode_line(tok_pair[1])
                    tgt = encode_line(tok_pair[0])
                    if len(src) > 0 and len(tgt) > 0:
                        print(src, file=src_file)
                        print(tgt, file=tgt_file)

            else:
                src = encode_line(src_line.strip())
                tgt = encode_line(tgt_line.strip())
                if len(src) > 0 and len(tgt) > 0:
                    print(src, file=src_file)
                    print(tgt, file=tgt_file)

def split_train_valid(input_path, mode, log, ratio_valid=0.1, max_lines_total=-1, max_lines_valid=5000):
    """
    Generates the training and the validation splits for the sequence-to-sequence training.
    """
    src_lines, tgt_lines = list(), list()

    log.info(f"Reading '{input_path}' (max_lines_total={max_lines_total:.0f})")    

    with open(input_path) as input_file:

        line = input_file.readline()
        line_idx = 0
        
        while line:
            
            if line_idx % 3 == 0: # header line
                elems = line.split(';')
                if len(elems) != 3:
                    log.error(f"Line [{line_idx}]: '{line}' length(={len(elems)}) != 3 line:'{line}'")
                    exit(-1)

            elif line_idx % 3 == 1: # original text
                if is_generation_mode(mode):
                    src_lines.append(line)
                elif is_correction_mode(mode):
                    tgt_lines.append(line)                

            elif line_idx % 3 == 2: # recognized text
                if is_generation_mode(mode):
                    tgt_lines.append(line)                    
                elif is_correction_mode(mode):
                    src_lines.append(line)              

            if max_lines_total > 0:
                num_lines = min(len(src_lines), len(tgt_lines))
                if num_lines >= max_lines_total:
                    break

            line_idx += 1
            line = input_file.readline()    

    num_lines = len(src_lines)
    num_valid = min(int(num_lines * ratio_valid + 0.5), max_lines_valid)

    log.info(f"Loaded {num_lines} lines. Setting {num_valid} lines aside for validation.")
    
    import random as rand
    # indices = rand.sample(range(num_lines), num_valid)
    indices = slice(0, num_valid, 1)
    
    valid_lines = zip(src_lines[indices], tgt_lines[indices])
    
    # remove  elements of the validation set
    del src_lines[indices]
    del tgt_lines[indices]

    train_lines = zip(src_lines, tgt_lines)    
    
    log.info(f"Writing train/valid splits..")

    src_train_path, src_valid_path, tgt_train_path, tgt_valid_path = get_filenames_for_splits(input_path, mode, max_lines_total)

    _write_lines(train_lines, src_train_path, tgt_train_path, mode, log)
    log.info(f"Training data ready ({src_train_path}, {tgt_train_path}).")

    _write_lines(valid_lines, src_valid_path, tgt_valid_path, mode, log)
    log.info(f"Validation data ready ({src_valid_path}, {tgt_valid_path}).")
