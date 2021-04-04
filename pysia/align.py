import os

from pysia.procparams import ProcParams
from pysia.algos import iterative_levenshtein, align_text

from robust_ner.embeddings import clear_embeddings


def generate_paired_sentences_from_text_files(corpus_name, log, start_from_file="", start_from_line=0, chunk_size=1000, save_img=False):
    """
    Generates a parallel data set from text files.
    """

    clear_existing = len(start_from_file) == 0
    output_dir, proc_params = _prepare_paired_sentence_generation(corpus_name, save_img, log, clear_existing)

    corpus_dir = f"resources/corpora/{corpus_name}/"    
    _process_sentences_from_text_files(corpus_dir, corpus_name, output_dir, proc_params, log, start_from_file, start_from_line, chunk_size, save_img)

def generate_paired_sentences_from_seq_lab_corpus(corpus, corpus_name, log, tag_type = None, save_img = True):
    """
    Generates a parallel data set from a sequence labeling corpus in a standard CoNLL format.
    """

    output_dir, proc_params = _prepare_paired_sentence_generation(corpus_name, save_img, log, True)    

    idx0 = 0        
    _, _, idx0 = _generate_paired_sentences(corpus.test, output_dir, "test", proc_params, idx0, log, tag_type, save_img)
    _, _, idx0 = _generate_paired_sentences(corpus.dev, output_dir, "dev", proc_params, idx0, log, tag_type, save_img)
    _, _, idx0 = _generate_paired_sentences(corpus.train, output_dir, "train", proc_params, idx0, log, tag_type, save_img)


def _process_sentences_from_text_files(path, suffix, output_dir, proc_params, log, 
    start_from_file="", start_from_line=0, chunk_size=1000, save_img=False):
    """
    Processes sentences from a text file(s) (containing each sentence in a separate line)
    """

    file_inputs = os.path.join(output_dir, f"{suffix}_input.txt")
    idx_file, idx0 = 0, 0

    continue_from_file_found = len(start_from_file) == 0

    with open(file_inputs, "a", buffering=1) as f_input_files:        
        if os.path.isdir(path): # process a directory
            for subdir, dirs, files in os.walk(path):
                for f in sorted(files):

                    filepath = os.path.join(subdir, f)
                    start_line_idx = 0                    

                    # "continue from" file not found yet
                    if not continue_from_file_found:
                        # check whether the current file is the right one
                        if start_from_file == f:
                            start_line_idx = start_from_line
                            continue_from_file_found = True                            
                        else:
                            # advance the global index by adding all lines in the skipped file
                            lines_to_skip = 0

                            with open(os.path.join(subdir, f)) as file:
                            
                                line = file.readline()
                                while file:
                                    line = file.readline()
                                    lines_to_skip += 1
                                idx0 += lines_to_skip

                            log.warning(f"File '{f}' skipped. It does not match '{start_from_file}'. Skipping {lines_to_skip} lines.")
                            idx_file += 1
                            continue
                    else:                                                
                        print(f"{idx_file};{filepath}", file=f_input_files)

                    idx0 = _process_sentences_from_text_file(filepath, idx_file, suffix, output_dir, proc_params, idx0, log, 
                        start_line_idx, chunk_size, save_img)
                                        
                    idx_file += 1

        elif os.path.isfile(path): # process a file

            print(f"Reading {path}.")

            print(f"{idx_file};{path}", file=f_input_files)
            idx0 = _process_sentences_from_text_file(path, idx_file, suffix, output_dir, proc_params, idx0, log, 
                start_line_idx, chunk_size, save_img)
            idx_file += 1

def _process_sentences_from_text_file(filepath, idx_file, suffix, output_dir, proc_params, idx0, log, start_from_line=0, chunk_size=1000, save_img=False):
    """
    Processes sentences from a text file (containing each sentence in a separate line)
    """

    from flair.data import Sentence

    log.info(f"Extracting sentences from '{filepath}' (starting from line: {start_from_line})")
        
    with open(filepath) as f:

        idx_line = 0
        
        # jump to the given line by skipping a given number of lines
        if start_from_line > 0:

            line = f.readline()

            while f and idx_line < start_from_line:
                line = f.readline()
                if line:
                    idx_line += 1
                
            idx0 += idx_line
        
        else:
            line = f.readline()

        sentences = list()

        while f:
        
            sentences.append((Sentence(line), idx_file, idx_line))
            line = f.readline()
            if line:
                idx_line += 1

            if len(sentences) >= chunk_size or not line:

                # append sentences to the results
                _, _, idx0 = _generate_paired_sentences(sentences, output_dir, suffix, proc_params, idx0, log, None, save_img)
                sentences.clear() # clear sentences and start a new chunk

    return idx0

def _prepare_paired_sentence_generation(corpus_name, save_img, log, clear_existing):
    """
    Initializes the resources needed for parallel data set generation.
    """

    output_dir = f"results/generated/{corpus_name}/"
    proc_params_path = os.path.join(output_dir, f"{corpus_name}_params.txt")

    if clear_existing:

        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir, ignore_errors=False)

        os.makedirs(output_dir, exist_ok=True)        

    proc_params = ProcParams(_is_tesseract4())
    proc_params.save_img = save_img

    log.info(_get_tesseract_version())

    fonts_dir = "resources/fonts/latin"
    proc_params.load_fonts(fonts_dir)

    proc_params.write(proc_params_path)

    return output_dir, proc_params

def _generate_and_recognize(idx, text_lines, proc_params, image_dir, idx0):

    idx_global = idx + idx0

    original_text = text_lines[idx].strip()

    import string
    has_printable_chars = len([c for c in original_text if c in string.printable]) > 0

    if has_printable_chars:    

        from trdg.data_generator import FakeTextDataGenerator

        font = proc_params.fonts[idx_global % len(proc_params.fonts)]

        if isinstance(proc_params.distorsion_type, list): 
            import random
            distorsion_type = random.choice(proc_params.distorsion_type)            
        else: 
            distorsion_type = proc_params.distorsion_type
        
        img = FakeTextDataGenerator.generate(idx, original_text, font=font, out_dir=None, extension=None, name_format=0, 
            size=proc_params.size, skewing_angle=proc_params.skewing_angle, random_skew=proc_params.random_skew, blur=proc_params.blur, 
            random_blur=proc_params.random_blur, background_type=proc_params.background_type, distorsion_type=distorsion_type, 
            distorsion_orientation=proc_params.distorsion_orientation, is_handwritten=proc_params.is_handwritten, width=proc_params.width,
            alignment=proc_params.alignment, text_color=proc_params.text_color, orientation=proc_params.orientation, 
            space_width=proc_params.space_width, character_spacing=proc_params.character_spacing, margins=proc_params.margins, 
            fit=proc_params.fit, output_mask=proc_params.output_mask)        

        if proc_params.save_img:
            # img.save(os.path.join(image_dir, f"{idx_global:05d}.jpg")) # keep all images
            img.save(os.path.join(image_dir, f"{idx:05d}.jpg")) # keep last N images (ring buffer like saving)

        from tesserocr import PyTessBaseAPI, PSM, OEM

        oem = OEM.LSTM_ONLY if _is_tesseract4() else OEM.DEFAULT
        psm = PSM.RAW_LINE if _is_tesseract4() else PSM.SINGLE_LINE

        with PyTessBaseAPI(psm=psm, oem=oem) as ocr:
            
            ocr.SetImage(img)

            try:
                recognized_text = ocr.GetUTF8Text()        
                recognized_text = recognized_text.strip().replace("\n", "").replace("\t", "")
            except RuntimeError:
                recognized_text="" # set empty recognized text - it will be ignored in the latter stage
    else:
        recognized_text = "" # set empty recognized text - it will be ignored in the latter stage
        
    return (img, original_text, recognized_text)

def _generate_paired_sentences(sentences, output_dir, suffix, proc_params, idx0, log, tag_type, save_img):

    image_dir = os.path.join(output_dir, f"{suffix}_images")
    sent_pairs_file = os.path.join(output_dir, f"{suffix}_pairs.txt")
    sent_align_file = os.path.join(output_dir, f"{suffix}_align.txt")
    conll_file = os.path.join(output_dir, f"{suffix}.csv")

    if proc_params.save_img:
        os.makedirs(image_dir, exist_ok=True)

    from trdg.generators import GeneratorFromStrings
            
    text_lines = list()
    
    for sent in sentences:
        if isinstance(sent, tuple):
            text_lines.append(sent[0].to_plain_string())
        else:
            text_lines.append(sent.to_plain_string())

    import multiprocessing as mp
    from functools import partial

    # thread pool
    num_cpu = mp.cpu_count()
    num_processes = 2 #int(mp.cpu_count() * 0.5)

    log.info(f"Chunk started: num_processes={num_processes} num_cpu={num_cpu} size={len(text_lines)}")

    with mp.Pool(processes = num_processes) as p:
        results = p.map(partial(_generate_and_recognize, text_lines=text_lines, proc_params=proc_params, image_dir=image_dir, idx0=idx0), range(len(text_lines)))
    
    if tag_type != None:        
        file_conll = open(conll_file, "a", buffering=1)

    len_total, LD_total = 0, 0

    font_err_dict = dict()

    with open(sent_pairs_file, "a", buffering=1) as file_pairs, \
        open(sent_align_file, "a", buffering=1) as file_align:
        
        for idx, (img, original_text, recognized_text) in enumerate(results):
            
            idx_global = idx + idx0

            if tag_type != None:
                # do not allow empty token text, which would be problematic while parsing CoNLL format files
                if len(original_text) == 0:
                    original_text = " "
                if len(recognized_text) == 0:
                    recognized_text = " "

            LD, edit_ops = iterative_levenshtein(original_text, recognized_text)            

            len_total += len(original_text)
            LD_total += LD
            
            original_aligned = align_text(original_text, edit_ops, "i", 172)
            recognized_aligned = align_text(recognized_text, edit_ops, "d", 166)

            if isinstance(sentences[idx], tuple):
                tagged_sentence, idx_file, idx_line = sentences[idx]
            else:
                tagged_sentence = sentences[idx]
            
            if tag_type != None:
                noisy_sentence, transfer_status = _transfer_tags(original_aligned, recognized_aligned, edit_ops, tagged_sentence, tag_type, log)

                for token_clean, token_noisy in zip(tagged_sentence, noisy_sentence):                    
                    # replace all whitespace characters wih char(172) in a noisy token text
                    # token_noisy_text = token_noisy.text
                    token_noisy_text = token_noisy.text.replace(" ", chr(172))

                    print(f"{token_noisy_text}\t{token_clean.text}\t{token_noisy.get_tag(tag_type).value}", file=file_conll)

                print(f"", file=file_conll)
            else:
                transfer_status = True

            font = proc_params.fonts[idx_global % len(proc_params.fonts)]

            font_err_res = font_err_dict.get(font, [0, 0])
            font_err_res[0] += LD
            font_err_res[1] += len(original_text)
            font_err_dict.update({font : font_err_res})

            # save to file
            if isinstance(sentences[idx], tuple):
                print(f"{idx_global:05d};{idx_file};{idx_line};{len(original_aligned)};{font};{LD}", file=file_align)
            else:
                print(f"{idx_global:05d};{len(original_aligned)};{font};{LD}", file=file_align)

            print(f"{original_aligned}", file=file_align)
            print(f"{recognized_aligned}", file=file_align)
            print(f"{edit_ops}", file=file_align)

            print(f"{idx_global:05d};{len(original_text)};{len(recognized_text)}", file=file_pairs)
            print(f"{original_text}", file=file_pairs)
            print(f"{recognized_text}", file=file_pairs)
            
            if not transfer_status:
                exit(-1)

    if tag_type != None:
        file_conll.close()

    idx_global += 1

    # print error rates for the current chunk
    log.info(f"Chunk finished: idx_global={idx_global} cnt={len(results)} LD={LD_total} len_total={len_total} CER={LD_total * 100.0 / len_total:.2f}%")

    print_erroneous_fonts = True

    if print_erroneous_fonts:

        d = sorted(font_err_dict.items(), key=lambda kv: (kv[1][0] / kv[1][1]))
        d.reverse()
        for i, (key, value) in enumerate(d):
            LD, length = value
            CER = LD * 100.0 / length
            log.info(f"\t{key}: {LD}/{length} (CER={CER:.2f}%)")
            if i == 10:
               break

    return len_total, LD_total, idx_global

def _transfer_tags(original_text, noisy_text, edit_ops, tagged_sentence, tag_type, log, convert_tags=True):

    from copy import deepcopy
    noisy_tagged_sentence = deepcopy(tagged_sentence)

    #TODO: empty noisy text!!
    status = True

    idx = 0
    for token in noisy_tagged_sentence:

        clean_token_text = token.text
        noisy_token_text = ""
        
        token_idx = 0
        while token_idx < len(clean_token_text) and idx < len(edit_ops):
                        
            op = edit_ops[idx]

            char_token = clean_token_text[token_idx]
            char_orig = original_text[idx]
            
            if op in ["-", "s"]:
                noisy_token_text += noisy_text[idx]
                token_idx += 1
                check = True
            elif op == "i":                
                noisy_token_text += noisy_text[idx] # insert char and do not move to the next one
                check = False
            elif op == "d":
                token_idx += 1 # skip char and move to the next one
                check = True

            if check and char_orig != char_token:
                log.error(f"WRONG!!! idx={idx} {char_orig} != {char_token}")
                status = False

            idx += 1

        # the next char is a whitespace (if we are not at the end of a sentence)
        # check whether it is substituted with another character, which will be 
        # included into the noisy token text
        
        if idx < len(edit_ops) and edit_ops[idx] == "s" and original_text[idx].isspace():
            noisy_token_text += noisy_text[idx]
            idx += 1
        
        # alternatively, there could be one or more insertions at the end
        # include them into the noisy token text
        while idx < len(edit_ops) and edit_ops[idx] == "i":
            noisy_token_text += noisy_text[idx]
            idx += 1
        
        if idx < len(edit_ops) and edit_ops[idx] in ["-", "d"] and original_text[idx].isspace():
            idx += 1        

        token.text = noisy_token_text

        if convert_tags:
            token.get_tag(tag_type).value = _bioes_to_bio(token.get_tag(tag_type).value)

        # if clean_token_text != noisy_token_text:
        #     log.info(f"*{clean_token_text}* -> *{noisy_token_text}* [{token.get_tag(tag_type).value}]")
    
    clear_embeddings(noisy_tagged_sentence)

    return noisy_tagged_sentence, status

def _split_contractions(tokens):
    """
    A function to split apostrophe contractions at the end of alphanumeric (and hyphenated) tokens.

    Takes the output of any of the tokenizer functions and produces and updated list.

    :param tokens: a list of tokens
    :returns: an updated list if a split was made or the original list otherwise

    Credit: (adapted from 'segtok/tokenizer.py')
    """

    from segtok.tokenizer import IS_CONTRACTION, APOSTROPHES

    repeat = True

    while (repeat):

        repeat = False
        idx = -1

        for token in list(tokens):
            idx += 1

            if IS_CONTRACTION.match(token) is not None:
                length = len(token)

                if length > 1:
                    for pos in range(length - 1, -1, -1):
                        if token[pos] in APOSTROPHES:
                            if 2 < length and pos + 2 == length and token[-1] == 't' and token[pos - 1] == 'n':
                                pos -= 1
                            else:
                                repeat = True

                            tokens.insert(idx, token[:pos])
                            idx += 1
                            tokens[idx] = token[pos:]
                            
                            break
    return tokens

def _get_aligned_tokens(original_text, noisy_text, log):
    """
    Adopted from "_transfer_tags()"
    """

    # Unicode Character 'LINE SEPARATOR' (U+2028)
    # https://www.fileformat.info/info/unicode/char/2028/index.htm
    # Some lines in the input contain this character, which separates two lines in one line
    # in the input and should be handled by the data generators (but it wasn't).
    # Ultimately, I should handle it in the alignment procedure that writes the data.

    if u"\u2028" in original_text:
        log.warning(f"Found <U+2028> in: *{original_text}*")
        original_text = original_text.replace(u"\u2028", "")
        log.warning(f"Normalized version: {original_text}")
    
    LD, edit_ops = iterative_levenshtein(original_text, noisy_text)

    original_aligned = align_text(original_text, edit_ops, "i", 172)
    noisy_aligned = align_text(noisy_text, edit_ops, "d", 166)

    return _get_aligned_tokens_core(original_text, original_aligned, noisy_aligned, edit_ops)

def _get_aligned_tokens_core(original_text, original_aligned, noisy_aligned, edit_ops):    
    
    #TODO: empty noisy text!!
    status = True    
    
    from segtok.tokenizer import word_tokenizer #, split_contractions
    tokenized_original_text = _split_contractions(word_tokenizer(original_text))

    aligned_tokens = list()
    
    idx = 0
    for clean_token_text in tokenized_original_text:

        noisy_token_text = ""
        
        token_idx = 0
    
        # loop till the first char of the token match with the character of
        # the aligned original text. It will skip spurious tokens that could
        # arise from insertion errors between tokens.
        while token_idx < len(clean_token_text) and idx < len(edit_ops):
            
            char_token = clean_token_text[token_idx]
            char_orig = original_aligned[idx]

            if char_token == char_orig:
                break

            idx += 1
            
   
        while token_idx < len(clean_token_text) and idx < len(edit_ops):

            op = edit_ops[idx]

            char_token = clean_token_text[token_idx]
            char_orig = original_aligned[idx]
            
            if op == "-":
                noisy_token_text += noisy_aligned[idx]
                token_idx += 1
                check = True
            elif op == "s":
                noisy_token_text += noisy_aligned[idx]
                token_idx += 1
                check = True
            elif op == "i":
                noisy_token_text += noisy_aligned[idx] # insert char and do not move to the next one
                check = False
            elif op == "d":
                token_idx += 1 # skip char and move to the next one
                check = True

            if check and char_orig != char_token:
                print(f"WRONG!!! idx={idx} {char_orig} != {char_token}")
                print(f"noisy_token_text={noisy_token_text}")
                status = False

            idx += 1

        # the next char is a whitespace (if we are not at the end of a sentence)
        # check whether it is substituted with another character, which will be 
        # included into the noisy token text
        
        if idx < len(edit_ops) and edit_ops[idx] == "s" and original_aligned[idx].isspace():
            noisy_token_text += noisy_aligned[idx]
            idx += 1
        
        # alternatively, there could be one or more insertions at the end
        # include them into the noisy token text
        while idx < len(edit_ops) and edit_ops[idx] == "i":
            noisy_token_text += noisy_aligned[idx]
            idx += 1
        
        if idx < len(edit_ops) and edit_ops[idx] in ["-", "d"] and original_aligned[idx].isspace():
            idx += 1        

        aligned_tokens.append((clean_token_text, noisy_token_text))

        # if clean_token_text != noisy_token_text:
        #     log.info(f"*{clean_token_text}* -> *{noisy_token_text}*")

    if not status:
        print(f"{original_text}")
        print(f"{original_aligned}")
        print(f"{noisy_aligned}")
        print(f"{edit_ops}")
        print(f"{tokenized_original_text}")
        print(f"{aligned_tokens}")
        exit(-1)

    return aligned_tokens, status

def _bioes_to_bio(tag):
    
    split = tag.split("-")

    if len(split) == 2:
        if split[0] == "S":
            return "-".join(["B", split[1]])
        elif split[0] == "E":
            return "-".join(["I", split[1]])

    return tag

def _get_tesseract_version():
    
    from tesserocr import tesseract_version
    return tesseract_version()

def _is_tesseract4():

    return _get_tesseract_version().startswith("tesseract 4.")
