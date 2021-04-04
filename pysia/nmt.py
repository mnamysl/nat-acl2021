import os
import torch

from pysia.utils import (
    recreate_directory, 
    get_filenames_for_splits,
    decode_line,
    encode_line,
    is_correction_mode,
    is_generation_mode,
)

from robust_ner.enums import Seq2SeqMode


def onmt_preprocess(onmt_path, input_path, data_path, mode, log, shard_size, num_threads, max_seq_len=int(1e6), max_lines_total=-1):
    """
    Performs the data pre-processing using ONMT.
    """

    recreate_directory(os.path.dirname(data_path))

    suffixes = list()
    suffixes.append("-share_vocab")

    src_train_path, src_valid_path, tgt_train_path, tgt_valid_path = get_filenames_for_splits(input_path, mode, max_lines_total)
    
    log.info(f"Starting data pre-processing (shard_size={shard_size} num_threads={num_threads})..")
    os.system(f"python3.6 {onmt_path}/onmt/bin/preprocess.py -train_src {src_train_path} -train_tgt {tgt_train_path} -valid_src {src_valid_path} " \
        f"-valid_tgt {tgt_valid_path} -save_data {data_path} -overwrite -dynamic_dict -shard_size {shard_size} -num_threads {num_threads} " \
        f"-src_seq_length {max_seq_len} -tgt_seq_length {max_seq_len} " \
        f"{' '.join(suffixes)}")

def onmt_train(onmt_path, data_path, model_path, mode, log, lr=1.0, max_batches=32, train_steps=1e5, valid_steps=1e4, 
    decay_steps=2e4, start_decay_steps=2e4):
    """
    Performs training of the ONMT model.
    """

    src_word_vec_size, tgt_word_vec_size = 25, 25
    gpu_settings = ""

    if torch.cuda.is_available():
        gpu_settings = "-world_size 1 -gpu_ranks 0"
        
    log.info(f"Starting training (src_word_vec_size={src_word_vec_size}, tgt_word_vec_size={tgt_word_vec_size})..")
    
    recreate_directory(os.path.dirname(model_path))

    suffixes = list()
    suffixes.append("-share_embeddings")
    if mode != Seq2SeqMode.ErrorCorrectionTok:
        # Natas use ONMT v0.8.2 and had problems with ensemble model and coppy_attn
        suffixes.append("-copy_attn")            

    os.system(f"python3.6 {onmt_path}/onmt/bin/train.py -data {data_path} -save_model {model_path} " \
        f"-src_word_vec_size {src_word_vec_size} -tgt_word_vec_size {tgt_word_vec_size} -encoder_type brnn -decoder_type rnn " \
        f"-learning_rate {lr} -decay_steps {decay_steps} -start_decay_steps {start_decay_steps} -valid_steps {valid_steps} " \
        f"-max_generator_batches {max_batches} -train_steps {train_steps} " \
        f"{gpu_settings} {' '.join(suffixes)}")
        # -single_pass (Make a single pass over the training dataset.)

def _build_args(model_path, mode, log, temp=1.0, topk=-1, nbest=1, beam_size=10, shard_size=0, batch_size=32, verbose=False):

    if verbose:
        log.info(f"model_path={model_path}")
        log.info(f"mode={mode}")
        log.info(f"temp={temp}")
        log.info(f"topk={topk}")
        log.info(f"nbest={nbest}")
        log.info(f"beam_size={beam_size}")

    src, tgt, out = None, None, None

    gpu = 0 if torch.cuda.is_available() else -1

    suffixes = list()
    suffixes.append("-share_vocab")

    if is_generation_mode(mode): # use random sampling
        import time
        t = int(time.time() * 1000.0)
        seed = (((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24))

        args = f"-model {model_path} -src {src} -tgt {tgt} -output {out} -shard_size {shard_size} -batch_size {batch_size}" \
            f" -n_best {1} -beam_size {1} -seed {seed} -random_sampling_temp {temp} -random_sampling_topk {topk}" \
            f" -gpu {gpu} {' '.join(suffixes)}"
            
    elif is_correction_mode(mode): # use beam search   
        args = f"-model {model_path} -src {src} -tgt {tgt} -output {out} -shard_size {shard_size} -batch_size {batch_size}" \
            f" -n_best {nbest} -beam_size {beam_size} -random_sampling_topk {1}" \
            f" -gpu {gpu} {' '.join(suffixes)}"
    else:
        print("ERROR: neither generation nor correction mode specified!")
        exit(-1)

    return args

def _build_translator(args):
    """
    Initializes a seq2seq translator model
    """

    from onmt.utils.parse import ArgumentParser    
    parser = ArgumentParser()
    
    import onmt.opts as opts
    opts.config_opts(parser)
    opts.translate_opts(parser)
    
    opt = parser.parse_args(args=args)    
    ArgumentParser.validate_translate_opts(opt)

    from onmt.translate.translator import build_translator
    translator = build_translator(opt, report_score=False)

    return translator, opt

def _translate_lines(src_lines, translator, opt, mode):
    """    
    Translates the given lines using a translator and opt object.
    Returns output scores and predictions.
    """
    
    # prepare shards (encode source lines)
    encoded_src_lines = [[encode_line(line).encode('utf-8') for line in src_lines]]

    for src_shard, tgt_shard in zip(encoded_src_lines, encoded_src_lines):
        scores, predictions = translator.translate(src=src_shard, tgt=tgt_shard, src_dir=opt.src_dir, 
            batch_size=opt.batch_size, batch_type=opt.batch_type, 
            attn_debug=opt.attn_debug, align_debug=opt.align_debug)

    # decode predictions
    decoded_predictions = [[decode_line(pred) for pred in preds] for preds in predictions]

    return scores, decoded_predictions

def init_translator(model_name, mode, log, temp=1.0, topk=-1, nbest=1, beam_size=10, shard_size=0, batch_size=32, verbose=False):

    args = _build_args(model_name, mode, log, temp=temp, topk=topk, nbest=nbest, beam_size=beam_size, 
        shard_size=shard_size, batch_size=batch_size, verbose=verbose)
    translator, opt = _build_translator(args)
    
    return translator, opt
