# Empirical Error Modeling Improves Robustness of Noisy Neural Sequence Labeling

This repository contains the code and the data from the paper: "**Empirical Error Modeling Improves Robustness of Noisy Neural Sequence Labeling**" that was submitted to the [ACL 2021](https://2021.aclweb.org/) conference. 
It is an extended version of the Noise-Aware Training ([NAT](https://github.com/mnamysl/nat-acl2020)) framework.



## Background

### Original NAT Approach
NAT is a method that utilizes both the original and the perturbed input for training of the sequence labeling models. It improves accuracy on the data from noisy sources such as user-generated text or text produced by the Optical Character Recognition ([OCR](https://en.wikipedia.org/wiki/Optical_character_recognition)) process.

### Empirical Error Modeling
At training time, the original NAT method used a vanilla synthetic error model to induce the noise into the error-free sentences. Although utilizing randomized error patterns during training often provided moderate improvements, we argue that the empirical error modeling would be significantly more beneficial. Our extension implements a data-driven error generator based on the sequence-to-sequence modeling paradigm. Our models were trained to solve the monotone string translation problem, akin to the error correction setting but in the opposite direction, i.e., we used an error-free sentence as input and trained the models to produce an erroneous text from it. Please refer to our paper for a more detailed explanation of our approach.

### Noisy Sequence Labeling Data Sets
Moreover, to better imitate the real-world application scenario, we generated a set of noisy data sets by applying an OCR engine on the sentences extracted from the original sequence labeling benchmarks. Our code could be readily used to produce noisy version of other sequence labeling benchmarks. We encourage the user to experiment with this functionality. We hope that our work will facilitate the future research on robustness in Natural Language Processing.



## Project Structure

The structure of the project reminds that from the original NAT framework with some differences (marked as [new] in the diagram below):

```
├── flair [extern]
├── flair_ext
│   ├── models
│   ├── trainers
│   └── visual
├── natas [extern]
├── onmt [extern]
├── pysia [new]
├── resources
│   ├── cmx
│   ├── conversion [new]
│   ├── corpora [new]
│   ├── dictionaries [new]
│   ├── fonts [new]
│   ├── language_models [new]
│   ├── taggers
│   ├── tasks
│   └── typos
├── results
├── robust_ner
├── scripts [new]
└── trdg [extern]
```

* **natas**: contains the [NATAS](https://github.com/mikahama/natas) library for OCR post-correction that we used as a baseline for comparison with our method.

* **onmt**: includes the [ONMT](https://github.com/OpenNMT/OpenNMT-py) toolkit that we utilized to train our sequence-to-sequence error generators and the error correction model employed by NATAS.

* **pysia**: contains the code of our **Python Sentence Inter-Alignment (PySIA)** toolkit. It constitutes the core part of our contribution and contains method for sentence alignment, training data preparation, and wrapper methods for the sequence-to-sequence training.

* **scripts**: includes a slightly modified version of the punctuation normalization script from the [1 Billion Word Language Model Benchmark](https://www.statmt.org/lm-benchmark/) that we used in our experiments.

* **trdg**: contains the [Text Recognition Data Generator](https://github.com/Belval/TextRecognitionDataGenerator) (TRDG) toolkit employed for text rendering.

Moreover, we extended the basic NAT framework by implementing our error generation methods and included it in the extended sequence labeling model (*flair_ext/nat_sequence_tagger_model.py*). Additionally, we modified the trainer class (*flair_ext/trainers/trainer.py*) and extended the former NAT functionality contained in **robust_ner** (see, e.g., *robust_ner/seq2seq.py*).

Furthermore, we added additional data into the **resources** directory including dictionaries extracted from the test sets of the sequence labeling benchmarks that were used by the error correction methods (*resources/dictionaries*), fonts that were utilized by the text rendering module (*resources/fonts*), and edit operations and checksums required to recreate and validate the noisy sequence labeling data sets used in our experiments (*resources/conversion*). 

Note that *FLAIR*, *NA*TAS, *ONMT* and *TRDG* are not included in this repository. See the *Quick Start* section for more information about the installation of the additional dependencies.

Please refer to the description of the remaining components [here](https://github.com/mnamysl/nat-acl2020#project-structure).



## Quick Start

### Prerequisites

1. Please install the python packages as shown below:
```
pip install -r requirements.txt
```

2. To use Hunspell, you need to install the [required packages](http://hunspell.github.io/):
```
sudo apt-get install hunspell hunspell-en-us libhunspell-dev python-dev
pip install hunspell
```

3. To use the OCR functionality, please refer to the requirements of the [tesserocr](https://github.com/sirfz/tesserocr) package.

4. If you experience problems installing *Matplotlib*, make sure that the *FreeType* library has been properly installed:

```py
sudo apt-get install python-dev libfreetype6-dev
```

### Project Dependencies

Please download the following projects and move them to the working directory as described below:

1. Download the *FLAIR* framework (v0.5) from [here](https://github.com/zalandoresearch/flair/releases/tag/v0.5), rename the *flair-0.5* directory to *flair* and move it to the working directory.
2. Download the *NATAS* library (v1.0.5) from [here](https://github.com/mikahama/natas/releases/tag/1.0.5), rename it to *natas* and move it to the working directory.
3. Download the *ONMT* toolkit (v1.1.1) from [here](https://github.com/OpenNMT/OpenNMT-py/releases/tag/1.1.1), rename it to *onmt* and move it to the working directory.
4. Download the *TRDG* toolkit (v1.6.0) from [here](https://github.com/Belval/TextRecognitionDataGenerator/releases/tag/v1.6.0), rename it to *trdg* and move it to the working directory.

### Data

Please follow the instruction on these websites to get the original data:

#### Sequence Labeling Data Sets

* **CoNLL 2003**: https://www.clips.uantwerpen.be/conll2003/ner/

* **UD English EWT**: https://universaldependencies.org/treebanks/en_ewt/index.html

#### Text Corpora

* **1 Billion Word Language Model Benchmark**: https://www.statmt.org/lm-benchmark/



## Using the code

The *main.py* python script can be used to reproduce our experiments. In this section, we present the command-line arguments and their usage.

### Command-Line Arguments

In addition to the [original configuration](https://github.com/mnamysl/nat-acl2020#using-the-code), we introduce, or modify/extend the following parameters (__*empasized bold*__ values in the table below):

| Parameter | Description | Value |
| --------------------- | ------------------------ | ---------------------------------------------------------- |
| *--mode* | Execution mode | One of: *train*, __*train_lm*__, *tune*, *eval*, __*sent_gen*__, __*sent_gen_txt*__, __**noisy_crp**__, __*onmt*__, __*ds_restore*__, __*ds_check*__. |
| *--corpus* | Data set to use | One of: conll03_en (default), conll03_de, germeval, __*ud_en*__, __*conll03_en_tess3_01*__, __*conll03_en_tess4_01*__, __*conll03_en_tess4_02*__, __*conll03_en_tess4_03*__, __*conll03_en_typos*__, __*ud_en_tess3_01*__, __*ud_en_tess4_01*__, __*ud_en_tess4_02*__, __*ud_en_tess4_03*__, __*ud_en_typos*__. |
| __*--text_corpus*__ | Text corpus path | The name of a text corpus (default: empty) |
| --*train_mode* | Training mode | One of: __*combined*__ (default), __*not-specified*__. |
|__*--alpha*__|Weight of the data augmentation objective|Floating point value (default: 0.0)|
|__*--beta*__|Weight of the stability training objective|Floating point value (default: 0.0)|
| --*type* | Type of embeddings | One of: __*flair+glove*__ (default), __*flair+wiki*__, *flair*, *bert*, *elmo*, __*glove+char*__, __*wiki+char*__, __*myflair+glove*__, __*myflair*__. |
| --*typos_file* | File containing look-up table with typos. | e.g.: *en.natural*, *moe_misspellings_train.tsv*. |
| __*--correction_module*__ | Spell- or OCR post-correction module | One of: *not-specified* (default), *hunspell*, *natas*. |
| __*--errgen_model*__ | Path to the trained sequence-to-sequence error generator | File path (default: empty). |
| __*--errgen_mode*__ | Error generation mode | One of: *errgen_tok*, *errgen_ch*, *errcorr_tok*. |
| __*--errgen_temp*__ | Sampling temperature | Floating point value (default: 1.0). |
| __*--errgen_topk*__ | Top-K sampling candidates to use | Integer value (default: -1). |
| __*--errgen_nbest*__ | N-best beams to use | Integer value (default: 5). |
| __*--errgen_beam_size*__ | Beam size to use | Integer value (default: 10). |
| __*--seek_file*__ | File name to start with when generating paired data | File name (default: empty). |
| __*--seek_line*__ | Line number to seek when generating paired data | Integer value (default: 0). |
| __*--storage_mode*__ | Embedding storage mode | One of: *auto* (default), *gpu*, *cpu*, *none*. |
| **--use_amp** | Use mixed-precision training | No parameters, turned off by default. |
| -*h* | Print help | No parameters. |
| __*--lm_type*__ | Type of the language model to use | One of: *forward*, *backward* |
| __*--num_layers*__ | The number of network layers | Integer value (default: 1) |
| __*--patience*__ | The number of epochs with no improvement until annealing the learning rate | Used only in the case of LM-training. Integer value (default: 50) |
| __*--anneal_factor*__ | The factor by which the learning rate is annealed | Used only in the case of LM-training. Integer value (default: 0.25) |
| __*--sequence_length*__ | Truncated BPTT window length | Used only in the case of LM-training. Integer value (default: 250) |

### Command-Line Usage

The basic command-line calls can be found [here](https://github.com/mnamysl/nat-acl2020#training-a-model-from-scratch). In addition, we present how to use the functionality related to our approach.

####  Parallel Data Set Generation

Assuming that the original *<text_corpus>* is stored in *resources/corpora/<text_corpus>*, the following call will run the parallel data generation procedure. The results will be stored in *results/generated/<text_corpus>* afterwards.
```
python3 main.py --mode sent_gen_txt --text_corpus <text_corpus>
```

#### Noisy Sequence Labeling Data Set Generation

Similarly, the following command will read the sentences from the sequence labeling data set *<seq_lab_corpus>* in *resources/task/<seq_lab_corpus>*, render them, and store the results in *results/generated/<seq_lab_corpus>*.
```
python3 main.py --mode sent_gen --corpus <seq_lab_corpus>
```
We can move the resultant *<seq_lab_corpus>* to the *results/tasks* directory, so we can use it for evaluation or training.

#### Preprocessing

The parallel data set needs to be normalized prior to using it further. To this end, we apply the normalization script as follows:
```
scripts/normalize-punctuation.sh <text_corpus>
```
As before, we assume that the *<text_corpus>* is located under *results/generated/<text_corpus>*.

#### Restoring the Noisy Data Sets

To restore the noisy data sets used in our experiments, we execute the following command:
```
python3 main.py --mode ds_restore --corpus <seq_lab_corpus>
```
where the *<seq_lab_corpus>* is either *conll03_en* or *ud_en*. Our scripts will then recreate the underlying data based on the sequences of edit operations stored in *resources/conversion/<seq_lab_corpus>/<train/dev/test>_ops.txt*. 

Moreover, we can validate the generated data using the checksums distributed with our library by running:
```
python3 main.py --mode ds_check --corpus <seq_lab_corpus>
```
The resultant train/test/dev splits with the *_restored* suffix can be copied to the *resources/task* folder to be employed, e.g., for evaluation or training.

#### Sequence-to-Sequence Model Training

The normalized parallel data can be utilized to train a sequence-to-sequence error generation or correction model. The following command will start the procedure that splits the parallel data, converts it to the format used by ONMT and starts the training of the model:
```
python3 main.py --mode onmt --text_corpus <text_corpus>
```

The results will be stored in the *results/generated/<text_corpus>/<model_name>* directory, where *<model_name>* is constructed as follows:
```
model_<mode>_<size>
```
where *<mode>* is one of the following: *errgen_tok*, *errgen_ch*, or *errcorr_tok*, and *<size>* is the size of the parallel data set (*100*, *1k*, *10k*, *100k*, *1M*, *10M*), e.g., *model_errgen_tok_100k*.

#### NAT with the Sequence-to-Sequence Error Generator

Having the trained error generation model, we can utilize it to train a downstream sequence labeling model *<model_name>* using NAT technique. The following command will start the stability training of a *<model_name>* on the English CoNLL 2003 training data using the error generator model stored in *results/generated/<text_corpus>/model_errgen_tok_100k/<text_corpus>_step_16000.pt*, the token-to-token generation mode, the sampling temperature 1.1, and top-k = 10 best candidates.

```
python3 main.py --mode train --model <model_name> --corpus conll03_en --type flair+glove --errgen_model results/generated/<text_corpus>/model_errgen_tok_100k/<text_corpus>_step_16000.pt --errgen_mode errgen_tok --errgen_temp 1.1 --errgen_topk 10 --beta 1.0 --use_misspell_lut
```

The models will be stored in the *resources/taggers* directory.

####  Noisy Corpus Extraction for Noisy Language Modeling (NLM)

To extract the data for NLM training, we can use the following command:
```
python3 main.py --mode noisy_crp --text_corpus <text_corpus>
```
where the *<text_corpus>* refers to the data stored in *resources/corpora/<text_corpus>*. As a result, it will create two sub-directories: *<text_corpus>_pairs_norm_org\__<max_lines>* and *<text_corpus>_pairs_norm_rec\_<max_lines>*, where the former and the latter will contain the clean- and the noisy-part of the parallel text corpus, respectively. The *<max_lines>* parameter refers to the maximum number of lines that need to be extracted from the source texxt file and is unbounded by default, but it can be adjusted in the code if necessary.

#### NLM Embeddings Training

Previously extracted noisy corpus could be used as the source of text for NLM training. Except for the source of textual input, the NLM training follows the standard routines of the FLAIR library. For reference, please refer to the [instructions](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md) on how to prepare the data for the language model training. Subsequently, the NLM training can be performed using NAT framework with the following call:

```
python3 main.py --mode train_lm --text_corpus <lm_text_corpus> --lm_type <lm_type> --model custom_<lm_type>
```
where *<lm_text_corpus>* is the text corpus prepared for LM training located in the *resources/corpora* directory and *<lm_type>* refers to the type of a LM to be trained (either *forward* or *backward*). The results of this call will be stored in the *resources/language_models/custom_<lm_type>* directory.

#### NAT Training using NLM Embeddings

Previously trained NLM embeddings can be used to train a NAT model as follows:
```
python3 main.py --model train --corpus <data_set> --model <model_name> --type <embeddings_type>
```
where *<embeddings_type>* refers to the type of embeddings to be employed - *myflair* and *myflair+glove* values are in-built aliases for the custom flair embedding models. Please refer to the *init_embeddings()* function in *main.py* for further details. 

#### Error Correction using Hunspell or Natas

Finally, we can utilize a specific error correction module for evaluation in the following way:

```
python3 main.py --mode eval --model model_name --corpus conll03_en_tess4_01 --col_idx 2 --text_idx 1 --correction_module hunspell
```
Additional remarks: *conll03_en_tess4_01* is a noisy data set generated using our approach and derived from the original English CoNLL 2003 benchmark. To utilize it we need to specify two additional parameters: *--col_idx* and *--text_idx* that represent the column index of the class labels and the text column, respectively. The first column in the generated noisy data sets always corresponds to the possibly erroneous text and the second column contains the error-free tokens. In the example above, we will use the noisy tokens.



## License

This project is licensed under the MIT License - see the *LICENSE* file for details
