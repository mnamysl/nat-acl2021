# normalize paired data set
CORPUS_NAME=${1:-"1bilion.1"}
DATA_DIR="results/generated/${CORPUS_NAME}/"
cat "${DATA_DIR}/${CORPUS_NAME}_pairs.txt" | scripts/normalize-punctuation.perl -l en > "${DATA_DIR}/${CORPUS_NAME}_pairs_norm.txt"
cat "${DATA_DIR}/${CORPUS_NAME}_align.txt" | scripts/normalize-punctuation.perl -l en > "${DATA_DIR}/${CORPUS_NAME}_align_norm.txt"

# normalize conll-format data set
# DATA_DIR=${1:-"resources/tasks/ud_en_tess4_02/"}

# cat "${DATA_DIR}/test.csv" | scripts/normalize-punctuation.perl -l en > "${DATA_DIR}/test_norm.csv"
# cat "${DATA_DIR}/dev.csv" | scripts/normalize-punctuation.perl -l en > "${DATA_DIR}/dev_norm.csv"
# cat "${DATA_DIR}/train.csv" | scripts/normalize-punctuation.perl -l en > "${DATA_DIR}/train_norm.csv"
