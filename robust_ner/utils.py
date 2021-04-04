
def _replace_chars_in_corpus(corpus, oldchar, newchar, verbose=True):
    _replace_chars_in_dataset(corpus.train, oldchar, newchar)
    _replace_chars_in_dataset(corpus.test, oldchar, newchar)
    _replace_chars_in_dataset(corpus.dev, oldchar, newchar)

    if _find_chars_in_dataset(corpus.train, oldchar, verbose):
        print(f"ERROR: replacing chars in the train set failed!")
        return False
    if _find_chars_in_dataset(corpus.dev, oldchar, verbose):
        print(f"ERROR: replacing chars in the dev set failed!")
        return False
    if _find_chars_in_dataset(corpus.test, oldchar, verbose):
        print(f"ERROR: replacing chars in the test set failed!")
        return False

def _replace_chars_in_dataset(dataset, oldchar, newchar):
    for sentence in dataset:
        for token in sentence:
            token.text = token.text.replace(oldchar, newchar)

def _find_chars_in_dataset(dataset, oldchar, verbose=True):
    for sentence in dataset:
        for token in sentence:
            if oldchar in token.text:
                if verbose:
                    print(f"Found '{oldchar}' in '{token.text}'")
                return True
    return False