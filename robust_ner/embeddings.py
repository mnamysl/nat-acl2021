import logging
import torch

from typing import List
from flair.data import Sentence

log = logging.getLogger("flair")


def clear_embeddings(sentences):
    """
    Clears the embeddings and the stored tokenized versions for each sentence on the given list.
    """

    from flair.training_utils import store_embeddings

    # store_embeddings(sentences, storage_mode="none")

    # A simplified version of store_embeddings(sentences, "none").
    # It does not set the global 'flair.embedding_storage_mode' flag
    for s in sentences:
        s.clear_embeddings()

    # Clears cached tokenized sentence (otherwise it causes problems in flair/embeddings.py line:1839)
    # when Flair embeddings are retrieved from the look-up table using wrong offsets.
    for s in sentences:
        s.tokenized = None


def check_embeddings(sentList1: List[Sentence], sentList2: List[Sentence], embed1: torch.tensor, embed2: torch.tensor):
    """
    Checks embeddings of the original and perturbed sentences.
    Returns false if any token of the first sentence has the same embeddings but different text as the
    corresponding token of the second sentence
    """

    for i, (s1, s2) in enumerate(zip(sentList1, sentList2)):
        for j, (tok1, tok2) in enumerate(zip(s1, s2)):
            text1, text2 = tok1.text, tok2.text
            e1, e2 = embed1[i][j], embed2[i][j]
            
            diff = torch.sum(e1 - e2).item()
            if text1 != text2 and diff == 0.0:
                log.error(
                    f"ERROR: same embeddings, different text! "
                    f"diff={diff} text1: {text1} text2: {text2}"
                )            
                return False

    return True