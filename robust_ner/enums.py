from enum import Enum

class TrainingMode(Enum):
    """
    Training mode (one of: not-specified, combined)
    """

    NotSpecified = 'not-specified'
    Combined = 'combined'

    def __str__(self):
        return self.name


class EvalMode(Enum):
    """
    Evaluation mode (one of: standard, misspellings)
    """

    Standard = 'standard'
    Misspellings = 'misspellings'

    def __str__(self):
        return self.name


class MisspellingMode(Enum):
    """
    Misspellings mode (one of: rand, cmx, typos, seq2seq)
    """

    Random = 'rand'
    ConfusionMatrixBased = 'cmx'
    Typos = 'typos'
    Seq2Seq = 'seq2seq'

    def __str__(self):
        return self.name


class CorrectionMode(Enum):
    """
    Text correction mode (one of: not-specified, hunspell, natas)
    """

    NotSpecified = "not-specified"
    Hunspell = 'hunspell'
    Natas = 'natas'
    
    def __str__(self):
        return self.name


class Seq2SeqMode(Enum):
    """
    Seq2seq mode (one of: errcorr, errcorr_ch, errcorr_tok, errgen, errgen_ch, errgen_tok)
    """

    ErrorCorrectionTok = 'errcorr_tok'
    ErrorGenerationCh = 'errgen_ch'
    ErrorGenerationTok = 'errgen_tok'

    def __str__(self):
        return self.name