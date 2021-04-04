import hunspell
import codecs

def _normalize_utf8(text):
    """
    Normalizes text by replacing all non-latin-1 characters.
    """

    text = codecs.encode(text, 'latin-1', 'replace')
    text = codecs.decode(text, 'latin-1', 'replace')
    return text

def init_hunspell(lang, dictionary=None):
    """
    Initializes and returns the Hunspell spell checker object.
    """
    
    if lang == "en":
        spell_check = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
    elif lang == "de":
        spell_check = hunspell.HunSpell('/usr/share/hunspell/de_DE.dic', '/usr/share/hunspell/de_DE.aff')
    else:
        spell_check = None

    if dictionary != None:
        for token in dictionary:
            spell_check.add(_normalize_utf8(token))

    return spell_check


def correct_text_with_hunspell(input, spellcheck, dictionary=None, verbose=False):
    """
    Checks whether the input is correctly spelled and correct it otherwise.
    Returns the corrected input.
    """
    
    # Remove all unicode characters from the input (Hunspell has a problem with them)
    # https://stackoverflow.com/questions/11339955/string-encoding-and-decoding
    input = _normalize_utf8(input)

    output = input

    ok = spellcheck.spell(input)
    if not ok:
        suggestions = spellcheck.suggest(input)
        if len(suggestions) > 0:
            output = suggestions[0]
            if verbose and input != output:
                print(f"{input} -> {output}")

    return output