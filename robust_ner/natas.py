def correct_text_with_natas(input, ext_dictionary=None, model_path="path/to/your/seq2seq/model.pt", verbose=False):
    """
    Corrects the given input text with using the Natas toolkit.
    Note: it needs to manually set the path to the trained sequence-to-sequence correction model 
          as the default value for the parameter 'model_path'.
    """

    input = input.strip()
    
    if not input:
        # do not try to correct if the input is empty
        return input

    import natas
    from natas.normalize import _normalize, wiktionary
    dictionary=wiktionary

    # extend the default dictionary
    if ext_dictionary != None:
        for tok in ext_dictionary:
            dictionary.add(tok)        
    
    output = input

    is_dict_word = input in dictionary

    ok = is_dict_word or natas.is_correctly_spelled(input, dictionary=dictionary)
    if not ok:        
        try:
            suggestions = _normalize([input], model_path, all_candidates=False, dictionary=dictionary)
        except RuntimeError:
            print(f"RuntimeError on input: '{input}'")
            raise
        except IndexError: # sometimes occurs for long urls (UD_en)
            print(f"IndexError on input: '{input}'")            
            suggestions = []
        
        if len(suggestions) > 0 and len(suggestions[0]) > 0:
            # take the best suggestion from the NMT model
            output = suggestions[0][0]
        
    return output