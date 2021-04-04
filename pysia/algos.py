
def iterative_levenshtein(s, t):
    """ 
    iterative_levenshtein(s, t) -> ldist
    ldist is the Levenshtein distance between the strings s and t.
    For all i and j, dist[i,j] will contain the Levenshtein distance 
    between the first i characters of s and the first j characters of t.

    Credit: https://www.python-course.eu/levenshtein_distance.php
    """
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings 
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i
    
    row, col = 0, 0
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution

    ldist = dist[row][col]

    edit_ops = list()

    dist_last = ldist

    ldist2 = 0
    while row > 0 or col > 0:        
        dist_diag = dist[row-1][col-1] if row > 0 and col > 0 else ldist + 1
        dist_up = dist[row-1][col] if row > 0 else ldist + 1
        dist_left = dist[row][col-1] if col > 0 else ldist + 1
        dist_min = min(dist_diag, dist_up, dist_left)
        
        if dist_diag == dist_min and dist_min == dist_last: # no change            
            row -= 1
            col -= 1
            edit_ops.insert(0, "-")
        elif dist_up == dist_min: # deletion
            row -= 1
            ldist2 += 1
            edit_ops.insert(0, "d")
        elif dist_left == dist_min: # insertion
            col -= 1
            ldist2 += 1
            edit_ops.insert(0, "i")
        elif dist_diag == dist_min and dist_min < dist_last: # substitution            
            row -= 1
            col -= 1
            ldist2 += 1
            edit_ops.insert(0, "s")
        
        dist_last = dist_min

    if ldist != ldist2:
        print(f"WRONG!!! {ldist}/{ldist2}")
        for r in range(rows):
            print(dist[r])
        exit(-1)
 
    return ldist, ''.join(edit_ops)

def _insert_char_at_pos(text, pos, ch):
    """
    Inserts a character at the given position in string.
    """
    return text[:pos] + ch + text[pos:]

def align_text(text, edit_ops, align_op, mark=172):
    """
    Creates an aligned version of a given string based on a sequence of edit operations.
    """    
    result = text
    for idx, op in enumerate(edit_ops):
        if op == align_op:
            result = _insert_char_at_pos(result, idx, chr(mark))
        
    return result