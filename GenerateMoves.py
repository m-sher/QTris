def generate_all_finesse_moves_dict():
    # IGNORE WITH AUTOREGRESSIVE
    
    # Define option lists (order matters)
    possible_holds = ["", "h"]
    standard_horizontal = ["", "L", "R", "Lr", "Rl", "ll", "rr", "l", "r"]
    standard_rotations = ["", "c", "a", "1"]
    spin_moves = ["", "c", "a", "1", "cc", "aa", "c1", "a1"]

    standard_moves = []
    for h in standard_horizontal:
        for r in standard_rotations:
            standard_moves.append(h+r)
    for h in standard_horizontal:
        for r in standard_rotations:
            if r+h not in standard_moves:
                standard_moves.append(r+h)

    # This dictionary will hold the mapping:
    # key: move string (ending with "H")
    # value: [hold, standard, slide, spin]
    str_to_ind = {}
    ind_to_str = {}
    
    for hold_index, hold in enumerate(possible_holds):
        for std_index, standard in enumerate(standard_moves):
            for spin_index, spin in enumerate(spin_moves):
                move_str = hold + standard + (("s" + spin) if spin else "") + "H"
                indices = (hold_index, std_index, spin_index)
                str_to_ind[move_str] = indices
                ind_to_str[indices] = move_str
        
    return str_to_ind, ind_to_str

# str_to_ind, ind_to_str = generate_all_finesse_moves_dict()
# print(len(str_to_ind))

def generate_key_dict():
    ind_to_str = {
        0: 'S',
        1: 'h',
        2: 'l',
        3: 'r',
        4: 'L',
        5: 'R',
        6: 'a',
        7: 'c',
        8: '1',
        9: 's',
        10: 'H',
        11: 'P'
    }
    
    str_to_ind = {val: key for key, val in ind_to_str.items()}
    
    return ind_to_str, str_to_ind