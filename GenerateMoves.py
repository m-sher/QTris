def generate_all_finesse_moves_dict():
    # IGNORE WITH AUTOREGRESSIVE
    
    # Define option lists (order matters)
    possible_holds = ["", "h"]
    
    standard_moves = ["L", "Lr", "ll", "l", "", "r", "rr", "Rl", "R",
                      "L1", "ll1", "l1", "1", "r1", "rr1", "Rl1", "R1",
                      "Lc", "llc", "lc", "c", "rc", "rrc", "Rlc", "Rc",
                      "La", "lla", "la", "a", "ra", "rra", "Rla", "Ra",
                      "cL", "cR", "aL", "aR"]
                    
    spin_moves = ["", "c", "a", "1", "cc", "aa", "c1", "a1"]

    # This dictionary will hold the mapping:
    # key: move string (ending with "H")
    # value: (hold, standard, spin)
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