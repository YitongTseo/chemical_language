from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque


def map_initial_words_to_features(mol, words_to_embed, radius=2, nBits=1024):
    """
    Maps words from the Morgan fingerprint (words_to_embed) to specific bonds,
    serving as the initial, strong chemical relevance signal.

    Args:
        mol (Chem.Mol): The RDKit molecule object.
        words_to_embed (list): Tuples of (word, bit_idx, ranking).
        radius (int): Morgan fingerprint radius.
        nBits (int): Morgan fingerprint size.

    Returns:
        tuple: (bit_to_word, bond_to_word_map, bit_info)
    """
    # 1. Get fingerprint information
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=info)

    # 2. Map active bits to their assigned words
    bit_to_word = {bit_idx: word for word, bit_idx, ranking in words_to_embed}

    # 3. Assign words to bonds based on fingerprint environment
    # This acts as the initial, strong chemical relevance.
    bond_to_word_map = {}
    for bit_idx, word in bit_to_word.items():
        if bit_idx in info:
            atom_idx, rad = info[bit_idx][0]
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
            for bond_idx in env:
                # Assign the word to the first bond found in the environment 
                # that hasn't been assigned a word yet.
                if bond_idx not in bond_to_word_map:
                    bond_to_word_map[bond_idx] = word
                    break
    return bit_to_word, bond_to_word_map, info


def generate_poem_by_molecular_walk(
    mol, 
    initial_bond_word_map, 
    probability_matrix, 
    top_words_voc, 
    molecular_relevance_weight=0.3
):
    random.seed(42)
    np.random.seed(42)

    """
    Refines the words assigned to the chemically relevant bonds (initial_bond_word_map)
    by walking the molecular graph and using the Markov chain to choose new words.
    """
    # Setup for Markov Indexing
    word_to_index = {word: i for i, word in enumerate(top_words_voc)}
    
    # 1. Initialization
    
    # These are the only bonds we care about (the target indices)
    target_bond_indices = set(initial_bond_word_map.keys())
    
    # These are the bonds we still need to process/re-assign
    unprocessed_bonds = target_bond_indices.copy()
    
    # The result map
    final_bond_word_map = {}
    
    # Start the walk from the bond with the lowest index in the target set
    if not target_bond_indices:
        return final_bond_word_map
    
    # Arbitrarily pick a start bond and immediately remove it from the unprocessed set
    start_bond_idx = random.choice(list(target_bond_indices))
    unprocessed_bonds.remove(start_bond_idx)
    
    # Queue for BFS: starts with the first bond
    queue = deque([start_bond_idx])
    
    # The word for the starting bond is guaranteed to be its initial map
    current_word = initial_bond_word_map[start_bond_idx]
    final_bond_word_map[start_bond_idx] = current_word # Keep the initial word for the start
    
    
    # --- Helper Function for BFS ---
    def find_adjacent_target_bonds(bond_idx):
        """Finds adjacent bonds that are in the target set and are unprocessed."""
        neighbor_bonds = []
        bond = mol.GetBondWithIdx(bond_idx)
        
        # Check atoms at both ends of the bond
        for atom_idx in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
            atom = mol.GetAtomWithIdx(atom_idx)
            # Find all other bonds connected to this atom
            for neighbor_bond in atom.GetBonds():
                neighbor_idx = neighbor_bond.GetIdx()
                # Check if it is a target bond AND hasn't been processed yet
                if neighbor_idx in unprocessed_bonds:
                    neighbor_bonds.append(neighbor_idx)
                    unprocessed_bonds.remove(neighbor_idx) # Mark as processed/queued
        return neighbor_bonds

    # 2. Main Walk Loop
    while queue or unprocessed_bonds:
        # A. Connected Traversal (BFS)
        while queue:
            current_bond_idx = queue.popleft()
            
            # Find neighbor bonds to re-assign words to
            neighbors = find_adjacent_target_bonds(current_bond_idx)
            
            # The current word for Markov transition is the one we *just* assigned/reassigned
            current_word = final_bond_word_map[current_bond_idx] 
            
            for next_bond_idx in neighbors:
                # i. Get the Markov Probability Vector P_markov (Flow)
                if current_word in word_to_index:
                    current_idx = word_to_index[current_word]
                    P_markov = probability_matrix[current_idx, :]
                else:
                    P_markov = np.ones(len(top_words_voc)) / len(top_words_voc) 

                # ii. Get the Molecular Relevance Vector R_mol (Bias)
                R_mol = np.zeros(len(top_words_voc))
                chem_word = initial_bond_word_map[next_bond_idx]
                if chem_word in word_to_index:
                    R_mol[word_to_index[chem_word]] = 1.0 
                
                # iii. Weighted Combination
                P_final = (1 - molecular_relevance_weight) * P_markov + \
                          molecular_relevance_weight * R_mol

                # ----------------------------------------------------
                # 💡 New: NON-REPETITION CONSTRAINT
                
                # 1. Identify indices of used words
                used_words = set(final_bond_word_map.values())
                used_indices = [word_to_index[word] for word in used_words if word in word_to_index]
                
                # 2. Hard constraint: Set probability of used words to 0
                P_final[used_indices] = 0.0
                
                # 3. Check if we've zeroed out all options (shouldn't happen unless target is very large)
                if np.sum(P_final) == 0:
                    # Fallback: if all words are used, re-enable the *chemical* word only.
                    # This is a robust way to ensure a choice can still be made.
                    if chem_word in word_to_index:
                         P_final[word_to_index[chem_word]] = 1.0
                    else:
                         # Fallback to pure uniform if even the chem_word is gone
                         P_final = np.ones(len(top_words_voc))
                         
                # Normalization
                P_final /= P_final.sum()
                # ----------------------------------------------------

                # iv. Selection and Update
                chosen_word_idx = np.random.choice(len(top_words_voc), p=P_final)
                chosen_word = top_words_voc[chosen_word_idx]

                # Assign (or re-assign) the word
                final_bond_word_map[next_bond_idx] = chosen_word
                
                # Add the newly re-assigned bond to the queue for its neighbors to be checked
                queue.append(next_bond_idx)

        # B. Disconnected Fragment Handling
        if unprocessed_bonds:
            # Randomly pull a bond from a disconnected, unvisited component
            random_bond_idx = random.choice(list(unprocessed_bonds))
            unprocessed_bonds.remove(random_bond_idx)
            
            # Start a new sub-walk with this bond. This word is assigned using the 
            # R_mol bias alone, as there is no 'current_word' flow from a neighbor.
            chem_word = initial_bond_word_map[random_bond_idx]
            
            # Set the new flow's starting word
            current_word = chem_word 
                
            final_bond_word_map[random_bond_idx] = current_word
            
            # Start the new BFS component walk
            queue.append(random_bond_idx)
    return final_bond_word_map