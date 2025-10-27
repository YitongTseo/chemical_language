from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def get_fingerprint_array(smiles, show_hydrogens, nBits=1024):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    for mol in mols:
        if not show_hydrogens:
            mol = Chem.RemoveHs(mol)
        else:
            mol = Chem.AddHs(mol)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=nBits) for m in mols]
    fps_array = np.array([np.frombuffer(fp.ToBitString().encode('utf-8'), 'u1') - ord('0') for fp in fps])
    return fps_array

def mol_to_words(mol, sorted_column_indices, top_words, nBits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
    fp_array = np.frombuffer(fp.ToBitString().encode('utf-8'), 'u1') - ord('0')
    active_columns = np.where(fp_array == 1)[0]
    word_rankings = [(col, list(sorted_column_indices).index(col)) for col in active_columns]
    words_with_col= [(top_words[ranking][0], col, ranking) for col, ranking in word_rankings]
    return words_with_col
