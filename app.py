import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import matplotlib.pyplot as plt
import io # Needed to save plot in memory
# Copy your existing functions (row_to_words and typewriter_molecule_art_v2) here.

# --- Streamlit Interface ---
st.title("Chemical Language")

# 1. File Uploads
uploaded_csv = st.file_uploader("Upload Molecule Data (a list of smiles strings with header 'smiles')", type="csv")
uploaded_txt = st.file_uploader("Upload Plain Text Corpus (can be anything)", type="txt")

if uploaded_csv and uploaded_txt:
    # Load and process data (This mirrors your initial setup)
    df = pd.read_csv(uploaded_csv)
    top_words = pd.read_csv(uploaded_txt, delimiter='\t')
    
    # Data cleaning (as in your original script)
    df = df.dropna(subset=['smiles'])
    df = df[df['name']!= '[]']
    df['smiles'] = df['smiles'].apply(lambda x: x[2:-2])
    mols = [Chem.MolFromSmiles(s) for s in df['smiles']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    fps_array = np.array([np.frombuffer(fp.ToBitString().encode('utf-8'), 'u1') - ord('0') for fp in fps])
    column_populations = fps_array.sum(axis=0)
    sorted_column_indices = np.argsort(column_populations)[::-1]
    
    # 2. Molecule Selection
    smiles_list = df['smiles'].tolist()
    selected_smiles = st.selectbox("Select or Enter SMILES string:", smiles_list + ["Custom SMILES"])
    
    if selected_smiles == "Custom SMILES":
        smiles_input = st.text_input("Enter SMILES string:", "C1=CC=CC=C1")
    else:
        smiles_input = selected_smiles

    if smiles_input:
        # 3. Parameter Sliders (Use st.sidebar for a cleaner look)
        st.sidebar.header("Art Parameters")
        target_length = st.sidebar.slider("Target Word Length", 5, 40, 20)
        font_size = st.sidebar.slider("Font Size", 10, 50, 21)
        vertical_jitter = st.sidebar.slider("Vertical Jitter", 0.0, 0.1, 0.03)
        letter_spacing = st.sidebar.slider("Letter Spacing", 0.0, 0.02, 0.0085)

        # Get words for the selected molecule
        try:
            mol_index = df[df['smiles'] == smiles_input].index[0]
            words_to_embed = row_to_words(fps_array[mol_index]) # Use your existing row_to_words logic
        except Exception:
            st.warning("Could not compute fingerprints for the entered SMILES or find it in the data. Using placeholder words.")
            words_to_embed = [('WATER', 10), ('STONE', 50), ('BREAD', 100), ('PEARL', 200), ('JUAN', 300), ('KINO', 400)]

        # --- Generate Art ---
        if st.button("Generate Typewriter Art"):
            st.subheader(f"Rendering: {smiles_input}")
            
            # The function must now return the Matplotlib figure object
            fig, bond_words = typewriter_molecule_art_v2(
                smiles=smiles_input,
                words_to_embed=words_to_embed,
                target_length=target_length,
                font_size=font_size,
                vertical_jitter=vertical_jitter,
                letter_spacing=letter_spacing,
                # Pass other arguments as needed
            )
            
            # Display the figure in Streamlit
            st.pyplot(fig)
            st.caption(f"Words used on bonds: {list(bond_words.values())}")