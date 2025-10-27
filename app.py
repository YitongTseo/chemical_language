import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import matplotlib.pyplot as plt
import io # Needed to save plot in memory
from text_utils import get_top_words, create_numpy_transition_matrix
from molecule_utils import get_fingerprint_array, mol_to_words
from render_utils import typewriter_molecule_art
from combining_utils import map_initial_words_to_features, generate_poem_by_molecular_walk


# Note: Using your provided text/smiles loading logic
with open('the_pearl.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('seashell_smiles.txt', 'r', encoding='utf-8') as f:
    smiles = f.read()


# --- Streamlit Interface ---
st.title("Chem Lang")

# 1. Horizontal Corpus Layout (Row 1)
col1, col2 = st.columns(2)

with col1:
    # --- Molecule Corpus Input ---
    molecule_data = st.text_area(
        "Molecules (SMILES strings, one per line)",
        value=smiles,
        height=250,
        help="Enter a list of SMILES strings. Each string should be on a new line."
    )

with col2:
    # --- Text Corpus Input ---
    text_corpus = st.text_area(
        "Text Corpus (can be anything)",
        value=text,
        height=250,
        help="Enter the text body the molecular language will draw from."
    )

st.write("---") # Separator for visual clarity

# 2. Horizontal Configuration Layout (Row 2)
# We use the same columns for alignment
col1_config, col2_config = st.columns(2)

with col1_config:    
    # Select box for top_n words (powers of 2)
    top_n_options = [2**i for i in range(8, 13)] # 256, 512, 1024, 2048
    top_n_words = st.selectbox(
        "# of Functional Group Bins",
        options=top_n_options,
        index=top_n_options.index(1024),
        key='top_n',
        help="Choose the number of functional group to word mappings."
    )
    show_hydrogens = st.checkbox(
        "Include Hydrogens",
        value=False,
        key='show_hydrogens',
        help="hmm... there might be too many hydrogens..."
    )

with col2_config:
    
    # List of words to scrub
    words_to_scrub = st.text_area(
        "Keywords to Scrub",
            value=','.join([
            'Kino', 'Juana', 'Coyotito', 'Juan', 'Tomás',
            'La', 'Paz', 'Gulf', 'California', 'Mexico'
        ]),
        height=68,
        help="Enter any specific words you want to exclude, separated by commas, case insensitive"
    )
    
    # Boolean input for stop word filtering
    filter_stopwords = st.checkbox(
        "Filter Stop Words",
        value=True,
        key='filter_stop',
        help="Check to filter out common English stop words (e.g., 'a', 'the', 'his', 'not', etc.)"
    )


# 3. Big, Eye-Catching Button
st.markdown("""
<style>
/* Targets the specific Streamlit primary button and increases its size */
div.stButton > button:first-child {
    font-size: 20px;
    height: 3em;
    width: 100%;
}
</style>""", unsafe_allow_html=True)

if st.button("Create Mapping", type='primary'):
    st.session_state.ready_to_process = True
    
# --- Python Logic Placeholder ---
if st.session_state.get('ready_to_process', False):
    # st.success("Mapping made! Ready now for molecule by molecule translation")

    words = words_to_scrub.split(',')
    top_words = get_top_words(
        text=text_corpus,
        top_n=top_n_words,
        scrubbed_words=words_to_scrub.split(','),
        filter_stop_words=filter_stopwords,
    )
    probability_matrix, top_words_voc = create_numpy_transition_matrix(text=text_corpus, top_words_list=top_words)
    molecule_data = '\n'.join([
        smile.strip()
        for smile in molecule_data.split('\n')
        if smile.strip() and not smile.strip().startswith('#')
    ])
    fps_array = get_fingerprint_array([mol for mol in molecule_data.split('\n') if len(mol) > 0], nBits=top_n_words, show_hydrogens=show_hydrogens)
    column_populations = fps_array.sum(axis=0)
    sorted_column_indices = np.argsort(column_populations)[::-1]

    # --- Target Molecule Input ---
    target_smiles = st.text_input(
        "Molecule to Translate",
        value="CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C",
        help="Enter the single SMILES string you wish to translate. NOTE: can be whatever you want! Doesn't have to have been part of the SMILES mapping corpus above, but the idea is if its in distribution you'll get a more appropriate translation."
    )
    # 1. New Slider for Molecular Relevance Weight
    molecular_relevance_weight = st.slider(
        "Text vs Chemical Importance Weighting (0 = Full Text Importance, 1 = Full Chemical Importance)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key='molecular_relevance_weight',
        help="Controls the balance between chemical structure and textual flow. 0.0 means the word selection is purely driven by the textual Markov chain, i.e., how the words naturally flow in the source text. 1.0 means the selection is purely driven by the molecular structure relevance, i.e., which words match closest to the functional group at hand."
    )

    # 2. Collapsible Section for Optional Rendering Parameters
    with st.expander("Parameters if you want to tweak the render (or stop it from dancing)"):
        st.subheader("Molecular Drawing Configuration")
        
        # 3. Drawing Configuration Layout (3 Columns)
        draw_col1, draw_col2, draw_col3 = st.columns(3)

        # --- Column 1: Font and Atom Display ---
        with draw_col1:
            font_options = ["American Typewriter", "Courier New", "Monospace", "Arial", "Times New Roman"]
            selected_font = st.selectbox(
                "Font",
                options=font_options,
                index=0,
                key='font_select'
            )
            
            max_char_cutoff = st.number_input(
                "Max Character Cutoff Length",
                min_value=1,
                max_value=100,
                value=20,
                key='max_char_cutoff'
            )
            
            show_atoms = st.checkbox(
                "Show Atoms",
                value=False,
                key='show_atoms'
            )
            
            atom_size = st.number_input(
                "Atom Size",
                min_value=1,
                max_value=30,
                value=3,
                key='atom_size',
                disabled=not(show_atoms),
            )


        # --- Column 2: Jitter and Spacing ---
        with draw_col2:
            letter_spacing = st.number_input(
                "Letter Spacing",
                min_value=0.0,
                max_value=1.0,
                value=0.0085,
                step=0.0001,
                format="%.4f",
                key='letter_spacing'
            )
            
            letter_horizontal_jitter = st.number_input(
                "Horizontal Jitter",
                min_value=0.0,
                max_value=1.0,
                value=0.03,
                step=0.01,
                format="%.2f",
                key='h_jitter'
            )
            
            letter_vertical_jitter = st.number_input(
                "Vertical Jitter",
                min_value=0.0,
                max_value=1.0,
                value=0.03,
                step=0.01,
                format="%.2f",
                key='v_jitter'
            )

            aromatic_circle_offset = st.number_input(
                "Aromatic Circle Offset",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                format="%.1f",
                key='aromatic_offset'
            )

        # --- Column 3: Scaling and Font Size ---
        with draw_col3:
            autoscaling = st.checkbox(
                "Autoscale Font Size",
                value=True,
                key='autoscaling'
            )
            
            # Font size input, conditionally disabled
            font_size = st.number_input(
                "Font Size (px)",
                min_value=5,
                max_value=50,
                value=24,
                key='font_size',
                disabled=autoscaling, # This handles the "greyed out" feature
                help="Disabled if 'Autoscale Font Size' is checked."
            )

            dashed_bond_length = st.number_input(
                "Dashed Bond Length",
                min_value=1.0,
                max_value=30.0,
                key='dashed_bond_length',
                value=4.5,
            )

    st.write("---")
    
    if st.button("Translate Molecule!", type='primary'):
        # Here is where you would call your typewriter_molecule_art_v2 function 
        # using all the configuration inputs defined above.
        # st.success("Translating...")
        mol = Chem.MolFromSmiles(target_smiles)
        if not show_hydrogens:
            mol = Chem.RemoveHs(mol)
        else:
            mol = Chem.AddHs(mol)
        # AllChem.Compute2DCoords(mol)

        words_to_embed = mol_to_words(mol, sorted_column_indices, top_words, nBits=top_n_words)
        _bit_to_word, bond_to_word_map, _info = map_initial_words_to_features(mol, words_to_embed, nBits=top_n_words)
        final_bond_word_map = generate_poem_by_molecular_walk(
            mol, 
            bond_to_word_map, 
            probability_matrix, 
            top_words_voc, 
            molecular_relevance_weight=molecular_relevance_weight # TODO: make this a parameter... very important one too!
        )

        fig, bond_words = typewriter_molecule_art(
            mol, 
            final_bond_word_map, 
            family=selected_font,
            letter_spacing=letter_spacing,
            target_length=max_char_cutoff,
            spacing_char='',
            font_size=font_size,
            vertical_jitter=letter_vertical_jitter,
            letter_spacing_jitter=letter_horizontal_jitter,
            dashed_line_multiplier_for_bonds=dashed_bond_length,
            aromatic_circle_offset= aromatic_circle_offset,
            show_hydrogens=show_hydrogens,
            show_atoms=show_atoms,
            atom_size=atom_size,
            radius=2,
            reference_size = 15,
            auto_scale = autoscaling
        )

        # 2. Display the Matplotlib Figure using st.pyplot()
        st.pyplot(fig) # This is the command that displays the image!

        # # 3. Optionally display the words that were mapped
        # st.subheader("Mapped Words")

        # lookup = {word: ranking for word, index, ranking in words_to_embed}
        # setup = [(lookup[word], bond, word )for bond, word in bond_words.items()]
        # setup.sort(key=lambda x: x[0])

        # # bond_words is a dictionary, display it cleanly
        # word_df = pd.DataFrame(setup, columns=['Ranking', 'Bond Index', 'Word'])
        # st.dataframe(word_df, hide_index=True)