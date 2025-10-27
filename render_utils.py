from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np

def typewriter_molecule_art(smiles, words_to_embed, 
                               family='American Typewriter',
                               letter_spacing=0.0085,
                               target_length=20,
                               spacing_char='',
                               font_size=21,
                               vertical_jitter=0.03,
                               letter_spacing_jitter=0.03,
                               dashed_line_multiplier_for_bonds=4.5,
                               aromatic_circle_offset= 0.3,
                               show_hydrogens=False,
                               show_atoms=False,
                               atom_size=2,
                               radius=2,
                               reference_size = 15,
                               auto_scale = True):
    """
    Cleaner typewriter-style molecule with better spacing.
    Now includes aromatic ring "typed" circles.
    """

    # Prepare molecule
    mol = Chem.MolFromSmiles(smiles)
    if not show_hydrogens:
        mol = Chem.RemoveHs(mol)
    else:
        mol = Chem.AddHs(mol)

    AllChem.Compute2DCoords(mol)

    fig, ax = plt.subplots(figsize=(20, 20), facecolor='white')
    conf = mol.GetConformer()


    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    mol_size = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale_factor = mol_size / reference_size if auto_scale else 1.0
    font_size = font_size / scale_factor
    print('scale_factor ', scale_factor , ' font_size ', font_size)

    # Get fingerprint info - ONLY radius 2
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=1024, bitInfo=info)

    # Map bits to words
    bit_to_word = {bit_idx: word for word, bit_idx, ranking in words_to_embed}

    # Assign words to bonds
    bond_words = {}
    for bit_idx, word in bit_to_word.items():
        if bit_idx in info:
            atom_idx, rad = info[bit_idx][0]
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
            for bond_idx in env:
                if bond_idx not in bond_words:
                    bond_words[bond_idx] = word
                    break

    print(f"Mapped {len(bond_words)} words to {mol.GetNumBonds()} bonds")
    print(f"Bonds with words: {sorted(bond_words.keys())}")
    print(f"Bonds without words: {[b.GetIdx() for b in mol.GetBonds() if b.GetIdx() not in bond_words]}")

    # --- Helper Functions ---
    def format_word_for_bond(word, target_len, spacing_char):
        """Format word to target length with tighter spacing"""
        if spacing_char == '-' or spacing_char == '.':
            spaced = spacing_char.join(list(word))
            if len(spaced) < target_len:
                padding_needed = target_len - len(spaced)
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                spaced = spacing_char * left_pad + spaced + spacing_char * right_pad
            return spaced[:target_len]
        elif spacing_char is None:
            repeated = (word * (target_len // len(word) + 1))[:target_len]
            return repeated
        else:
            spaced = spacing_char.join(list(word))
            if len(spaced) < target_len:
                padding_needed = target_len - len(spaced)
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                spaced = spacing_char * left_pad + spaced + spacing_char * right_pad
            return spaced[:target_len]

    def draw_jittered_text(ax, text, x, y, angle, fontsize, jitter_y, jitter_x):
        """Draw text with subtle jitter - TIGHTER SPACING"""
        angle_rad = np.radians(angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        perp_dx = -np.sin(angle_rad)
        perp_dy = np.cos(angle_rad)
        char_width = fontsize * letter_spacing  # Reduced spacing
        text_len = len(text)
        start_offset = -(text_len * char_width) / 2

        for i, char in enumerate(text):
            h_offset = start_offset + i * char_width + np.random.uniform(-jitter_x, jitter_x)
            v_offset = np.random.uniform(-jitter_y, jitter_y)
            char_x = x + h_offset * dx + v_offset * perp_dx
            char_y = y + h_offset * dy + v_offset * perp_dy
            char_angle = angle + np.random.uniform(-1, 1)
            ax.text(char_x, char_y, char,
                    rotation=char_angle,
                    rotation_mode='anchor',
                    ha='center', va='center',
                    fontsize=fontsize,
                    weight='normal',
                    family=family,
                    color='black')

    def draw_typewriter_circle(ax, center_x, center_y, radius, text='-', font_size=11,
                           vertical_jitter=0.02, letter_spacing_jitter=0.01, num_chars=100):
        """Draws a circular dashed line using typewriter aesthetic, 
        with proper tangent alignment."""
        char_angle = 2 * np.pi / num_chars
        for i in range(num_chars):
            # Angle at the midpoint of the dash placement
            theta = char_angle * (i + 0.5)
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)
            
            # Tangent angle — perpendicular to the radius
            angle = np.degrees(theta + np.pi / 2)

            # Draw each dash or word following the circular path
            draw_jittered_text(ax, text, x, y, angle,
                            font_size, vertical_jitter, letter_spacing_jitter)


    # --- Draw Bonds ---
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        p1 = conf.GetAtomPosition(begin_idx)
        p2 = conf.GetAtomPosition(end_idx)
        bond_type = bond.GetBondType()
        is_double = (bond_type == Chem.BondType.DOUBLE)
        is_triple = (bond_type == Chem.BondType.TRIPLE)

        # Determine if bond is aromatic
        is_aromatic = bond.GetIsAromatic()

        if bond_idx in bond_words:
            word = bond_words[bond_idx]
            formatted_word = format_word_for_bond(word, target_length, spacing_char)
            angle = np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))
            if angle > 90 or angle < -90:
                angle += 180
            mid_x, mid_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
            if is_double:
                angle_rad = np.radians(angle)
                perp_dx = -np.sin(angle_rad) * 0.15
                perp_dy = np.cos(angle_rad) * 0.15
                draw_jittered_text(ax, formatted_word, mid_x + perp_dx, mid_y + perp_dy, angle,
                                   font_size, vertical_jitter, letter_spacing_jitter)
                draw_jittered_text(ax, formatted_word, mid_x - perp_dx, mid_y - perp_dy, angle,
                                   font_size, vertical_jitter, letter_spacing_jitter)
            elif is_triple:
                angle_rad = np.radians(angle)
                perp_dx = -np.sin(angle_rad) * 0.15
                perp_dy = np.cos(angle_rad) * 0.15
                draw_jittered_text(ax, formatted_word, mid_x + perp_dx, mid_y + perp_dy, angle,
                                   font_size, vertical_jitter, letter_spacing_jitter)
                draw_jittered_text(ax, formatted_word, mid_x, mid_y, angle,
                                   font_size, vertical_jitter, letter_spacing_jitter)
                draw_jittered_text(ax, formatted_word, mid_x - perp_dx, mid_y - perp_dy, angle,
                                   font_size, vertical_jitter, letter_spacing_jitter)
            else:
                draw_jittered_text(ax, formatted_word, mid_x, mid_y, angle,
                                   font_size, vertical_jitter, letter_spacing_jitter)
        else:
            # Draw as dashed line (typewriter style)
            bond_length = np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            num_dashes = int(bond_length * dashed_line_multiplier_for_bonds)
            dash_text = '-' * num_dashes
            angle = np.degrees(np.arctan2(p2.y - p1.y, p2.x - p1.x))
            if angle > 90 or angle < -90:
                angle += 180
            mid_x, mid_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
            if is_double or is_triple:
                angle_rad = np.radians(angle)
                perp_dx = -np.sin(angle_rad) * 0.4
                perp_dy = np.cos(angle_rad) * 0.4
                offsets = [-0.15, 0, 0.15] if is_triple else [-0.15, 0.15]
                for off in offsets:
                    draw_jittered_text(ax, dash_text, mid_x + off * perp_dx, mid_y + off * perp_dy, angle,
                                       font_size * 1.4, vertical_jitter, letter_spacing_jitter)
            else:
                draw_jittered_text(ax, dash_text , mid_x, mid_y, angle,
                                   font_size * 1.4, vertical_jitter, letter_spacing_jitter)

    # --- Draw aromatic ring circles ---
    ring_info = mol.GetRingInfo()
    aromatic_rings = [ring for ring in ring_info.AtomRings()
                      if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]

    for ring in aromatic_rings:
        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in ring])
        centroid = coords.mean(axis=0)
        radius_ring = np.mean(np.sqrt(np.sum((coords - centroid)**2, axis=1)))
        inner_radius = radius_ring * (1 - aromatic_circle_offset)

        draw_typewriter_circle(ax, centroid[0], centroid[1], inner_radius,
                               text='-', font_size=font_size * 1.4,
                               vertical_jitter=vertical_jitter,
                               letter_spacing_jitter=letter_spacing_jitter,
                               num_chars=30)

    # --- Draw atoms ---
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if not show_hydrogens and atom.GetSymbol() == 'H':
            continue
        pos = conf.GetAtomPosition(i)
        if not(show_atoms):
            ax.plot(pos.x, pos.y, 'o',
                    markersize=0,
                    color='black',
                    markerfacecolor='white',
                    markeredgewidth=2.5,
                    zorder=2)
        else:
            ax.plot(pos.x, pos.y, 'o',
                    markersize=atom_size,
                    color='black',
                    markerfacecolor='white',
                    markeredgewidth=2.5,
                    zorder=2)

    
    ax.axis('equal')
    ax.axis('off')
    ax.margins(0.12)
    plt.tight_layout()
    return fig, bond_words
