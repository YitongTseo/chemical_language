from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np

def typewriter_molecule_art(mol, bond_word_map, 
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

        if bond_idx in bond_word_map:
            word = bond_word_map[bond_idx]
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
    return fig, bond_word_map


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import os

# --- Helper Functions for Text Formatting and Jittered Drawing ---

def format_word_for_bond(word, target_len):
    """Format word to target length, centralizing it (Simplified for animation)."""
    target_len = min(target_len, 20)
    
    if len(word) > target_len:
        return word[:target_len]
    
    padding_needed = target_len - len(word)
    left_pad = padding_needed // 2
    
    # Use spaces for padding
    return (' ' * left_pad) + word
    

def draw_jittered_text(ax, text, x, y, angle, fontsize, jitter_y, jitter_x, family, color='black'):
    """Draw text with subtle jitter by applying a small, random, per-frame offset."""
    
    # Calculate position with small, random, per-frame offset for "jiggle"
    jiggle_x = np.random.uniform(-jitter_x, jitter_x) * 0.5 
    jiggle_y = np.random.uniform(-jitter_y, jitter_y) * 0.5
    
    char_x = x + jiggle_x
    char_y = y + jiggle_y
    
    # Draw the whole text object as one string
    return ax.text(char_x, char_y, text,
                   rotation=angle,
                   rotation_mode='anchor',
                   ha='center', va='center',
                   fontsize=fontsize,
                   weight='normal',
                   family=family,
                   color=color,
                   zorder=3)


# --- Main Animation Function ---

def animate_molecular_poems(mols_data, sim_params, draw_params):
    """
    Creates an animation where multiple molecular poems float, jiggle, and interact.
    mols_data is a list of tuples: (mol, bond_word_map, smiles_id)
    """
    frame_count = sim_params['frame_count']
    interval_ms = sim_params['interval_ms']
    sim_size = sim_params['sim_size']
    
    # 1. INITIALIZE MOLECULE STATES
    molecular_states = []
    
    # Determine the global scaling factor based on all molecules' sizes
    all_sizes = []
    for mol, _, _ in mols_data:
        conf = mol.GetConformer()
        xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
        ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
        mol_size = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        all_sizes.append(mol_size)
        
    reference_size = draw_params.get('reference_size', 15)
    global_scale_factor = max(all_sizes) / reference_size if draw_params['auto_scale'] and all_sizes else 1.0
    
    base_font_size = draw_params['font_size'] / global_scale_factor
    
    for mol, bond_word_map, _ in mols_data:
        conf = mol.GetConformer()
        xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
        ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
        
        mol_width = max(xs) - min(xs)
        mol_height = max(ys) - min(ys)
        
        state = {
            'mol': mol,
            'map': bond_word_map,
            # Effective radius for collision, normalized by scale factor
            'radius': max(mol_width, mol_height) / 2.0 / global_scale_factor * 0.5, 
            'pos': np.random.uniform(sim_size * 0.2, sim_size * 0.8, size=2), # Initial position
            'vel': np.random.uniform(-0.05, 0.05, size=2) * 0.5, # Initial slow velocity
            # RDKit's normalized center, used as internal offset
            'center_offset': np.array([np.mean(xs), np.mean(ys)]) / global_scale_factor 
        }
        molecular_states.append(state)

    # 2. SETUP PLOT
    # We use a square plot area for simple 2D physics
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white') 
    ax.set_xlim(0, sim_size)
    ax.set_ylim(0, sim_size)
    ax.axis('off')
    ax.set_facecolor('white')

    # 3. UPDATE FUNCTION (Executed per frame)
    def update_frame(frame):
        ax.clear()
        ax.set_xlim(0, sim_size)
        ax.set_ylim(0, sim_size)
        ax.axis('off')
        
        # Physics Step
        for i in range(len(molecular_states)):
            state_i = molecular_states[i]
            
            # A. Move and Brownian Jiggle
            state_i['pos'] += state_i['vel'] * 0.5
            state_i['vel'] += np.random.uniform(-0.005, 0.005, size=2) # Brownian force
            
            # B. Wall Collision (Elastic)
            if state_i['pos'][0] < state_i['radius'] or state_i['pos'][0] > sim_size - state_i['radius']:
                state_i['vel'][0] *= -1
            if state_i['pos'][1] < state_i['radius'] or state_i['pos'][1] > sim_size - state_i['radius']:
                state_i['vel'][1] *= -1

            # C. Molecule Collision (Simple Repulsion)
            for j in range(i + 1, len(molecular_states)):
                state_j = molecular_states[j]
                
                delta = state_j['pos'] - state_i['pos']
                distance = np.linalg.norm(delta)
                min_distance = state_i['radius'] + state_j['radius']
                
                if distance < min_distance and distance > 1e-6:
                    # Simple push-apart force to resolve overlap
                    overlap = min_distance - distance
                    push = delta / distance * (overlap * 0.01) 
                    
                    state_i['pos'] -= push
                    state_j['pos'] += push
                    
                    # Reflect velocities (basic elastic collision)
                    state_i['vel'] = state_i['vel'] - 2 * np.dot(state_i['vel'], delta) / (distance**2) * delta
                    state_j['vel'] = state_j['vel'] - 2 * np.dot(state_j['vel'], -delta) / (distance**2) * (-delta)

        # Drawing Step (Render all molecules)
        for state in molecular_states:
            mol = state['mol']
            bond_word_map = state['map']
            conf = mol.GetConformer()
            
            # Calculate the global translation vector
            global_offset = state['pos'] + state['center_offset'] 
            
            # --- Draw Words (Bonds) ---
            for bond in mol.GetBonds():
                bond_idx = bond.GetIdx()
                if bond_idx in bond_word_map:
                    word = bond_word_map[bond_idx]
                    formatted_word = format_word_for_bond(word, draw_params['target_length'])
                    
                    # Get RDKit coordinates
                    p1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
                    p2 = conf.GetAtomPosition(bond.GetEndAtomIdx())

                    # Apply scaling and translation
                    p1_scaled = np.array([p1.x, p1.y]) / global_scale_factor
                    p2_scaled = np.array([p2.x, p2.y]) / global_scale_factor
                    
                    # Translate to current simulation position
                    p1_sim = p1_scaled + global_offset - state['center_offset']
                    p2_sim = p2_scaled + global_offset - state['center_offset']
                    
                    mid_x, mid_y = (p1_sim + p2_sim) / 2
                    
                    angle = np.degrees(np.arctan2(p2_sim[1] - p1_sim[1], p2_sim[0] - p1_sim[0]))
                    if angle > 90 or angle < -90:
                        angle += 180

                    # Draw text with jitter
                    draw_jittered_text(ax, formatted_word, mid_x, mid_y, angle,
                                       base_font_size * 1.4, # Slightly larger for dashed bonds
                                       draw_params['vertical_jitter'], draw_params['letter_spacing_jitter'], 
                                       draw_params['family'])

        return ax.artists # Required for FuncAnimation

    # 4. CREATE ANIMATION
    anim = FuncAnimation(
        fig, 
        update_frame, 
        frames=frame_count, 
        interval=interval_ms, 
        blit=False, 
        repeat=True
    )
    
    # 5. Save and Return Path to Streamlit
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmpfile:
        anim_path = tmpfile.name
    
    # Save as GIF for Streamlit compatibility
    anim.save(anim_path, writer='pillow', fps=1000 // interval_ms)
    plt.close(fig) # Close the matplotlib figure
    
    return anim_path
