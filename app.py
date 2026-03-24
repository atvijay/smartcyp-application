import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from stmol import showmol
import py3Dmol
from streamlit_ketcher import st_ketcher

st.set_page_config(page_title="SMARTCyp Pro", layout="wide")

# --- 1. THE ENGINE (Defined First) ---
RULES = {
    "Aliphatic": {"smarts": "[CX4H3,CX4H2,CX4H1]", "energy": 52.0},
    "Aromatic": {"smarts": "c[H]", "energy": 62.0},
    "N-Dealkyl": {"smarts": "[NX3][CX4H3,CX4H2]", "energy": 45.0},
    "O-Dealkyl": {"smarts": "[OX2][CX4H3,CX4H2]", "energy": 51.0}
}

def analyze_isoform(mol, isoform_type):
    results = []
    anchors_2d6 = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H1,H2,H0;!$(NC=O)]"))
    anchors_2c9 = mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H,OX1-]-[CX3]=[OX1]"))
    
    for name, data in RULES.items():
        pattern = Chem.MolFromSmarts(data['smarts'])
        for match in mol.GetSubstructMatches(pattern):
            idx = match[0]
            score = data['energy']
            
            if isoform_type == "CYP2D6" and anchors_2d6:
                distances = [0 if idx == a[0] else len(Chem.GetShortestPath(mol, idx, a[0])) for a in anchors_2d6]
                score += (min(distances) - 5)**2
            elif isoform_type == "CYP2C9" and anchors_2c9:
                distances = [0 if idx == a[0] else len(Chem.GetShortestPath(mol, idx, a[0])) for a in anchors_2c9]
                score += (min(distances) - 4)**2
                
            results.append({"Atom": idx + 1, "Mechanism": name, "Score": round(score, 1)})
    return pd.DataFrame(results).sort_values("Score") if results else pd.DataFrame()

# --- 2. THE UI (Sidebar & Ketcher) ---
st.title("🧪 SMARTCyp Pro: Sketcher & 3D Predictor")

with st.sidebar:
    st.header("Settings")
    selected_isoform = st.selectbox("Current View:", ["CYP3A4", "CYP2D6", "CYP2C9"])
    st.info(f"Analyzing for **{selected_isoform}** logic.")
    st.sidebar.markdown("---")
    st.sidebar.info("**SMARTCyp Pro v1.0**\nBuilt using RDKit logic.")

# Molecular Input Section
st.subheader("Structure Input")
col_kbd, col_sketch = st.columns([1, 2])

with col_kbd:
    # We use session_state to sync the text box and the sketcher
    input_smiles = st.text_input("Enter/Edit SMILES:", "CNC1=CC=C(C=C1)C2=CC=CC=C2")

with col_sketch:
    st.write("Modify structure visually:")
    # Ketcher takes the input_smiles and returns updates
    smiles = st_ketcher(formula=input_smiles)

# --- 3. THE ANALYSIS LOGIC ---
if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        tab1, tab2, tab3 = st.tabs(["📊 Single Isoform Focus", "🏁 Comparison Grid", "🧊 3D View"])
        
        with tab1:
            df_selected = analyze_isoform(mol, selected_isoform)
            if not df_selected.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(f"{selected_isoform} Highlight")
                    top_idx = [int(df_selected.iloc[0]['Atom'] - 1)]
                    st.image(Draw.MolToImage(mol, size=(400, 400), highlightAtoms=top_idx))
                with c2:
                    st.subheader("Vulnerability Table")
                    st.dataframe(df_selected, hide_index=True, use_container_width=True)

        with tab2:
            cols = st.columns(3)
            for i, iso in enumerate(["CYP3A4", "CYP2D6", "CYP2C9"]):
                with cols[i]:
                    st.markdown(f"**{iso}**")
                    df_iso = analyze_isoform(mol, iso)
                    if not df_iso.empty:
                        idx = [int(df_iso.iloc[0]['Atom'] - 1)]
                        st.image(Draw.MolToImage(mol, size=(300, 300), highlightAtoms=idx))
                        st.dataframe(df_iso.head(3), hide_index=True)

        with tab3:
            st.subheader("Interactive 3D Structure")
            m3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
            mblock = Chem.MolToMolBlock(m3d)
            
            view = py3Dmol.view(width=800, height=450)
            view.addModel(mblock, 'mol')
            view.setStyle({'stick': {'colorscheme': 'greenCarbon'}})
            view.zoomTo()
            showmol(view, height=450, width=800)
    else:
        st.error("Invalid SMILES format. Please check your structure.")

st.sidebar.info("""
**SMARTCyp Pro v1.0**
Predicts metabolic sites for CYP3A4, 2D6, and 2C9.
Built using RDKit and SMARTCyp 3.0 logic.
""")