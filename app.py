from pathlib import Path

ROOT = Path(__file__).parent

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs

st.set_page_config(page_title="C5aR1 Potency Predictor", layout="wide")
st.title("🧪 C5aR1 Antagonist Potency Predictor")
st.write("Paste a molecule's SMILES string to predict its potency against the C5a receptor (C5aR1).")

# Load saved model and reference data
model = joblib.load(ROOT / 'models' / 'rf_model.joblib')
ref_df = pd.read_csv(ROOT / 'data' / 'c5ar1_ic50_chembl2373.csv')
X_ref = np.unpackbits(np.load(ROOT / 'models' / 'reference_fps.npy'), axis=1)

def smiles_to_fp(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return mol, np.array(fp)

smiles_input = st.text_input("Enter SMILES string:", value="CC(=O)Nc1ccc(O)cc1")

if st.button("Predict"):
    mol, fp = smiles_to_fp(smiles_input)

    if mol is None:
        st.error("Invalid SMILES string — couldn't parse this molecule.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Molecule")
            img = Draw.MolToImage(mol, size=(300,300))
            st.image(img)

        with col2:
            st.subheader("Prediction")
            pred = model.predict(fp.reshape(1,-1))[0]
            st.metric("Predicted pChEMBL value", f"{pred:.2f}")
            st.caption("Higher = more potent. Typical range in our dataset: ~4-9")

        st.divider()
        st.subheader("🔍 Most Similar Known Molecules")

        # Compute Tanimoto similarity to every reference molecule
        similarities = []
        for i, ref_fp in enumerate(X_ref):
            sim = DataStructs.TanimotoSimilarity(
                DataStructs.CreateFromBitString(''.join(map(str, fp))),
                DataStructs.CreateFromBitString(''.join(map(str, ref_fp)))
            )
            similarities.append(sim)

        ref_df_copy = ref_df.copy()
        ref_df_copy['similarity'] = similarities
        top_similar = ref_df_copy.sort_values('similarity', ascending=False).head(5)

        for _, row in top_similar.iterrows():
            c1, c2, c3 = st.columns([1,2,1])
            with c1:
                sim_mol = Chem.MolFromSmiles(row['canonical_smiles'])
                if sim_mol:
                    st.image(Draw.MolToImage(sim_mol, size=(150,150)))
            with c2:
                st.write(f"**{row['molecule_chembl_id']}**")
                st.write(f"Similarity: {row['similarity']:.2f}")
            with c3:
                st.metric("Actual pChEMBL", f"{row['pchembl_value']:.2f}")