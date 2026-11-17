"""Reproduce the C5aR1 potency model from the cleaned ChEMBL data and save it for the app.

Data: IC50 records for human C5aR1 (ChEMBL target CHEMBL2373), averaged per molecule
as pChEMBL values (758 compounds). Features: 1024-bit Morgan fingerprints, radius 2.
"""
import json, random
from collections import defaultdict
import numpy as np, pandas as pd, joblib
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)

df = pd.read_csv("data/c5ar1_ic50_chembl2373.csv")
df["fp"] = df["canonical_smiles"].apply(fp)
df = df.dropna(subset=["fp"]).reset_index(drop=True)
X, y = np.stack(df["fp"].values), df["pchembl_value"].values
rf = lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))

# Random 80/20 split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
p = rf().fit(Xtr, ytr).predict(Xte)
m = {"random_r2": r2_score(yte, p), "random_rmse": rmse(yte, p)}

# Scaffold split: whole Murcko scaffolds go to either train or test
scaf = defaultdict(list)
for i, s in enumerate(df["canonical_smiles"]):
    scaf[Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))].append(i)
keys = list(scaf); random.seed(42); random.shuffle(keys)
n = int(len(keys) * 0.2)
te = [i for k in keys[:n] for i in scaf[k]]; tr = [i for k in keys[n:] for i in scaf[k]]
p = rf().fit(X[tr], y[tr]).predict(X[te])
d = DummyRegressor().fit(X[tr], y[tr]).predict(X[te])
m.update(scaffold_r2=r2_score(y[te], p), scaffold_rmse=rmse(y[te], p),
         baseline_r2=r2_score(y[te], d), n_compounds=len(df), n_scaffolds=len(keys))

# Final model for the app: trained on all compounds
joblib.dump(rf().fit(X, y), "models/rf_model.joblib", compress=3)
np.save("models/reference_fps.npy", np.packbits(X, axis=1))
json.dump({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()},
          open("models/metrics.json", "w"), indent=2)
print(json.dumps(m, indent=2))
