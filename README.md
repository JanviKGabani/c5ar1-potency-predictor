# C5aR1 Potency Predictor

A QSAR regression model that estimates pChEMBL potency against the human C5a receptor (C5aR1) from a molecule's structure, with a nearest-neighbour lookup into the known ChEMBL compounds.

**Live demo:** coming soon · **Portfolio:** [janvikgabani.github.io](https://janvikgabani.github.io)

## Data

Public IC50 records for ChEMBL target `CHEMBL2373` (human C5aR1), averaged per molecule as pChEMBL values: **758 compounds** across **251 Murcko scaffolds** (`data/c5ar1_ic50_chembl2373.csv`).

## Method

- **Features:** 1024-bit Morgan fingerprints, radius 2 (RDKit).
- **Model:** Random Forest regressor, 200 trees.
- **Evaluation:** a random 80/20 split, and a stricter scaffold split where whole Murcko scaffolds go to either train or test, so the model is tested on chemistry it has not seen.

## Results

| Split | R² | RMSE |
|:---|---:|---:|
| Random 80/20 | 0.627 | 0.584 |
| Scaffold split | 0.458 | 0.709 |
| Mean baseline (scaffold split) | −0.019 | 0.973 |

The gap between the two splits is the point: a random split flatters the model because near-duplicate analogues land on both sides. The scaffold split is the honest number, and it still beats a mean-predicting baseline by a wide margin.

## Run locally

```bash
pip install -r requirements.txt
python train.py        # rebuilds models/ from the CSV
streamlit run app.py
```

## Files

| Path | What it is |
|---|---|
| `app.py` | Streamlit demo: enter a structure, see the predicted value and the most similar known compounds |
| `train.py` | Rebuilds the model and both evaluation splits from the CSV |
| `data/c5ar1_ic50_chembl2373.csv` | Cleaned ChEMBL activity data (758 compounds) |
| `models/rf_model.joblib` | Random Forest trained on all 758 compounds |
| `models/reference_fps.npy` | Packed fingerprints of the reference set, for similarity search |
| `models/metrics.json` | The numbers in the table above |

Research and coursework only. Predictions from a model this size are a starting point for triage, not evidence about any real compound.
