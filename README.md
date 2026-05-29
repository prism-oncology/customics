<p align="center">
  <img src="https://raw.githubusercontent.com/prism-oncology/customics/main/docs/assets/customics.png" alt="customics_logo" width="300"/>
</p>
<p align="center"><b><i>
	CustOmics: a versatile deep-learning based strategy for multi-omics integration
</b></i></p>

<div align="center">

[![PyPI version](https://badge.fury.io/py/customics.svg)](https://badge.fury.io/py/customics)
[![Build](https://github.com/prism-oncology/customics/actions/workflows/ci.yml/badge.svg)](https://github.com/prism-oncology/customics/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

`customics` is a Python package for integrating multiple genomic data modalities (RNA-seq, CNV, DNA methylation, …) using a hierarchical deep-learning architecture. It supports classification, survival outcome prediction, and SHAP-based explainability — all in a single scikit-learn-style API.

## Installation

`customics` can be installed via `pip`:

```bash
pip install customics
```

Or install from source:

```bash
git clone https://github.com/prism-oncology/customics.git
cd customics
pip install -e .
```

## Usage demo

```python
import torch
import pandas as pd
from customics import CustOMICS

# --- 1. Prepare your data ---
# omics_train: dict mapping source name → pd.DataFrame (samples × features)
# clinical_df: pd.DataFrame with columns for labels, event indicator, and survival time
omics_train = {
    "rna":   pd.read_csv("rna_train.csv",   index_col=0),
    "cnv":   pd.read_csv("cnv_train.csv",   index_col=0),
    "methyl": pd.read_csv("methyl_train.csv", index_col=0),
}
clinical_df = pd.read_csv("clinical.csv", index_col=0)

# --- 2. Configure the model ---
source_params = {
    "rna":   {"input_dim": 5000, "hidden_dim": [1024, 512], "latent_dim": 128, "norm": True, "dropout": 0.2},
    "cnv":   {"input_dim": 2000, "hidden_dim": [512, 256],  "latent_dim": 128, "norm": True, "dropout": 0.2},
    "methyl":{"input_dim": 8000, "hidden_dim": [1024, 512], "latent_dim": 128, "norm": True, "dropout": 0.2},
}
central_params = {"hidden_dim": [512, 256], "latent_dim": 128, "norm": True, "dropout": 0.2, "beta": 1}
classif_params = {"n_class": 5, "lambda": 5.0, "hidden_layers": [128, 64], "dropout": 0.2}
surv_params    = {"lambda": 1.0, "dims": [64, 32], "activation": "SELU",
                  "l2_reg": 1e-2, "norm": True, "dropout": 0.2}
train_params   = {"switch": 10, "lr": 1e-3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. Train ---
model = CustOMICS(
    source_params=source_params,
    central_params=central_params,
    classif_params=classif_params,
    surv_params=surv_params,
    train_params=train_params,
    device=device,
)
model.fit(
    omics_train=omics_train,
    clinical_df=clinical_df,
    label="PAM50",       # classification target column
    event="OS",          # survival event column (0/1)
    surv_time="OS.time", # survival time column
    omics_val=omics_val, # optional validation set
    batch_size=32,
    n_epochs=30,
    verbose=True,
)

# --- 4. Evaluate ---
# Classification metrics (Accuracy, F1, AUC, …)
metrics = model.evaluate(
    omics_test, clinical_df,
    label="PAM50", event="OS", surv_time="OS.time",
    task="classification",
)
# Survival concordance index
ci = model.evaluate(
    omics_test, clinical_df,
    label="PAM50", event="OS", surv_time="OS.time",
    task="survival",
)

# --- 5. Visualise & explain ---
model.plot_loss()
model.plot_representation(omics_train, clinical_df, label="PAM50",
                          filename="latent_space", title="t-SNE of latent space")
model.stratify(omics_train, clinical_df, event="OS", surv_time="OS.time")
model.explain(sample_ids, omics_train, clinical_df,
              source="rna", subtype="Her2", label="PAM50")
```

## Reproducing Paper Results

Download TCGA data from the [GDC Data Portal](https://portal.gdc.cancer.gov/) or [cBioPortal](https://www.cbioportal.org/). Pre-computed 5-fold CV splits for BRCA, LUAD, UCEC, BLCA, GBM, OV, and PANCAN are included in the `data/splits/` directory.

See our documentation for a complete end-to-end walkthrough using the bundled toy dataset.

## Citation

If you use `customics` in your research, please cite:

```bibtex
@article{benkirane2023,
    doi       = {10.1371/journal.pcbi.1010921},
    author    = {Benkirane, Hakim AND Pradat, Yoann AND Michiels, Stefan AND Cournède, Paul-Henry},
    journal   = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title     = {CustOmics: A versatile deep-learning based strategy for multi-omics integration},
    year      = {2023},
    month     = {03},
    volume    = {19},
    pages     = {1--19},
    number    = {3}
}
```
