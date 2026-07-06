<p align="center">
  <img src="https://raw.githubusercontent.com/prism-oncology/customics/main/docs/assets/customics.png" alt="customics_logo" width="300"/>
</p>
<p align="center"><b><i>
	CustOmics: a versatile deep-learning based strategy for multi-omics integration
</b></i></p>

<div align="center">

[![PyPI version](https://badge.fury.io/py/customics.svg)](https://badge.fury.io/py/customics)
[![Downloads](https://static.pepy.tech/badge/customics)](https://pepy.tech/project/customics)
[![Docs](https://img.shields.io/badge/docs-Zensical-526CFE?logo=materialformkdocs&logoColor=white)](https://prism-oncology.github.io/customics/)
[![Build](https://github.com/prism-oncology/customics/actions/workflows/ci.yml/badge.svg)](https://github.com/prism-oncology/customics/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

**Integrate RNA-seq, CNV, DNA methylation, and more into a single predictive model.**

`customics` is a Python package for integrating multiple genomic data modalities using a hierarchical deep-learning architecture. It supports classification, survival outcome prediction, and SHAP-based explainability — all in a single scikit-learn-style API, built on [MuData](https://mudata.readthedocs.io/stable/) from [scverse](https://scverse.org/).

## Documentation

Check [customics's documentation](https://prism-oncology.github.io/customics/) to get started. It contains installation explanations, API details, and tutorials.

## Installation

`customics` can be installed from `PyPI` on all OS, for any Python version `>=3.11`:

```bash
pip install customics
```

## Features

- **Multi-omics integration** — a hierarchical architecture (per-source autoencoders feeding a central VAE) that fuses heterogeneous, high-dimensional modalities into a shared latent space.
- **Built-in tasks** — tumor classification and survival prediction (Cox), with one scikit-learn-style `fit` / `predict` / `evaluate` API.
- **Explainability** — per-source feature attribution via SHAP.
- **Visualization** — latent-space projection (t-SNE) and Kaplan-Meier survival stratification out of the box.


## Usage


```python
import customics
from customics import CustOMICS

# --- 1. Load and prepare data ---
# toy_dataset() returns a MuData object: one modality per omics source,
# with clinical annotations in `.obs`.
mdata = customics.toy_dataset()
customics.prepare_input(
    mdata,
    label="PAM50",       # classification target column
    event="OS",          # survival event column (0/1)
    surv_time="OS.time", # survival time column
)

# --- 2. Instantiate the model ---
# Each config dict tunes one component. See the configuration reference for the
# full schema: https://prism-oncology.github.io/customics/hyperparameter_tuning/
model = CustOMICS(
    source_params=source_params,
    central_params=central_params,
    classif_params=classif_params,
    surv_params=surv_params,
    train_params=train_params
)

# --- 3. Train ---
model.fit(mdata, batch_size=32, n_epochs=30, verbose=True)

# --- 4. Evaluate ---
metrics = model.evaluate(mdata, task="classification")  # Accuracy, F1, AUC, …
ci = model.evaluate(mdata, task="survival")             # concordance index

# --- 5. Visualise & explain ---
model.plot_loss()
model.plot_representation(mdata, color="PAM50")
model.stratify(mdata, show=True)
model.explain(sample_ids, mdata, source="rna", subtype="Her2")
```

> [!NOTE]
> See this [usage section](https://prism-oncology.github.io/customics/tutorials/usage/) for more details about usage.

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
