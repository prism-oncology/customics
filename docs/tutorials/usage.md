# Main usage

This notebook walks through a complete **customics** workflow on the bundled toy dataset:

1. Load and explore multi-omics data
2. Split into train / validation / test sets
3. Configure the model architecture
4. Build and train a `CustOMICS` model
5. Evaluate classification and survival performance
6. Visualise the learned latent space
7. Stratify patients by predicted risk (Kaplan-Meier)
8. Explain model decisions with SHAP values
9. Save and reload a trained model

---

## Architecture at a glance

```
  protein  ──► AE_protein  ──┐
                             │
  rna      ──► AE_rna    ──►─┤  Central VAE  ──► Classifier (tumour subtype)
                             │   (latent z)
  methyl   ──► AE_methyl   ──┘               └──► Survival predictor (Cox)
```

**Phase 1** (epochs 0 → `switch`): each source autoencoder is trained
independently; task heads operate on per-source representations.
**Phase 2** (epochs `switch` → end): the central VAE integrates all sources
into a unified latent space used by the task heads.



```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import customics
from customics import CustOMICS
```

---

## 1. Loading and Exploring the Data

The toy dataset contains **100 patients** with three omics modalities:

| Source    | Features | Description                        |
| --------- | -------- | ---------------------------------- |
| `protein` | 160      | Reverse Phase Protein Array (RPPA) |
| `rna`     | 131      | RNA-seq gene expression            |
| `methyl`  | 367      | DNA methylation (450K array)       |



```python
mdata = customics.toy_dataset()
```

    /Users/alihamraoui/projects/tests/CustOmics/.venv/lib/python3.12/site-packages/mudata/_core/mudata.py:1416: UserWarning: var_names are not unique. To make them unique, call `.var_names_make_unique`.
      self._update_attr("var", axis=0, join_common=join_common)


`mdata` is a [`mudata.MuData`](https://mudata.readthedocs.io/stable/) object:

- `mdata.mod` — dict of modality names → `AnnData` matrices (rows = samples, columns = features)
- `mdata.obs` — shared clinical metadata DataFrame indexed by **sample ID**



```python
list(mdata.mod.keys())
```




    ['rna', 'protein', 'methyl']




```python
mdata["protein"].to_df().head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>probe</th>
      <th>ACC1</th>
      <th>ACC_pS79</th>
      <th>ACVRL1</th>
      <th>Akt_pS473</th>
      <th>PRAS40_pT246</th>
      <th>Annexin.1</th>
      <th>AR</th>
      <th>A.Raf_pS299</th>
      <th>ASNS</th>
      <th>ATM</th>
      <th>...</th>
      <th>XBP1</th>
      <th>XRCC1</th>
      <th>Ku80</th>
      <th>YAP</th>
      <th>YAP_pS127</th>
      <th>YB.1</th>
      <th>YB.1_pS102</th>
      <th>14.3.3_beta</th>
      <th>14.3.3_epsilon</th>
      <th>14.3.3_zeta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>subject1</th>
      <td>-0.528421</td>
      <td>-0.826949</td>
      <td>2.465380</td>
      <td>0.118108</td>
      <td>2.682673</td>
      <td>-0.473918</td>
      <td>2.692947</td>
      <td>0.392130</td>
      <td>-1.458304</td>
      <td>-0.518608</td>
      <td>...</td>
      <td>2.365702</td>
      <td>-0.115504</td>
      <td>2.125662</td>
      <td>0.198529</td>
      <td>0.263075</td>
      <td>2.035369</td>
      <td>3.276020</td>
      <td>-0.164232</td>
      <td>2.240264</td>
      <td>0.390613</td>
    </tr>
    <tr>
      <th>subject2</th>
      <td>-0.804377</td>
      <td>-0.858630</td>
      <td>2.679618</td>
      <td>1.163897</td>
      <td>2.789568</td>
      <td>0.326823</td>
      <td>3.703054</td>
      <td>-0.207030</td>
      <td>-0.567368</td>
      <td>-1.447359</td>
      <td>...</td>
      <td>2.369684</td>
      <td>-0.250648</td>
      <td>2.028639</td>
      <td>0.294427</td>
      <td>0.087710</td>
      <td>2.748533</td>
      <td>2.655492</td>
      <td>-0.020322</td>
      <td>2.584052</td>
      <td>1.083867</td>
    </tr>
    <tr>
      <th>subject3</th>
      <td>0.596001</td>
      <td>0.175652</td>
      <td>2.782368</td>
      <td>-1.550062</td>
      <td>2.136330</td>
      <td>-0.729363</td>
      <td>3.987616</td>
      <td>-0.065740</td>
      <td>0.225819</td>
      <td>0.321197</td>
      <td>...</td>
      <td>2.912468</td>
      <td>0.235292</td>
      <td>3.102378</td>
      <td>0.717648</td>
      <td>0.118142</td>
      <td>2.608048</td>
      <td>1.894032</td>
      <td>0.423970</td>
      <td>2.470960</td>
      <td>0.355451</td>
    </tr>
    <tr>
      <th>subject4</th>
      <td>2.306769</td>
      <td>2.387911</td>
      <td>2.152993</td>
      <td>0.175379</td>
      <td>-0.243862</td>
      <td>0.206090</td>
      <td>3.851968</td>
      <td>-0.347185</td>
      <td>1.595250</td>
      <td>2.998184</td>
      <td>...</td>
      <td>0.019668</td>
      <td>2.319597</td>
      <td>0.211079</td>
      <td>0.178415</td>
      <td>0.312633</td>
      <td>-0.205992</td>
      <td>0.151597</td>
      <td>2.289823</td>
      <td>0.153826</td>
      <td>-0.634842</td>
    </tr>
    <tr>
      <th>subject5</th>
      <td>-0.948945</td>
      <td>-0.640623</td>
      <td>2.242055</td>
      <td>0.829983</td>
      <td>2.732090</td>
      <td>-0.218177</td>
      <td>2.376110</td>
      <td>-0.510938</td>
      <td>-1.030711</td>
      <td>-0.125802</td>
      <td>...</td>
      <td>2.169958</td>
      <td>-0.044484</td>
      <td>2.905596</td>
      <td>0.539468</td>
      <td>0.338059</td>
      <td>2.257715</td>
      <td>2.718098</td>
      <td>-0.190475</td>
      <td>2.523407</td>
      <td>0.445861</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 160 columns</p>
</div>



`mdata.obs` is a DataFrame whose index are sample IDs, and contains the following columns:

- an event-indicator (`OS`, 0 = censored, 1 = event)
- a survival-time column (`OS.time`, in days).
- **tumour subtype label** (`cluster.id`, classes 1–5)



```python
mdata.obs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subjects</th>
      <th>cluster.id</th>
      <th>OS</th>
      <th>OS.time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>subject1</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>2531</td>
    </tr>
    <tr>
      <th>subject2</th>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>759</td>
    </tr>
    <tr>
      <th>subject3</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>2453</td>
    </tr>
    <tr>
      <th>subject4</th>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>220</td>
    </tr>
    <tr>
      <th>subject5</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>2431</td>
    </tr>
  </tbody>
</table>
</div>



We can show a quick summary of our omics and clinical data:



```python
print("Omics sources loaded:")
for name in mdata.mod:
    print(f"  {name:10s}  {mdata[name].shape[0]:>4d} samples * {mdata[name].shape[1]:>4d} features")
print(f"\nClinical data  : {mdata.obs.shape[0]} rows * {mdata.obs.shape[1]} columns")
```

    Omics sources loaded:
      rna          100 samples *  131 features
      protein      100 samples *  160 features
      methyl       100 samples *  367 features

    Clinical data  : 100 rows * 4 columns



```python
# Class distribution — the label we will train the classifier on.
print("\n=== Label distribution (cluster.id) ===")
counts = mdata.obs["cluster.id"].value_counts().sort_index()
print(counts.to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Samples per subtype")
axes[0].set_xlabel("Cluster ID (subtype)")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)

mdata.obs["OS.time"].hist(bins=20, ax=axes[1], color="coral", edgecolor="white")
axes[1].set_title("Overall survival time distribution")
axes[1].set_xlabel("Days")
plt.tight_layout()
plt.show()
```


    === Label distribution (cluster.id) ===
    cluster.id
    1    35
    2    13
    3    19
    4     8
    5    25




![png](usage_files/usage_11_1.png)




```python
# get_shared_samples returns the intersection of sample IDs across all modalities.
# This ensures every downstream step uses exactly the same set of sample IDs.
shared_samples = customics.get_shared_samples(mdata)
print(f"Samples present in ALL sources and clinical data: {len(shared_samples)}")
print(f"First 5 sample IDs: {shared_samples[:5]}")
```

    Samples present in ALL sources and clinical data: 100
    First 5 sample IDs: ['subject1', 'subject10', 'subject100', 'subject11', 'subject12']


---

## 2. Splitting the Data

We split the full cohort into **train / validation / test** sets.
`get_sub_mudata` applies the same sample selection to all modalities simultaneously,
keeping the `MuData` object in sync.



```python
samples_train, samples_test = train_test_split(shared_samples, test_size=0.20, random_state=42)
samples_train, samples_val = train_test_split(samples_train, test_size=0.15, random_state=42)

print(f"Train : {len(samples_train):>3d} samples")
print(f"Val   : {len(samples_val):>3d} samples")
print(f"Test  : {len(samples_test):>3d} samples")
```

    Train :  68 samples
    Val   :  12 samples
    Test  :  20 samples



```python
# prepare_input validates clinical columns and registers them in mdata.uns
# so fit() and evaluate() can find them without repeating column names each time.
label, event, surv_time = "cluster.id", "OS", "OS.time"
customics.prepare_input(mdata, label="cluster.id", event="OS", surv_time="OS.time")

# get_sub_mudata returns a new MuData filtered to the given sample IDs.
mdata_train = customics.get_sub_mudata(mdata, samples_train)
mdata_val = customics.get_sub_mudata(mdata, samples_val)
mdata_test = customics.get_sub_mudata(mdata, samples_test)

# Record input dimensions per source — used to configure the autoencoders below.
x_dim = {mod: mdata[mod].shape[1] for mod in mdata.mod}
print("Feature dimensions per source:", x_dim)
```

    Feature dimensions per source: {'rna': 131, 'protein': 160, 'methyl': 367}


    /Users/alihamraoui/projects/tests/CustOmics/.venv/lib/python3.12/site-packages/mudata/_core/mudata.py:1416: UserWarning: var_names are not unique. To make them unique, call `.var_names_make_unique`.
      self._update_attr("var", axis=0, join_common=join_common)
    /Users/alihamraoui/projects/tests/CustOmics/.venv/lib/python3.12/site-packages/mudata/_core/mudata.py:1416: UserWarning: var_names are not unique. To make them unique, call `.var_names_make_unique`.
      self._update_attr("var", axis=0, join_common=join_common)
    /Users/alihamraoui/projects/tests/CustOmics/.venv/lib/python3.12/site-packages/mudata/_core/mudata.py:1416: UserWarning: var_names are not unique. To make them unique, call `.var_names_make_unique`.
      self._update_attr("var", axis=0, join_common=join_common)


---

## 3. Configuring the Model

`CustOMICS` is configured through five parameter dictionaries.
Each controls one component of the architecture.

### `source_params` — per-source autoencoders

One entry per omics modality. The autoencoder compresses raw features into a
compact `latent_dim`-dimensional representation.

| Key          | Type      | Description                                                        |
| ------------ | --------- | ------------------------------------------------------------------ |
| `input_dim`  | int       | Number of input features (from the data)                           |
| `hidden_dim` | list[int] | Hidden layer sizes (encoder direction; decoder mirrors in reverse) |
| `latent_dim` | int       | Per-source embedding dimension                                     |
| `norm`       | bool      | Use BatchNorm after each FC layer                                  |
| `dropout`    | float     | Dropout probability in [0, 1]                                      |

### `central_params` — central VAE

Receives the concatenation of all per-source embeddings and produces the
shared latent code **z** used by both task heads.

| Key          | Type      | Description                                        |
| ------------ | --------- | -------------------------------------------------- |
| `hidden_dim` | list[int] | Hidden layer sizes                                 |
| `latent_dim` | int       | Dimension of **z** (the integrated representation) |
| `beta`       | float     | Weight of the MMD regularisation term              |
| `norm`       | bool      | BatchNorm in layers                                |
| `dropout`    | float     | Dropout probability                                |

### `classif_params` — classifier head

Maps **z** → tumour subtype probabilities via a small MLP.

| Key             | Type      | Description                                  |
| --------------- | --------- | -------------------------------------------- |
| `n_class`       | int       | Number of output classes                     |
| `lambda`        | float     | Classification loss weight in the total loss |
| `hidden_layers` | list[int] | MLP hidden sizes                             |
| `dropout`       | float     | Dropout probability                          |

### `surv_params` — Cox survival head

Maps **z** → log-hazard score. Set `lambda = 0` to disable.

| Key          | Type      | Description                                           |
| ------------ | --------- | ----------------------------------------------------- |
| `lambda`     | float     | Cox loss weight (`0` = disabled)                      |
| `dims`       | list[int] | Hidden layer sizes (final output size is always 1)    |
| `activation` | str       | Activation name: `"SELU"`, `"ReLU"`, `"LeakyReLU"`, … |
| `l2_reg`     | float     | L2 regularisation on the survival head weights        |
| `norm`       | bool      | BatchNorm in layers                                   |
| `dropout`    | float     | Dropout probability                                   |

### `train_params` — optimiser & phase schedule

| Key      | Type  | Description                                 |
| -------- | ----- | ------------------------------------------- |
| `switch` | int   | Epoch where phase 1 ends and phase 2 begins |
| `lr`     | float | Adam learning rate                          |



```python
# ── Training schedule ───────────────────────────────────────────────────
batch_size = 32
n_epochs = 30  # 30 epochs is enough for the 100-sample toy dataset

# ── Per-source autoencoder parameters ────────────────────────────────────
# x_dim is a dict {source_name: n_features}, computed from the data above.
# Using a dict comprehension keeps source_params in sync with x_dim automatically.
source_params = {
    source: {
        "input_dim": x_dim[source],  # auto-filled — do not hardcode
        "hidden_dim": [256, 128],
        "latent_dim": 64,
        "norm": True,
        "dropout": 0.2,
    }
    for source in x_dim
}

# ── Central VAE parameters ────────────────────────────────────────────
central_params = {
    "hidden_dim": [256, 128],
    "latent_dim": 64,  # dimension of the shared latent code z
    "norm": True,
    "dropout": 0.2,
    "beta": 1.0,  # MMD regularisation weight
}

# ── Classifier head parameters ──────────────────────────────────────
classif_params = {
    "n_class": 5,  # subtypes 1-5
    "lambda": 5.0,  # classification loss weight
    "hidden_layers": [64, 32],
    "dropout": 0.2,
}

# ── Survival head parameters ────────────────────────────────────────
# lambda=0 disables the Cox loss; the survival head is still built but not trained.
# To enable survival, set lambda > 0 and ensure OS / OS.time contain real data.
surv_params = {
    "lambda": 0.0,
    "dims": [32, 16],
    "activation": "SELU",
    "l2_reg": 1e-2,
    "norm": True,
    "dropout": 0.2,
}

# ── Optimiser & phase schedule ──────────────────────────────────────
# switch=15 means:
#   epochs 0–14  → Phase 1: source AEs train independently
#   epochs 15–29 → Phase 2: central VAE takes over, heads are jointly optimised
train_params = {"switch": 15, "lr": 1e-3}

print("source_params keys :", list(source_params.keys()))
print("input dimensions   :", {k: v["input_dim"] for k, v in source_params.items()})
print("central latent dim :", central_params["latent_dim"])
print(
    f"training phases    : phase 1 → epochs 0–{train_params['switch'] - 1}, "
    f"phase 2 → epochs {train_params['switch']}–{n_epochs - 1}"
)
```

    source_params keys : ['rna', 'protein', 'methyl']
    input dimensions   : {'rna': 131, 'protein': 160, 'methyl': 367}
    central latent dim : 64
    training phases    : phase 1 → epochs 0–14, phase 2 → epochs 15–29


---

## 4. Building and Training the Model

### Model instantiation

Pass the five config dicts to `CustOMICS`. The model is built immediately but
weights are random until `fit()` is called.



```python
model = CustOMICS(
    source_params=source_params,
    central_params=central_params,
    classif_params=classif_params,
    surv_params=surv_params,
    train_params=train_params,
)
print(f"Total trainable parameters: {model.get_number_parameters():,}")
```

    [36;20m[INFO] (customics.model)[0m Using cpu by default.


    Total trainable parameters: 790,778



```python
# fit() reads label/event/surv_time from mdata.uns (registered by prepare_input above).
# omics_val triggers validation loss logging after each epoch.
model.fit(
    mdata=mdata_train,
    omics_val=mdata_val,
    batch_size=batch_size,
    n_epochs=n_epochs,
    verbose=True,
);
```

    [36;20m[INFO] (customics.model)[0m Epoch 1/30 | train=28.1554 | val=31.3453


    [36;20m[INFO] (customics.model)[0m Epoch 2/30 | train=19.8262 | val=29.0017


    [36;20m[INFO] (customics.model)[0m Epoch 3/30 | train=18.5581 | val=25.5669


    [36;20m[INFO] (customics.model)[0m Epoch 4/30 | train=14.8026 | val=22.5208


    [36;20m[INFO] (customics.model)[0m Epoch 5/30 | train=16.5802 | val=19.5704


    [36;20m[INFO] (customics.model)[0m Epoch 6/30 | train=12.9810 | val=17.8585


    [36;20m[INFO] (customics.model)[0m Epoch 7/30 | train=11.4881 | val=16.9921


    [36;20m[INFO] (customics.model)[0m Epoch 8/30 | train=10.1333 | val=15.3518


    [36;20m[INFO] (customics.model)[0m Epoch 9/30 | train=10.5262 | val=15.3346


    [36;20m[INFO] (customics.model)[0m Epoch 10/30 | train=10.1794 | val=14.3454


    [36;20m[INFO] (customics.model)[0m Epoch 11/30 | train=10.5995 | val=13.7073


    [36;20m[INFO] (customics.model)[0m Epoch 12/30 | train=7.6281 | val=14.0633


    [36;20m[INFO] (customics.model)[0m Epoch 13/30 | train=7.1182 | val=13.5002


    [36;20m[INFO] (customics.model)[0m Epoch 14/30 | train=7.9152 | val=12.4953


    [36;20m[INFO] (customics.model)[0m Epoch 15/30 | train=6.1834 | val=12.7738


    [36;20m[INFO] (customics.model)[0m Epoch 16/30 | train=7.1652 | val=8.7022


    [36;20m[INFO] (customics.model)[0m Epoch 17/30 | train=5.0774 | val=7.4710


    [36;20m[INFO] (customics.model)[0m Epoch 18/30 | train=6.6187 | val=6.1886


    [36;20m[INFO] (customics.model)[0m Epoch 19/30 | train=3.7562 | val=5.8766


    [36;20m[INFO] (customics.model)[0m Epoch 20/30 | train=4.1006 | val=5.7664


    [36;20m[INFO] (customics.model)[0m Epoch 21/30 | train=4.7605 | val=5.6158


    [36;20m[INFO] (customics.model)[0m Epoch 22/30 | train=3.9775 | val=5.4532


    [36;20m[INFO] (customics.model)[0m Epoch 23/30 | train=3.2129 | val=5.4427


    [36;20m[INFO] (customics.model)[0m Epoch 24/30 | train=4.0675 | val=5.0563


    [36;20m[INFO] (customics.model)[0m Epoch 25/30 | train=4.6486 | val=5.2825


    [36;20m[INFO] (customics.model)[0m Epoch 26/30 | train=5.0474 | val=5.1783


    [36;20m[INFO] (customics.model)[0m Epoch 27/30 | train=4.5562 | val=5.3988


    [36;20m[INFO] (customics.model)[0m Epoch 28/30 | train=3.4257 | val=5.2837


    [36;20m[INFO] (customics.model)[0m Epoch 29/30 | train=4.1434 | val=4.8389


    [36;20m[INFO] (customics.model)[0m Epoch 30/30 | train=4.3941 | val=4.9779



```python
# Plot the train/validation loss curves to check for overfitting.
# The vertical dashed line marks the phase 1 → phase 2 transition.
model.plot_loss()
```



![png](usage_files/usage_21_0.png)



---

## 5. Evaluating the Model

`evaluate()` runs inference on the held-out **test set** and returns a metrics dict.

- **Classification** (`task="classification"`): accuracy, macro F1, weighted F1,
  and per-class ROC-AUC.
- **Survival** (`task="survival"`): concordance index (C-index) via the Cox head.

> Set `plot_roc=True` to overlay per-class ROC curves on a single figure.



```python
# Evaluate tumour-subtype classification on the test split.
# label/event/surv_time are read from mdata_test.uns (registered by prepare_input).
metrics = model.evaluate(
    mdata=mdata_test,
    task="classification",
    batch_size=1024,
    plot_roc=True,
)

print("\n── Classification metrics on the test set ──")
for k, v in metrics.items():
    print(f"  {k:<25s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
```


    ── Classification metrics on the test set ──
      Accuracy                 : 1.0000
      F1-score                 : 1.0000
      Precision                : 1.0000
      Recall                   : 1.0000
      AUC                      : 1.0000




![png](usage_files/usage_23_1.png)




```python
# Evaluate the Cox survival head on the test split.
# With synthetic OS/OS.time the C-index will be ~0.5 (random), which is expected.
# Replace OS/OS.time with real data to obtain meaningful survival performance.
surv_metrics = model.evaluate(
    mdata=mdata_test,
    task="survival",
    batch_size=1024,
)

print("\n── Survival metrics on the test set ──")
print("C-index :", surv_metrics)
```


    ── Survival metrics on the test set ──
    C-index : 0.3723404255319149


---

## 6. Visualising the Latent Space

`get_latent_representation()` encodes every sample in `mdata` through the
trained central VAE and returns a NumPy array of shape `(n_samples, latent_dim)`.

`plot_representation()` runs t-SNE on that array and colours each point by the
clinical label — a quick sanity check that the model has learned subtype-discriminative
features.



```python
from pathlib import Path

# Get the latent embedding for ALL samples (train + val + test combined).
z = model.get_latent_representation(mdata)
print(f"Latent space shape: {z.shape}  (n_samples × latent_dim)")


# Create the results directory if it doesn't exist yet.
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

model.plot_representation(
    mdata=mdata,
    color=label,
    filename=RESULTS_DIR / "latent_representation",
    title="t-SNE of the integrated latent space",
    show=True,
)
```

    Latent space shape: (100, 64)  (n_samples × latent_dim)




![png](usage_files/usage_26_1.png)




    <Figure size 640x480 with 0 Axes>


---

## 7. Survival Risk Stratification

`stratify()` uses the Cox head to assign each sample a risk score, then splits
the cohort into high-risk / low-risk groups at the median score and draws a
Kaplan-Meier curve.

> **Note:** with synthetic survival data the curves will overlap.
> This section shows the API; meaningful separation requires real OS/OS.time.



```python
model.stratify(mdata=mdata, show=True)
```



![png](usage_files/usage_28_0.png)



---

## 8. Feature Importance with SHAP

`explain()` uses `shap.DeepExplainer` to compute feature-attribution values for
one omics source and one tumour subtype.

**Parameters:**

| Parameter   | Description                                                      |
| ----------- | ---------------------------------------------------------------- |
| `sample_id` | List of sample IDs to explain                                    |
| `mdata`     | Full `MuData` object                                             |
| `source`    | Which omics source to explain (`"rna"`, `"protein"`, `"methyl"`) |
| `subtype`   | Integer label of the class to explain (1-based here)             |
| `label`     | Column in `mdata.obs` containing the class labels                |
| `device`    | `"cpu"` or `"cuda"`                                              |
| `show`      | Display the SHAP summary plot inline                             |

The bar plot shows the **mean absolute SHAP value** for each feature — higher
means more influential for predicting the chosen subtype.



```python
# Feature importance for RNA expression → subtype 1
model.explain(sample_ids=shared_samples, mdata=mdata, source="rna", subtype=1, show=True)
```

    [36;20m[INFO] (customics.model)[0m Using cpu by default.




![png](usage_files/usage_30_1.png)




    <Figure size 640x480 with 0 Axes>



```python
# Feature importance for protein expression → subtype 1
model.explain(sample_ids=shared_samples, mdata=mdata, source="protein", subtype=1, show=True)
```

    [36;20m[INFO] (customics.model)[0m Using cpu by default.




![png](usage_files/usage_31_1.png)




    <Figure size 640x480 with 0 Axes>



```python
# Feature importance for DNA methylation → subtype 1
model.explain(sample_ids=shared_samples, mdata=mdata, source="methyl", subtype=1, show=True)
```

    [36;20m[INFO] (customics.model)[0m Using cpu by default.




![png](usage_files/usage_32_1.png)




    <Figure size 640x480 with 0 Axes>


---

## 9. Saving and Reloading the Model

`CustOMICS` uses PyTorch's standard `state_dict` / `load_state_dict` mechanism.
The helper methods `save()` and `load()` wrap this pattern so you don't need to
worry about the architecture parameters — `load()` reconstructs the full model
from the checkpoint.

> **Best practice:** save after the final training run so you can reproduce
> predictions without retraining.



```python
checkpoint_path = RESULTS_DIR / "customics_model.pt"

# save() stores both the architecture config and the weight tensors.
model.save(checkpoint_path)
```

    [36;20m[INFO] (customics.model)[0m Model saved to results/customics_model.pt



```python
# load() is a class method — it rebuilds the model from the checkpoint file.
loaded_model = CustOMICS.load(checkpoint_path)
print(f"Parameters in loaded model: {loaded_model.get_number_parameters():,}")
```

    [36;20m[INFO] (customics.model)[0m Using cpu by default.


    [36;20m[INFO] (customics.model)[0m Model loaded from results/customics_model.pt


    Parameters in loaded model: 790,778



```python
# Sanity-check: predictions from the reloaded model should match the originals.
metrics_reloaded = loaded_model.evaluate(
    mdata=mdata_test,
    task="classification",
    batch_size=1024,
    plot_roc=False,
)
print("\n── Reloaded model — classification metrics ──")
for k, v in metrics_reloaded.items():
    print(f"  {k:<25s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
```


    ── Reloaded model — classification metrics ──
      Accuracy                 : 1.0000
      F1-score                 : 1.0000
      Precision                : 1.0000
      Recall                   : 1.0000
      AUC                      : 1.0000
