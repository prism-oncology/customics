# Hyperparameter Tuning

This guide gives practical recommendations for configuring `CustOMICS`.
The model has five parameter groups; the tables below describe each key,
its effect, and a sensible starting point scaled to your dataset.

---

## Quick-start defaults

For a first run, use the values below and iterate from there.
They are designed for a medium-sized cohort (200–1 000 samples, 3 omics sources).

```python
source_params = {
    source: {
        "input_dim": x_dim[source],   # always inferred from the data
        "hidden_dim": [256, 128],
        "latent_dim": 64,
        "norm": True,
        "dropout": 0.2,
    }
    for source in x_dim
}

central_params = {
    "hidden_dim": [256, 128],
    "latent_dim": 64,
    "norm": True,
    "dropout": 0.2,
    "beta": 1.0,
}

classif_params = {
    "n_class": N_CLASSES,
    "lambda": 5.0,
    "hidden_layers": [64, 32],
    "dropout": 0.2,
}

surv_params = {
    "lambda": 1.0,          # set to 0 to disable survival
    "dims": [32, 16],
    "activation": "SELU",
    "l2_reg": 1e-2,
    "norm": True,
    "dropout": 0.2,
}

train_params = {"switch": 15, "lr": 1e-3}
```

---

## Source autoencoders (`source_params`)

One entry per omics modality. The autoencoder compresses raw features into a
compact per-source latent code before integration.

| Key | Effect | Recommendation |
|---|---|---|
| `input_dim` | Number of input features | Always set from the data (`df.shape[1]`). Never hardcode. |
| `hidden_dim` | Encoder hidden layer sizes (decoder mirrors in reverse) | Start with `[256, 128]`. For high-dimensional sources (>5 000 features) add a layer: `[1024, 512, 128]`. Reduce for small sources (<200 features): `[128, 64]`. |
| `latent_dim` | Per-source embedding dimension | 32–128. A common rule: 5–10× smaller than `hidden_dim[-1]`. Sources with more features can use a larger `latent_dim`. All sources feed into the central VAE, so keep values consistent. |
| `norm` | BatchNorm after each hidden layer | `True` in almost all cases. Disable only with very small batches (< 8 samples) where BatchNorm statistics are unreliable. |
| `dropout` | Dropout rate | 0.1–0.3. Increase toward 0.4–0.5 if the model overfits. Set to 0 for very small datasets where regularisation hurts more than it helps. |

**Scaling with dataset size**

| Cohort size | Suggested `hidden_dim` | Suggested `latent_dim` |
|---|---|---|
| < 100 samples | `[128, 64]` | 16–32 |
| 100–500 samples | `[256, 128]` | 32–64 |
| 500–2 000 samples | `[512, 256]` | 64–128 |
| > 2 000 samples | `[1024, 512, 256]` | 128–256 |

---

## Central VAE (`central_params`)

Receives the concatenation of all per-source latent codes and produces the
shared latent **z** used by both task heads.

| Key | Effect | Recommendation |
|---|---|---|
| `hidden_dim` | VAE encoder/decoder hidden sizes | Match or slightly narrow the source encoder output. A good default is the same as `source_params["hidden_dim"]`. |
| `latent_dim` | Dimension of the integrated representation **z** | Similar to or slightly smaller than each source `latent_dim`. The central VAE sees `n_sources × source_latent_dim` features as input, so it has room to compress. |
| `beta` | MMD regularisation weight | Start at `1.0`. Increase (2–5) for better-structured latent spaces at the cost of reconstruction quality. Decrease (0.1–0.5) if training is unstable or reconstruction loss dominates. |
| `norm` | BatchNorm | Same advice as source autoencoders. |
| `dropout` | Dropout rate | Same advice as source autoencoders. |

---

## Classifier head (`classif_params`)

A small MLP that maps **z** to tumour subtype probabilities.

| Key | Effect | Recommendation |
|---|---|---|
| `n_class` | Number of output classes | Must match the number of unique labels in your dataset. |
| `lambda` | Weight of the classification loss | 1–10. Increase if classification metrics are poor relative to reconstruction. Decrease if the model ignores the reconstruction objective. |
| `hidden_layers` | MLP hidden sizes | Keep it simple: `[64, 32]` or even `[32]` is usually enough. The classifier is downstream of a well-regularised latent space. |
| `dropout` | Dropout rate | 0.1–0.3. |

!!! tip "Imbalanced classes"
    If your class distribution is heavily imbalanced, consider increasing `lambda`
    for the minority classes or applying sample weights in your data split.

---

## Survival head (`surv_params`)

A Cox proportional-hazards network that maps **z** to a log-hazard score.

| Key | Effect | Recommendation |
|---|---|---|
| `lambda` | Weight of the Cox loss | Set to `0` to disable survival entirely. When enabled, start at `0.5–1.0`. If the C-index is near 0.5 (random), the survival signal may not be strong enough to outweigh the classification signal — try reducing `lambda_classif` or increasing `lambda` here. |
| `dims` | Hidden layer sizes | `[32, 16]` is usually enough. Avoid overly deep survival nets. |
| `activation` | Activation function | `"SELU"` is the standard choice for Cox networks (self-normalising). `"ReLU"` or `"LeakyReLU"` are valid alternatives. |
| `l2_reg` | L2 regularisation on survival weights | 1e-2 to 1e-4. Increase if the survival head overfits. |
| `norm` | BatchNorm | `True` generally helps. |
| `dropout` | Dropout rate | 0.2–0.4. |

!!! note "Synthetic survival data"
    The toy dataset uses randomly generated `OS` / `OS.time` columns.
    A C-index around 0.5 (random) is expected and normal.
    Replace with real survival data to obtain meaningful survival performance.

---

## Training schedule (`train_params`)

| Key | Effect | Recommendation |
|---|---|---|
| `switch` | Epoch at which phase 1 ends and phase 2 begins | Roughly 30–50% of `n_epochs`. With 30 epochs, `switch=15` is a good starting point. Too early: source representations are not stable before integration. Too late: the central VAE has too few epochs to converge. |
| `lr` | Adam learning rate | `1e-3` is a safe default. If training is unstable, try `5e-4`. For fine-tuning a pretrained model, use `1e-4`. |

**Batch size** (passed directly to `fit()`):

- 32 is a good default.
- With very small datasets (< 100 samples), 16 or even 8 may work better.
- With large datasets and a GPU, 128–256 increases throughput with minimal accuracy loss.

**Number of epochs:**

- Small datasets (< 200 samples): 30–50 epochs is usually sufficient.
- Medium datasets: 50–100 epochs.
- Large datasets: 100–200 epochs. Monitor validation loss with `plot_loss()` and stop when it plateaus.

---

## Common issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Training loss oscillates wildly | Learning rate too high | Reduce `lr` to `5e-4` or `1e-4` |
| Validation loss diverges after `switch` | Phase 2 destabilises under-trained source AEs | Increase `switch` (more phase-1 epochs) |
| Classification metrics plateau near chance | `lambda` (classif) too low, or `latent_dim` too small | Increase `classif_params["lambda"]` or `central_params["latent_dim"]` |
| t-SNE shows no cluster structure | Central VAE not converging | Increase `beta`, or give more phase-2 epochs by reducing `switch` |
| `explain()` returns SHAP values near zero | Source `latent_dim` too large relative to dataset | Reduce `latent_dim`; sparser representations are more interpretable |
| Memory error on GPU | Batch size or model too large | Reduce `batch_size`; reduce `hidden_dim` |
| BatchNorm error with small batches | `norm=True` with fewer than 8 samples per batch | Set `norm=False` or increase `batch_size` |
