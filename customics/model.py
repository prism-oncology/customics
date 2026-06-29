from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import KaplanMeierFitter
from mudata import MuData
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.optim import Adam
from torch.utils.data import DataLoader

from ._constants import Keys
from .datasets import MultiOmicsDataset
from .exceptions import ConfigurationError, DataValidationError, ModelNotFittedError
from .loss import CoxLoss, classification_loss
from .metrics import CIndex_lifeline, multi_classification_evaluation, plot_roc_multiclass
from .modules import VAE, AutoEncoder, Decoder, Encoder, ProbabilisticDecoder, ProbabilisticEncoder
from .tasks import MultiClassifier, SurvivalNet
from .utils import get_shared_samples

logger = logging.getLogger(__name__)


class CustOMICS(nn.Module):
    """Multi-omics integration model with hierarchical autoencoders and multi-task learning."""

    def __init__(
        self,
        source_params: dict[str, dict],
        central_params: dict,
        classif_params: dict,
        surv_params: dict,
        train_params: dict,
        device: torch.device | None = None,
    ) -> None:
        """Model initialization.

        Args:
            source_params:
                Per-source configuration.  Keys are source names; each value is a dict
                with the following keys:

                * `input_dim` (int): number of input features.
                * `hidden_dim` (list of int): hidden layer sizes.
                * `latent_dim` (int): per-source latent dimension.
                * `norm` (bool): whether to use batch normalisation.
                * `dropout` (float): dropout rate in `[0, 1]`.

            central_params:
                Central VAE configuration.  Required keys: `hidden_dim` (list of
                int), `latent_dim` (int), `norm` (bool), `dropout` (float),
                `beta` (float — MMD regularisation weight).

            classif_params:
                Classifier configuration.  Required keys: `n_class` (int, >= 2),
                `lambda` (float — loss weight), `hidden_layers` (list of int),
                `dropout` (float).

            surv_params:
                Survival-predictor configuration.  Required keys: `lambda` (float),
                `dims` (list of int), `activation` (str), `l2_reg` (float),
                `norm` (bool), `dropout` (float).

            train_params:
                Training hyperparameters.  Required keys: `switch` (int — epoch at
                which to enter phase 2) and `lr` (float — learning rate).

            device:
                Torch compute device.
        """
        super().__init__()
        self._validate_params(source_params, central_params, classif_params, train_params)

        self.device = _parse_device(device)
        self.source_names: list[str] = list(source_params.keys())
        self.n_source = len(self.source_names)
        self.beta = central_params["beta"]
        self.num_classes = classif_params["n_class"]
        self.lambda_classif = classif_params["lambda"]
        self.lambda_survival = surv_params["lambda"]
        self.switch_epoch = train_params["switch"]
        self.lr = train_params["lr"]
        self.phase = 1
        self._is_fitted = False

        # Store config dicts so save() can serialise them.
        self._source_params = source_params
        self._central_params = central_params
        self._classif_params = classif_params
        self._surv_params = surv_params
        self._train_params = train_params

        # ------------------------------------------------------------------ #
        # Per-source autoencoders (nn.ModuleList so PyTorch tracks them)
        # ------------------------------------------------------------------ #
        self.autoencoders = nn.ModuleList([
            AutoEncoder(
                encoder=Encoder(
                    input_dim=source_params[s]["input_dim"],
                    hidden_dim=source_params[s]["hidden_dim"],
                    latent_dim=source_params[s]["latent_dim"],
                    norm_layer=source_params[s]["norm"],
                    dropout=source_params[s]["dropout"],
                ),
                decoder=Decoder(
                    latent_dim=source_params[s]["latent_dim"],
                    hidden_dim=source_params[s]["hidden_dim"],
                    output_dim=source_params[s]["input_dim"],
                    norm_layer=source_params[s]["norm"],
                    dropout=source_params[s]["dropout"],
                ),
                device=self.device,
            )
            for s in self.source_names
        ])

        # ------------------------------------------------------------------ #
        # Central VAE
        # ------------------------------------------------------------------ #
        self.rep_dim = sum(source_params[s]["latent_dim"] for s in self.source_names)
        self.central_layer = VAE(
            encoder=ProbabilisticEncoder(
                input_dim=self.rep_dim,
                hidden_dim=central_params["hidden_dim"],
                latent_dim=central_params["latent_dim"],
                norm_layer=central_params["norm"],
                dropout=central_params["dropout"],
            ),
            decoder=ProbabilisticDecoder(
                latent_dim=central_params["latent_dim"],
                hidden_dim=central_params["hidden_dim"],
                output_dim=self.rep_dim,
                norm_layer=central_params["norm"],
                dropout=central_params["dropout"],
            ),
            device=self.device,
        )

        # ------------------------------------------------------------------ #
        # Task heads
        # ------------------------------------------------------------------ #
        self.classifier = MultiClassifier(
            n_class=self.num_classes,
            latent_dim=central_params["latent_dim"],
            dropout=classif_params["dropout"],
            class_dim=classif_params["hidden_layers"],
        )
        self.survival_predictor = SurvivalNet({
            "drop": surv_params["dropout"],
            "norm": surv_params["norm"],
            "dims": [central_params["latent_dim"]] + surv_params["dims"] + [1],
            "activation": surv_params["activation"],
        })

        self._relocate()
        self.optimizer = self._build_optimizer()

        # Filled during fit()
        self.history: list[tuple] = []
        self.label_encoder: LabelEncoder | None = None
        self.one_hot_encoder: OneHotEncoder | None = None
        self.baseline = None

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_params(
        source_params: dict,
        central_params: dict,
        classif_params: dict,
        train_params: dict,
    ) -> None:
        if not source_params:
            raise ConfigurationError("source_params must contain at least one source.")
        for name, sp in source_params.items():
            for key in ("input_dim", "hidden_dim", "latent_dim", "norm", "dropout"):
                if key not in sp:
                    raise ConfigurationError(f"source_params['{name}'] is missing required key '{key}'.")
            if not 0.0 <= sp["dropout"] <= 1.0:
                raise ConfigurationError(f"dropout for source '{name}' must be in [0, 1], got {sp['dropout']}.")
        if classif_params.get("n_class", 0) < 2:
            raise ConfigurationError("classif_params['n_class'] must be >= 2.")
        for key in ("switch", "lr"):
            if key not in train_params:
                raise ConfigurationError(f"train_params is missing required key '{key}'.")

    def _build_optimizer(self) -> Adam:
        return Adam(self.parameters(), lr=self.lr)

    def _relocate(self) -> None:
        self.autoencoders.to(self.device)
        self.central_layer.to(self.device)
        self.classifier.to(self.device)
        self.survival_predictor.to(self.device)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ModelNotFittedError("This CustOMICS instance has not been fitted yet. Call fit() first.")

    # ------------------------------------------------------------------ #
    # Phase management
    # ------------------------------------------------------------------ #

    def _switch_phase(self, epoch: int) -> None:
        self.phase = 1 if epoch < self.switch_epoch else 2

    # ------------------------------------------------------------------ #
    # Forward pass (nn.Module interface)
    # ------------------------------------------------------------------ #

    def forward(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Full forward pass through per-source AEs and central encoder.

        Args:
            x: One tensor per omics source, shape `(batch, features_i)`.

        Returns:
            A tuple `(reconstructions, representations, mean)`:
                - reconstructions: Per-source reconstructions.
                - representations: Per-source latent vectors.
                - mean: Central VAE posterior mean, shape `(batch, central_latent_dim)`.
        """
        reconstructions, representations = [], []
        for xi, ae in zip(x, self.autoencoders):
            hat, rep = ae(xi)
            reconstructions.append(hat)
            representations.append(rep)
        mean, _ = self.central_layer.encoder(torch.cat(representations, dim=1))
        return reconstructions, representations, mean

    # ------------------------------------------------------------------ #
    # Internal helpers for training
    # ------------------------------------------------------------------ #

    def _compute_training_loss(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(latent_z, reconstruction_loss)` for the current phase.

        Phase 1 returns the first source's representation; phase 2 returns the
        central VAE mean.  Both are used as input to the task heads during
        training.
        """
        representations: list[torch.Tensor] = []
        reconstruction_loss = torch.tensor(0.0, device=self.device)
        for xi, ae in zip(x, self.autoencoders):
            _, rep = ae(xi)
            representations.append(rep)
            reconstruction_loss = reconstruction_loss + ae.loss(xi, self.beta)

        if self.phase == 1:
            return representations, reconstruction_loss

        central_concat = torch.cat(representations, dim=1)
        reconstruction_loss = reconstruction_loss + self.central_layer.loss(central_concat, self.beta)
        mean, _ = self.central_layer.encoder(central_concat)
        return mean, reconstruction_loss

    def _get_central_representation(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Always return the central VAE mean (used for inference)."""
        representations = [ae(xi)[1] for xi, ae in zip(x, self.autoencoders)]
        mean, _ = self.central_layer.encoder(torch.cat(representations, dim=1))
        return mean

    def _set_train_mode(self) -> None:
        self.autoencoders.train()
        self.central_layer.train()
        self.classifier.train()
        self.survival_predictor.train()

    def _set_eval_mode(self) -> None:
        self.autoencoders.eval()
        self.central_layer.eval()
        self.classifier.eval()
        self.survival_predictor.eval()

    def _train_step(
        self,
        x: list[torch.Tensor],
        labels: torch.Tensor,
        os_time: torch.Tensor,
        os_event: torch.Tensor,
    ) -> torch.Tensor:
        x = [xi.to(self.device) for xi in x]
        self.optimizer.zero_grad()

        z_or_reps, reconstruction_loss = self._compute_training_loss(x)

        if self.phase == 1:
            # Apply task heads to every per-source representation separately
            task_loss = torch.tensor(0.0, device=self.device)
            for z in z_or_reps:
                task_loss = task_loss + self.lambda_survival * CoxLoss(
                    os_time, os_event, self.survival_predictor(z), self.device
                )
                task_loss = task_loss + self.lambda_classif * classification_loss("CE", self.classifier(z), labels)
        else:
            z = z_or_reps
            task_loss = self.lambda_survival * CoxLoss(
                os_time, os_event, self.survival_predictor(z), self.device
            ) + self.lambda_classif * classification_loss("CE", self.classifier(z), labels)

        return reconstruction_loss + task_loss

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        total, n = 0.0, 0
        for x, labels, os_time, os_event in loader:
            if training:
                self._set_train_mode()
                loss = self._train_step(x, labels, os_time, os_event)
                loss.backward()
                self.optimizer.step()
            else:
                self._set_eval_mode()
                with torch.no_grad():
                    loss = self._train_step(x, labels, os_time, os_event)
            total += loss.item()
            n += 1
        return total / max(n, 1)

    # ------------------------------------------------------------------ #
    # Public training API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        mdata: MuData,
        omics_val: MuData | None = None,
        batch_size: int = 32,
        n_epochs: int = 30,
        verbose: bool = True,
    ) -> None:
        """Train the customics model.

        Args:
            mdata: Multi-omics object whose `obs` holds the clinical annotations.
            omics_val: Validation omics data; same format as `omics_train`.
            batch_size: Mini-batch size.
            n_epochs: Number of training epochs.
            verbose: Log epoch-level loss when True.

        Raises:
            DataValidationError: If required columns are missing or samples don't overlap.
        """

        label, event, surv_time = self._validate_fit_inputs(mdata)

        self.label_encoder = LabelEncoder().fit(mdata.obs[label].values)
        train_labels = pd.Series(self.label_encoder.transform(mdata.obs[label].values), index=mdata.obs_names)
        # Fit OHE on integer-encoded labels so it can transform integer y_true at eval time
        self.one_hot_encoder = OneHotEncoder(sparse_output=False).fit(train_labels.values.reshape(-1, 1))

        loader_kw: dict = {"num_workers": 2, "pin_memory": True} if self.device.type == "cuda" else {}

        shared_samples_train = get_shared_samples(mdata)
        self.baseline = self._compute_baseline(mdata.obs, shared_samples_train, event, surv_time)
        train_loader = DataLoader(
            MultiOmicsDataset(mdata, shared_samples_train, train_labels),
            batch_size=batch_size,
            shuffle=True,
            **loader_kw,
        )

        val_loader: DataLoader | None = None
        if omics_val is not None:
            shared_samples_val = get_shared_samples(omics_val)
            val_labels = pd.Series(self.label_encoder.transform(omics_val.obs[label].values), index=omics_val.obs_names)
            val_loader = DataLoader(
                MultiOmicsDataset(omics_val, shared_samples_val, val_labels),
                batch_size=batch_size,
                shuffle=False,
                **loader_kw,
            )

        self.history = []
        for epoch in range(n_epochs):
            self._switch_phase(epoch)
            train_loss = self._run_epoch(train_loader, training=True)
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, training=False)
                self.history.append((train_loss, val_loss))
                if verbose:
                    logger.info(
                        "Epoch %d/%d | train=%.4f | val=%.4f",
                        epoch + 1,
                        n_epochs,
                        train_loss,
                        val_loss,
                    )
            else:
                self.history.append((train_loss,))
                if verbose:
                    logger.info("Epoch %d/%d | train=%.4f", epoch + 1, n_epochs, train_loss)

        self._is_fitted = True

    def _validate_fit_inputs(self, mdata: MuData) -> tuple[str, str, str]:
        if not all(key in mdata.uns for key in [Keys.LABEL, Keys.EVENT, Keys.SURV_TIME]):
            raise DataValidationError(
                "Clinical parameters are not registered in mdata.uns. Please run `customics.prepare_input` first."
            )

        for source, adata in mdata.mod.items():
            overlap = set(adata.obs_names) & set(mdata.obs_names)
            if not overlap:
                raise DataValidationError(f"Source '{source}' shares no sample IDs with clinical_data.")

        return mdata.uns[Keys.LABEL], mdata.uns[Keys.EVENT], mdata.uns[Keys.SURV_TIME]

    def _compute_baseline(
        self,
        clinical_df: pd.DataFrame,
        shared_samples: list[str],
        event: str,
        surv_time: str,
    ):
        kmf = KaplanMeierFitter()
        kmf.fit(clinical_df.loc[shared_samples, surv_time], clinical_df.loc[shared_samples, event])
        return kmf.survival_function_

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def get_latent_representation(
        self,
        mdata: MuData,
    ) -> np.ndarray:
        """Compute the integrated central latent representation.

        Args:
            mdata: Multi-omics object with all sources (same keys as used in `fit`).

        Returns:
            Latent matrix, shape `(n_samples, central_latent_dim)`. Rows
            correspond to `get_shared_samples(mdata)`, in that order.

        Raises:
            ModelNotFittedError: If called before `fit()`.
        """
        self._require_fitted()
        self._set_eval_mode()
        shared_samples = get_shared_samples(mdata)
        x = [
            torch.tensor(np.asarray(mdata[mod][shared_samples].X), dtype=torch.float32).to(self.device)
            for mod in self.source_names
        ]
        with torch.no_grad():
            z = self._get_central_representation(x)
        return z.cpu().numpy()

    def predict(self, mdata: MuData) -> np.ndarray:
        """Predict class labels.

        Args:
            mdata: Multi-omics object matching the sources used in `fit`.

        Returns:
            Integer class predictions, shape `(n_samples,)`.

        Raises:
            ModelNotFittedError: If called before `fit()`.
        """
        self._require_fitted()
        self._set_eval_mode()
        z = torch.tensor(self.get_latent_representation(mdata), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.classifier(z)
        return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_survival(self, mdata: MuData) -> dict[str, pd.DataFrame]:
        """Compute patient-level estimated survival functions.

        Args:
            mdata: Multi-omics object matching the sources used in `fit`.

        Returns:
            Maps sample ID → estimated survival function DataFrame.

        Raises:
            ModelNotFittedError: If called before `fit()`.
        """
        self._require_fitted()
        shared_samples = get_shared_samples(mdata)
        z = torch.tensor(self.get_latent_representation(mdata), dtype=torch.float32).to(self.device)
        self._set_eval_mode()
        with torch.no_grad():
            risk_scores = self.survival_predictor(z).cpu().numpy()
        return {s: self.baseline * np.exp(r[0]) for s, r in zip(shared_samples, risk_scores)}

    def source_predict(self, x: torch.Tensor, source: str) -> torch.Tensor:
        """Predict class logits from a single-source input tensor.

        Used internally by the SHAP `customics.explain.shap.ModelWrapper`.

        Args:
            x: Input tensor for `source`, shape `(batch, features)`.
            source: Source name (must be in `self.source_names`).

        Returns:
            Class logits, shape `(batch, n_class)`.
        """
        if source not in self.source_names:
            raise ValueError(f"Source '{source}' not recognised. Known sources: {self.source_names}.")
        idx = self.source_names.index(source)
        z = self.autoencoders[idx].encoder(x)
        return self.classifier(z)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        mdata: MuData,
        task: str,
        batch_size: int = 32,
        plot_roc: bool = False,
    ) -> float | dict[str, float]:
        """Evaluate the model on held-out data.

        Args:
            mdata: Multi-omics object whose `obs` holds the clinical annotations.
            task: `'classification'` or `'survival'`.
            batch_size: Evaluation batch size.
            plot_roc: Save a ROC curve image (classification only).

        Returns:
            Concordance index (float) for `task='survival'`, or a metrics dict
            (Accuracy, F1-score, Precision, Recall, AUC) for
            `task='classification'`.

        Raises:
            ModelNotFittedError: If called before `fit()`.
            ValueError: If `task` is not `'classification'` or `'survival'`.
        """
        self._require_fitted()
        if task not in ("classification", "survival"):
            raise ValueError(f"task must be 'classification' or 'survival', got '{task}'.")

        label = mdata.uns[Keys.LABEL]

        encoded_labels = pd.Series(self.label_encoder.transform(mdata.obs[label].values), index=mdata.obs_names)

        loader_kw: dict = {"num_workers": 2, "pin_memory": True} if self.device.type == "cuda" else {}
        shared_samples = get_shared_samples(mdata)
        test_loader = DataLoader(
            MultiOmicsDataset(mdata, shared_samples, encoded_labels),
            batch_size=batch_size,
            shuffle=False,
            **loader_kw,
        )

        self._set_eval_mode()
        all_y_true, all_y_pred, all_y_proba = [], [], []
        all_hazard, all_os_time, all_os_event = [], [], []

        with torch.no_grad():
            for x, labels, os_time, os_event in test_loader:
                x = [xi.to(self.device) for xi in x]
                z = self._get_central_representation(x)

                if task == "survival":
                    hazard = self.survival_predictor(z).cpu().numpy().reshape(-1, 1)
                    all_hazard.append(hazard)
                    all_os_time.append(os_time.numpy())
                    all_os_event.append(os_event.numpy())
                else:
                    logits = self.classifier(z)
                    all_y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
                    all_y_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
                    all_y_true.append(labels.cpu().numpy())

        if task == "survival":
            hazard_cat = np.vstack(all_hazard)
            return float(
                CIndex_lifeline(
                    hazard_cat,
                    np.concatenate(all_os_event),
                    np.concatenate(all_os_time),
                )
            )

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        y_proba = np.vstack(all_y_proba)

        if plot_roc:
            plot_roc_multiclass(
                y_test=y_true,
                y_pred_proba=y_proba,
                filename="test",
                n_classes=self.num_classes,
                var_names=np.unique(mdata.obs[label].values.tolist()).tolist(),
            )

        return multi_classification_evaluation(y_true, y_pred, y_proba, ohe=self.one_hot_encoder)

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #

    def explain(
        self,
        sample_ids: list[str],
        mdata: MuData,
        source: str,
        subtype: str,
        device: str | None = None,
        show: bool = True,
    ) -> None:
        """Compute and plot SHAP values for one omics source and one subtype.

        Uses `shap.DeepExplainer` on the single-source forward path.
        SHAP values are computed for `subtype` against all other classes.

        Args:
            sample_ids: Sample IDs to use as the SHAP background and foreground sets.
            mdata: Multi-omics object whose `obs` holds the clinical metadata.
            source: Omics source key to explain.
            subtype: Class label to explain.
            device: Device for SHAP tensors.
            show: Display the SHAP plot interactively.

        Raises:
            ModelNotFittedError: If called before `fit()`.
        """
        import matplotlib.pyplot as plt
        import shap

        from customics.explain.shap import (
            ModelWrapper,
            add_to_tensor,
            process_phenotype_data_for_samples,
            random_training_sample,
            split_expr_and_sample,
        )

        self._require_fitted()

        device = _parse_device(device)

        expr_df = mdata[source].to_df()
        sample_ids = list(set(sample_ids) & set(expr_df.index))
        phenotype = process_phenotype_data_for_samples(mdata.obs, sample_ids)
        condition = phenotype[mdata.uns[Keys.LABEL]] == subtype

        expr_df = expr_df.loc[sample_ids, :]
        background = add_to_tensor(random_training_sample(expr_df, 10), device)
        foreground_df = split_expr_and_sample(condition=condition, sample_size=10, expr=expr_df)
        foreground = add_to_tensor(foreground_df, device)

        class_idx = int(self.label_encoder.transform([subtype])[0])
        explainer = shap.DeepExplainer(ModelWrapper(self, source=source), background)
        shap_values = explainer.shap_values(foreground, ranked_outputs=None)

        # SHAP ≥0.46 stacks class outputs into (n_samples, n_features, n_classes);
        # older versions return a list indexed [class][sample, feature].
        import numpy as np

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[..., class_idx]
        else:
            sv = shap_values[class_idx]

        shap.summary_plot(
            sv,
            features=foreground_df,
            feature_names=list(expr_df.columns),
            show=False,
            plot_type="violin",
            max_display=10,
            plot_size=[4, 6],
        )

        if show:
            plt.show()

        # SHAP registers forward/backward hook tensors as nn.Parameter on each
        # module. Remove them so state_dict() stays clean for save/load.
        for module in self.modules():
            module._parameters.pop("x", None)
            module._parameters.pop("y", None)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def get_number_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def plot_loss(self, show: bool = True) -> None:
        """Plot training (and validation) loss history.

        Args:
            show: Display the figure interactively.
        """
        from customics.visualization import plot_loss as _plot

        _plot(self.history, self.switch_epoch, show=show)

    def plot_representation(self, mdata: MuData, color: str | None = None, show: bool = True) -> None:
        """Compute latent representations and save a t-SNE scatter plot.

        Args:
            mdata: Multi-omics object.
            color: Column in `mdata.obs` to use for colouring samples. By default, use the label column.
            show: Display the figure interactively.
        """
        from customics.visualization import plot_representation as _plot

        _plot(self, mdata, color or mdata.uns[Keys.LABEL], show=show)

    def stratify(self, mdata: MuData, show: bool = True) -> None:
        """Stratify patients by predicted risk and plot Kaplan-Meier curves.

        Args:
            mdata: Multi-omics object.
            show: Display the figure interactively.
        """
        from customics.visualization import plot_survival_stratification as _plot

        _plot(self, mdata, mdata.uns[Keys.EVENT], mdata.uns[Keys.SURV_TIME], show)

    # ---------------------------------------------------------------------- #
    # Serialisation
    # ---------------------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        """Save the model architecture config and trained weights to *path*.

        The checkpoint contains the five parameter dicts needed to reconstruct
        the model plus the `state_dict` and fitted encoders so that inference
        works immediately after [load](api/train/#customics.CustOMICS.load).

        Args:
            path: Destination file (conventionally `*.pt` or `*.pth`).
        """
        checkpoint = {
            "source_params": self._source_params,
            "central_params": self._central_params,
            "classif_params": self._classif_params,
            "surv_params": self._surv_params,
            "train_params": self._train_params,
            "state_dict": self.state_dict(),
            "label_encoder": self.label_encoder,
            "one_hot_encoder": self.one_hot_encoder,
            "history": self.history,
            "_is_fitted": self._is_fitted,
        }

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> CustOMICS:
        """Load a model previously saved with [save](api/train/#customics.CustOMICS.save).

        Args:
            path: Path to the checkpoint file written by [save](api/train/#customics.CustOMICS.save).
            device: Target device.  Defaults to CPU when not specified.

        Returns:
            A fully initialised, ready-to-use model instance.
        """
        device = _parse_device(device)

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            source_params=checkpoint["source_params"],
            central_params=checkpoint["central_params"],
            classif_params=checkpoint["classif_params"],
            surv_params=checkpoint["surv_params"],
            train_params=checkpoint["train_params"],
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.label_encoder = checkpoint.get("label_encoder")
        model.one_hot_encoder = checkpoint.get("one_hot_encoder")
        model.history = checkpoint.get("history", [])
        model._is_fitted = checkpoint.get("_is_fitted", True)
        model.to(device)
        logger.info(f"Model loaded from {path}")
        return model


def _parse_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using {device} by default.")
        return device
    return device
