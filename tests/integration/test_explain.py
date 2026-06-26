"""Integration tests for model.explain() (SHAP-based feature attribution)."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture()
def explain_kwargs(mdata):
    """Minimal keyword arguments for explain() using the 'rna' source."""
    return {
        "sample_id": list(mdata.obs_names),
        "mdata": mdata,
        "source": "rna",
        "subtype": "TypeA",
        "label": "label",
        "device": "cpu",
        "show": False,
    }


class TestExplainSmoke:
    def test_explain_rna_runs(self, fitted_model, explain_kwargs, tmp_path, monkeypatch):
        """explain() completes without error for the rna source."""
        monkeypatch.chdir(tmp_path)
        fitted_model.explain(**explain_kwargs)

    def test_explain_cnv_runs(self, fitted_model, mdata, tmp_path, monkeypatch):
        """explain() completes without error for the cnv source."""
        monkeypatch.chdir(tmp_path)
        fitted_model.explain(
            sample_id=list(mdata.obs_names),
            mdata=mdata,
            source="cnv",
            subtype="TypeA",
            label="label",
            device="cpu",
            show=False,
        )

    def test_explain_saves_png(self, fitted_model, explain_kwargs, tmp_path, monkeypatch):
        """explain() must write shap_{source}_{subtype}.png to the working directory."""
        monkeypatch.chdir(tmp_path)
        fitted_model.explain(**explain_kwargs)
        assert (tmp_path / "shap_rna_TypeA.png").exists()

    def test_explain_returns_none(self, fitted_model, explain_kwargs, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = fitted_model.explain(**explain_kwargs)
        assert result is None


class TestExplainStateDict:
    """SHAP injects .x/.y tensors as nn.Parameter on modules.

    After explain() these must be cleaned up so that state_dict() and
    save()/load() continue to work (regression guard for the bug that caused
    'Unexpected key(s) in state_dict' on reload).
    """

    def test_no_shap_params_in_state_dict(self, fitted_model, explain_kwargs, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fitted_model.explain(**explain_kwargs)
        state = fitted_model.state_dict()
        shap_keys = [k for k in state if k.endswith(".x") or k.endswith(".y")]
        assert shap_keys == [], f"SHAP parameter(s) leaked into state_dict: {shap_keys}"

    def test_save_load_after_explain(self, fitted_model, explain_kwargs, tmp_path, monkeypatch):
        """save() then CustOMICS.load() must succeed after explain()."""
        from customics import CustOMICS

        monkeypatch.chdir(tmp_path)
        fitted_model.explain(**explain_kwargs)

        path = tmp_path / "model_post_explain.pt"
        fitted_model.save(path)

        loaded = CustOMICS.load(path, device=fitted_model.device)
        assert loaded.get_number_parameters() == fitted_model.get_number_parameters()

    def test_predictions_unchanged_after_explain(self, fitted_model, explain_kwargs, mdata, tmp_path, monkeypatch):
        """explain() must not alter the model weights (predictions must be identical before/after)."""
        preds_before = fitted_model.predict(mdata).copy()
        monkeypatch.chdir(tmp_path)
        fitted_model.explain(**explain_kwargs)
        preds_after = fitted_model.predict(mdata)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_multiple_explain_calls_stay_clean(self, fitted_model, mdata, tmp_path, monkeypatch):
        """Calling explain() twice must not accumulate stale SHAP parameters."""
        monkeypatch.chdir(tmp_path)
        for subtype in ["TypeA", "TypeB"]:
            fitted_model.explain(
                sample_id=list(mdata.obs_names),
                mdata=mdata,
                source="rna",
                subtype=subtype,
                label="label",
                device="cpu",
                show=False,
            )
        state = fitted_model.state_dict()
        shap_keys = [k for k in state if k.endswith(".x") or k.endswith(".y")]
        assert shap_keys == []


class TestExplainEdgeCases:
    def test_subset_of_samples(self, fitted_model, mdata, tmp_path, monkeypatch):
        """explain() with a partial sample list must not raise."""
        monkeypatch.chdir(tmp_path)
        partial = list(mdata.obs_names[:5])
        fitted_model.explain(
            sample_id=partial,
            mdata=mdata,
            source="rna",
            subtype="TypeA",
            label="label",
            device="cpu",
            show=False,
        )

    def test_unknown_samples_are_silently_dropped(self, fitted_model, mdata, tmp_path, monkeypatch):
        """Sample IDs not in the source modality are silently filtered out."""
        monkeypatch.chdir(tmp_path)
        sample_ids = [*list(mdata.obs_names[:5]), "DOES_NOT_EXIST"]
        fitted_model.explain(
            sample_id=sample_ids,
            mdata=mdata,
            source="rna",
            subtype="TypeA",
            label="label",
            device="cpu",
            show=False,
        )
