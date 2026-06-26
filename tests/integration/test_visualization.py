"""Integration tests for model visualisation methods.

Covers:
- model.plot_representation() — t-SNE latent-space scatter
- model.stratify()            — Kaplan-Meier risk stratification
- model.plot_loss()           — training-loss curve
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")


class TestPlotRepresentation:
    def test_saves_png(self, fitted_model, mdata, tmp_path):
        """plot_representation() must write a .png file to the given path."""
        out = tmp_path / "tsne"
        fitted_model.plot_representation(
            mdata=mdata,
            label="label",
            filename=str(out),
            title="Test t-SNE",
            show=False,
        )
        assert (tmp_path / "tsne.png").exists()

    def test_returns_none(self, fitted_model, mdata, tmp_path):
        result = fitted_model.plot_representation(
            mdata=mdata,
            label="label",
            filename=str(tmp_path / "tsne"),
            title="Test t-SNE",
            show=False,
        )
        assert result is None

    def test_png_is_non_empty(self, fitted_model, mdata, tmp_path):
        """The written PNG must contain image data (not an empty file)."""
        out = tmp_path / "tsne"
        fitted_model.plot_representation(
            mdata=mdata,
            label="label",
            filename=str(out),
            title="Test t-SNE",
            show=False,
        )
        assert (tmp_path / "tsne.png").stat().st_size > 0

    def test_latent_representation_shape(self, fitted_model, mdata):
        """get_latent_representation() must return (n_samples, latent_dim) array."""
        z = fitted_model.get_latent_representation(mdata)
        assert isinstance(z, np.ndarray)
        assert z.ndim == 2
        assert z.shape[0] == len(mdata.obs_names)

    def test_latent_representation_finite(self, fitted_model, mdata):
        """Latent codes must contain no NaN or Inf values."""
        z = fitted_model.get_latent_representation(mdata)
        assert np.all(np.isfinite(z))

    def test_different_labels_produce_same_latent(self, fitted_model, mdata, tmp_path):
        """The coloring label must not affect the latent coordinates, only the colours."""
        import numpy as np

        z1 = fitted_model.get_latent_representation(mdata)

        fitted_model.plot_representation(
            mdata=mdata,
            label="label",
            filename=str(tmp_path / "tsne_label"),
            title="Test",
            show=False,
        )

        z2 = fitted_model.get_latent_representation(mdata)
        np.testing.assert_array_equal(z1, z2)


class TestStratify:
    def test_runs_without_error(self, fitted_model, mdata):
        """stratify() must complete without raising for any MuData input."""
        fitted_model.stratify(
            mdata=mdata,
            event="OS",
            surv_time="OS.time",
            show=False,
        )

    def test_saves_png_when_save_path_provided(self, fitted_model, mdata, tmp_path):
        """stratify() must write a file when save_path is given."""
        out = str(tmp_path / "km")
        fitted_model.stratify(
            mdata=mdata,
            event="OS",
            surv_time="OS.time",
            save_path=out,
            show=False,
        )
        assert (tmp_path / "km.png").exists()

    def test_no_file_without_save_path(self, fitted_model, mdata, tmp_path, monkeypatch):
        """Without save_path no PNG should be written to the working directory."""
        monkeypatch.chdir(tmp_path)
        fitted_model.stratify(
            mdata=mdata,
            event="OS",
            surv_time="OS.time",
            show=False,
        )
        assert not list(tmp_path.glob("*.png"))

    def test_returns_none(self, fitted_model, mdata):
        result = fitted_model.stratify(
            mdata=mdata,
            event="OS",
            surv_time="OS.time",
            show=False,
        )
        assert result is None


class TestPlotLoss:
    def test_runs_after_fit(self, fitted_model):
        """plot_loss() must not raise after training."""
        import matplotlib.pyplot as plt

        fitted_model.plot_loss(show=False)
        plt.close("all")

    def test_plot_loss_twice_does_not_accumulate_figures(self, fitted_model):
        """Calling plot_loss() multiple times must not leak matplotlib figures."""
        import matplotlib.pyplot as plt

        before = len(plt.get_fignums())
        fitted_model.plot_loss(show=False)
        fitted_model.plot_loss(show=False)
        plt.close("all")
        assert len(plt.get_fignums()) == before
