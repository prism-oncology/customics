"""Unit tests for MultiOmicsDataset."""

import torch

from customics.datasets import MultiOmicsDataset


class TestMultiOmicsDataset:
    def test_len(self, mu_data, clinical_df, sample_ids):
        ds = MultiOmicsDataset(mu_data, sample_ids, clinical_df["label"])
        assert len(ds) == len(sample_ids)

    def test_getitem_shapes(self, mu_data, clinical_df, sample_ids):
        ds = MultiOmicsDataset(mu_data, sample_ids, clinical_df["label"])
        omics_tensors, _, os_time, os_event = ds[0]
        assert len(omics_tensors) == 2
        assert omics_tensors[0].shape == (50,)  # rna
        assert omics_tensors[1].shape == (30,)  # cnv
        assert isinstance(os_time, int)
        assert isinstance(os_event, int)

    def test_getitem_dtype(self, mu_data, clinical_df, sample_ids):
        ds = MultiOmicsDataset(mu_data, sample_ids, clinical_df["label"])
        omics_tensors, _, _, _ = ds[0]
        for t in omics_tensors:
            assert t.dtype == torch.float32

    def test_no_label(self, mu_data, clinical_df, sample_ids):
        ds = MultiOmicsDataset(mu_data, sample_ids, None)
        _, lbl, _, _ = ds[0]
        assert lbl == 0

    def test_get_samples(self, mu_data, clinical_df, sample_ids):
        ds = MultiOmicsDataset(mu_data, sample_ids, clinical_df["label"])
        assert ds.get_samples() == sample_ids
