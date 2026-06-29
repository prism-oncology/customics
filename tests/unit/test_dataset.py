"""Unit tests for MultiOmicsDataset."""

import pytest
import torch

from customics import prepare_input
from customics.datasets import MultiOmicsDataset


@pytest.fixture
def prepared_mdata(mdata):
    """mdata with clinical columns registered in uns (required by the dataset)."""
    prepare_input(mdata=mdata, label="label", event="OS", surv_time="OS.time")
    return mdata


class TestMultiOmicsDataset:
    def test_len(self, prepared_mdata, clinical_df, sample_ids):
        ds = MultiOmicsDataset(prepared_mdata, sample_ids, clinical_df["label"])
        assert len(ds) == len(sample_ids)

    def test_getitem_shapes(self, prepared_mdata, clinical_df, sample_ids):
        ds = MultiOmicsDataset(prepared_mdata, sample_ids, clinical_df["label"])
        omics_tensors, _, os_time, os_event = ds[0]
        assert len(omics_tensors) == 2
        assert omics_tensors[0].shape == (50,)  # rna
        assert omics_tensors[1].shape == (30,)  # cnv
        assert isinstance(os_time, int)
        assert isinstance(os_event, int)

    def test_getitem_event_is_binary(self, prepared_mdata, clinical_df, sample_ids):
        # Guards against reading os_event from the survival-time column.
        ds = MultiOmicsDataset(prepared_mdata, sample_ids, clinical_df["label"])
        for i in range(len(ds)):
            _, _, _, os_event = ds[i]
            assert os_event in (0, 1)

    def test_getitem_dtype(self, prepared_mdata, clinical_df, sample_ids):
        ds = MultiOmicsDataset(prepared_mdata, sample_ids, clinical_df["label"])
        omics_tensors, _, _, _ = ds[0]
        for t in omics_tensors:
            assert t.dtype == torch.float32

    def test_no_label(self, prepared_mdata, sample_ids):
        ds = MultiOmicsDataset(prepared_mdata, sample_ids, None)
        _, label, _, _ = ds[0]
        assert label == 0

    def test_get_samples(self, prepared_mdata, clinical_df, sample_ids):
        ds = MultiOmicsDataset(prepared_mdata, sample_ids, clinical_df["label"])
        assert ds.get_samples() == sample_ids
