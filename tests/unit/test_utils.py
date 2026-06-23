"""Unit tests for customics.utils."""

from customics import get_common_samples, get_sub_mudata, prepare_input


class TestGetSubOmicsDf:
    def test_subsets_all_modalities(self, mdata, sample_ids):
        keep = sample_ids[:5]
        sub = get_sub_mudata(mdata, keep)
        assert list(sub.mod) == list(mdata.mod)
        for name in mdata.mod:
            assert list(sub[name].obs_names) == keep
        assert list(sub.obs_names) == keep

    def test_carries_obs_and_uns(self, mdata, sample_ids):
        prepare_input(mdata=mdata, label="label", event="OS", surv_time="OS.time")
        sub = get_sub_mudata(mdata, sample_ids[:5])
        assert sub.uns == mdata.uns
        assert "label" in sub.obs.columns

    def test_result_is_usable_for_common_samples(self, mdata, sample_ids):
        keep = sample_ids[:5]
        sub = get_sub_mudata(mdata, keep)
        assert get_common_samples(sub) == sorted(keep)

    def test_ignores_unknown_samples(self, mdata, sample_ids):
        sub = get_sub_mudata(mdata, [*sample_ids[:3], "NOT_A_SAMPLE"])
        assert list(sub.obs_names) == sample_ids[:3]
