"""Unit tests for customics.utils."""

from customics import get_shared_samples, get_sub_mudata, prepare_input, split_mudata


class TestGetSubMuData:
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
        assert get_shared_samples(sub) == sorted(keep)

    def test_ignores_unknown_samples(self, mdata, sample_ids):
        sub = get_sub_mudata(mdata, [*sample_ids[:3], "NOT_A_SAMPLE"])
        assert list(sub.obs_names) == sample_ids[:3]


class TestSplitMuData:
    def test_partitions_shared_samples(self, mdata):
        train, val, test = split_mudata(mdata)

        train_ids = set(train.obs_names)
        val_ids = set(val.obs_names)
        test_ids = set(test.obs_names)

        # The three splits are disjoint and cover every shared sample.
        assert train_ids | val_ids | test_ids == set(get_shared_samples(mdata))
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_carries_uns(self, mdata):
        prepare_input(mdata=mdata, label="label", event="OS", surv_time="OS.time")
        for sub in split_mudata(mdata):
            assert sub.uns == mdata.uns

    def test_is_reproducible(self, mdata):
        first = [list(sub.obs_names) for sub in split_mudata(mdata, random_state=0)]
        second = [list(sub.obs_names) for sub in split_mudata(mdata, random_state=0)]
        assert first == second
