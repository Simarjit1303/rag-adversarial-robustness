from datetime import datetime, timedelta, timezone
from scripts.clean_scratch_volume import _human_size, _is_never_delete, find_candidates

class _FakePaginator:

    def __init__(self, contents):
        self._contents = contents

    def paginate(self, Bucket, Prefix):
        yield {'Contents': [o for o in self._contents if o['Key'].startswith(Prefix)]}

class _FakeS3:

    def __init__(self, contents):
        self._contents = contents

    def get_paginator(self, name):
        assert name == 'list_objects_v2'
        return _FakePaginator(self._contents)

def _obj(key, age_hours, size=100):
    return {'Key': key, 'Size': size, 'LastModified': datetime.now(timezone.utc) - timedelta(hours=age_hours)}

def test_old_tmp_file_is_a_candidate():
    s3 = _FakeS3([_obj('results/.tmp_abc123', age_hours=48)])
    candidates = list(find_candidates(s3, 'bucket', 'results/', min_age_hours=24))
    assert [c[0] for c in candidates] == ['results/.tmp_abc123']

def test_young_tmp_file_is_not_a_candidate():
    s3 = _FakeS3([_obj('results/.tmp_abc123', age_hours=1)])
    candidates = list(find_candidates(s3, 'bucket', 'results/', min_age_hours=24))
    assert candidates == []

def test_real_result_files_are_never_candidates_regardless_of_age():
    s3 = _FakeS3([_obj('results/baseline_raw_phi-4-mini_hotpot_qa_vllm.jsonl', age_hours=999), _obj('results/attack_summary_ministral-3-8b_ms_marco_ignore_vllm.csv', age_hours=999), _obj('phase1_results_complete/baseline_raw_llama-3.1-8b_ms_marco_vllm.jsonl', age_hours=999)])
    candidates = list(find_candidates(s3, 'bucket', '', min_age_hours=24))
    assert candidates == []

def test_never_delete_safety_net_catches_a_tmp_prefixed_protected_name():
    assert _is_never_delete('results/.tmp_notes.md') is True
    s3 = _FakeS3([_obj('results/.tmp_notes.md', age_hours=999)])
    candidates = list(find_candidates(s3, 'bucket', 'results/', min_age_hours=24))
    assert candidates == []

def test_human_size_formatting():
    assert _human_size(500) == '500B'
    assert _human_size(2048) == '2KB'
    assert _human_size(5 * 1024 * 1024) == '5MB'
if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
