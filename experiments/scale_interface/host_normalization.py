"""Pinned 3DGraphLLM answer normalization for host-normalized exact EM1.

``clean_answer`` is extracted from CognitiveAISystems/3DGraphLLM utils/helper.py
at commit 19d65cd9e220ec8e7704acc989aaef5e367f31dc. The two non-raw regex
literals use raw prefixes here to avoid Python 3.13 escape warnings; the
function's parsed AST is identical to the pinned source. No rule is added.

Upstream attribution also credits LEO embodied-generalist data/data_utils.py
at 477dc44b8b18dbfbe6823c307436d896ec8b062e (lines 322--379).
Copyright (c) 2024 Chat-Scene. MIT License; see LICENSE.host in this directory.

This is the published cleaner under our frozen host/input and invalid-output
policy, not a claim of official SQA3D or leaderboard-equivalent evaluation.
Only nonempty exact equality is used by the caller; refined EM is excluded.
"""
import re


METRIC = "host_normalized_em1_3dgraphllm_19d65cd9_v1"
NORMALIZATION = "pinned 3DGraphLLM clean_answer on prediction and every reference; nonempty exact equality"


def provenance():
    """Return the source and fixed policy attached to this metric's outputs."""
    return {
        "metric": METRIC,
        "score_name": "host_normalized_exact_EM1",
        "source_repository": "CognitiveAISystems/3DGraphLLM",
        "source_commit": "19d65cd9e220ec8e7704acc989aaef5e367f31dc",
        "source_path": "utils/helper.py",
        "source_url": "https://github.com/CognitiveAISystems/3DGraphLLM/blob/19d65cd9e220ec8e7704acc989aaef5e367f31dc/utils/helper.py",
        "source_sha256": "4713ad135a7a3abd20771b007b193639f868b9e506b87edda8a9b34a571bcbd8",
        "function": "clean_answer",
        "function_ast_sha256": "c39a43084e3e42c2836457982d1bfade0e3b6890caf8c913ed8e711bc2a87836",
        "license": "MIT; Copyright (c) 2024 Chat-Scene; LICENSE.host",
        "normalization_applied_identically_to_prediction_and_all_references": True,
        "matching": "nonempty exact equality; no refined or substring matching",
        "invalid_output_policy": "invalid_object_id=true and blank outputs are errors",
        "denominator": "all questions in the externally fixed paired cohort; failures retained",
        "scope": "host-normalized exact EM1 under our host/input protocol; no leaderboard equivalence claimed",
        "revision_selection": "post hoc after identifying numeric-surface mismatch",
    }


# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379
def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub(r'\.[ ]{2,}', '. ', data)
    data = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data
