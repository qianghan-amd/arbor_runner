"""scripts/find_ci_runs.py: which CI runs this site can run, and the CI-name prefix map (offline)."""
import pytest

from arbor_ab import config

import find_ci_runs


def test_prefixes_cover_list_prefixes():
    pre = find_ci_runs.prefixes(config.load_matrix())
    assert pre["dsv41flash"] == "dsv41flash" and pre["glm5.2"] == "glm52"
    assert pre["kimik31pro"] == "kimik31pro" and pre["kimik3.1"] == "kimik31pro"     # a list prefix (Kimi K3.1 Pro)


def ci(hw="cluster:mi355x-amds", framework="vllm", served="deepseek-ai/DeepSeek-V4.1-Flash", model="dsv41flash"):
    return {"hw": hw, "framework": framework, "served": served, "cells": [(model, 1)]}


@pytest.mark.parametrize("c,want", [
    (ci(), "yes"),
    (ci(hw="cluster:mi325x-amd"), "no: cluster:mi325x-amd is not gfx950"),
    (ci(hw="cluster:gb200-nv"), "no: cluster:gb200-nv is not gfx950"),
    (ci(framework="atom"), "no: framework atom"),
    (ci(framework="tokenspeed"), "no: framework tokenspeed"),
    (ci(model="glm52", framework="sglang", served="amd/GLM-5.3-Quark-MXFP4-AttnFP8"),
     "other model: serves amd/GLM-5.3-Quark-MXFP4-AttnFP8 (--allow-other-model)"),
    (ci(model="glm52", framework="sglang", served="amd/GLM-5.2-MXFP4"), "yes"),
])
def test_verdict(c, want):
    assert find_ci_runs.verdict(c, "gfx950") == want
