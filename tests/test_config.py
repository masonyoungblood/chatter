import pytest
from chatter.config import make_config

def test_make_config_warns_on_unknown():
    # Unknown keys should trigger a warning but still return a config
    with pytest.warns(UserWarning):
        cfg = make_config({"unknown_key": 123})
    assert "sr" in cfg

def test_make_config_allows_overrides():
    # User overrides should take effect
    cfg = make_config({"sr": 16000})
    assert cfg["sr"] == 16000

