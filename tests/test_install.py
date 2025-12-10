from importlib import import_module


def test_package_version_accessible():
    # Ensure the installed package exposes a version string
    mod = import_module("chatter")
    assert hasattr(mod, "__version__")
    assert isinstance(mod.__version__, str)
