import importlib.metadata

import spantrack


def test_version_is_non_empty_string():
    assert isinstance(spantrack.__version__, str)
    assert spantrack.__version__


def test_console_script_entry_point_resolves():
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    resolved = {ep.name: ep.value for ep in entry_points}

    assert resolved.get("spantrack") == "spantrack.cli:main"
