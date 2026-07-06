import email
import importlib.metadata
import importlib.util
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

import spantrack
from spantrack import cli

_PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

# Building an uploadable distribution needs the release toolchain; `--no-isolation`
# builds against hatchling in the current environment rather than downloading it.
_RELEASE_TOOLS = ("build", "twine", "hatchling")
_HAVE_RELEASE_TOOLS = all(
    importlib.util.find_spec(name) is not None for name in _RELEASE_TOOLS
)


def _project_table() -> dict[str, Any]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]


def test_version_is_non_empty_string():
    assert isinstance(spantrack.__version__, str)
    assert spantrack.__version__


def test_console_script_entry_point_resolves():
    entry_points = importlib.metadata.entry_points(group="console_scripts")
    resolved = {ep.name: ep.value for ep in entry_points}

    assert resolved.get("spantrack") == "spantrack.cli:main"


def test_dependencies_carry_no_pypi_rejected_direct_url():
    # PyPI refuses a distribution whose metadata carries a PEP 508 direct reference
    # (`name @ url`), so the runtime dependencies must not declare one.
    dependencies = cast("list[str]", _project_table()["dependencies"])
    offenders = [dep for dep in dependencies if "@" in dep]

    assert offenders == [], f"direct-URL deps block a PyPI upload: {offenders}"


def test_project_metadata_is_release_complete():
    project = _project_table()

    assert project["license"]
    assert project["classifiers"]
    assert project["urls"]


def test_readme_pointer_resolves_to_a_nonempty_file():
    readme = _project_table()["readme"]
    assert isinstance(readme, str)
    path = _PYPROJECT.parent / readme

    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip()


def test_readme_documents_every_cli_flag():
    # The README's command-line reference must list every long flag, so it cannot drift
    # out of date as flags are added.
    readme = (_PYPROJECT.parent / "README.md").read_text(encoding="utf-8")
    flags = {
        option
        for action in cli.build_parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    missing = sorted(flag for flag in flags if flag not in readme)

    assert missing == [], f"README is missing CLI flags: {missing}"


@pytest.mark.skipif(
    not _HAVE_RELEASE_TOOLS,
    reason="release build needs build + twine + hatchling",
)
def test_build_produces_an_uploadable_distribution(tmp_path: Path):
    # The whole release gate: the project builds an sdist + wheel, twine accepts them,
    # and the wheel metadata is upload-clean (no direct URL, a real long description).
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
            str(_PYPROJECT.parent),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = list(tmp_path.glob("*.whl"))
    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1

    checked = subprocess.run(
        [sys.executable, "-m", "twine", "check", *map(str, wheels + sdists)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "PASSED" in checked.stdout

    with zipfile.ZipFile(wheels[0]) as archive:
        meta_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(meta_name))

    requires = metadata.get_all("Requires-Dist") or []
    assert requires  # the runtime dependencies survive into the built metadata
    assert not any("@" in dep for dep in requires)  # no PyPI-rejected direct reference

    # the README is the long description
    assert metadata.get("Description-Content-Type")
    assert cast("str", metadata.get_payload()).strip()
