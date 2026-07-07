"""Tests for the backend-neutral pronunciation-lexicon loader."""

from pathlib import Path

import pytest

from spantrack.lexicon import Lexicon, LexiconError, load


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "lexicon.toml"
    path.write_text(content, encoding="utf-8")

    return path


def test_load_returns_case_insensitive_lexicon(tmp_path: Path):
    path = _write(tmp_path, '[words]\ncafe = "ka-FAY"\nGIF = "JIF"\n')
    lexicon = load(path)

    assert isinstance(lexicon, Lexicon)
    assert lexicon.get("cafe") == "ka-FAY"
    assert lexicon.get("CAFE") == "ka-FAY"
    assert lexicon.get("Gif") == "JIF"
    assert lexicon.get("missing") is None


def test_load_empty_file_is_empty_lexicon(tmp_path: Path):
    assert load(_write(tmp_path, "")).get("anything") is None


def test_load_empty_words_table_is_empty_lexicon(tmp_path: Path):
    assert load(_write(tmp_path, "[words]\n")).get("anything") is None


def test_malformed_toml_raises(tmp_path: Path):
    with pytest.raises(LexiconError):
        load(_write(tmp_path, "[words\nbroken"))


def test_words_not_a_table_raises(tmp_path: Path):
    with pytest.raises(LexiconError, match="table"):
        load(_write(tmp_path, 'words = "not a table"\n'))


def test_unexpected_top_level_key_raises(tmp_path: Path):
    # A "[word]" typo would otherwise silently drop every entry.
    with pytest.raises(LexiconError, match="unexpected"):
        load(_write(tmp_path, '[word]\ncafe = "x"\n'))


def test_non_string_value_raises(tmp_path: Path):
    with pytest.raises(LexiconError, match="string"):
        load(_write(tmp_path, "[words]\ncafe = 123\n"))


def test_nested_table_value_raises(tmp_path: Path):
    with pytest.raises(LexiconError, match="string"):
        load(_write(tmp_path, '[words.cafe]\nipa = "x"\n'))


def test_case_collision_raises(tmp_path: Path):
    with pytest.raises(LexiconError, match="duplicate"):
        load(_write(tmp_path, '[words]\ncafe = "a"\nCafe = "b"\n'))


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(LexiconError):
        load(tmp_path / "does-not-exist.toml")


def test_entries_exposes_case_folded_keys(tmp_path: Path):
    lexicon = load(_write(tmp_path, '[words]\nGiF = "JIF"\n'))
    assert lexicon.entries == {"gif": "JIF"}


def test_get_case_folds_beyond_ascii(tmp_path: Path):
    # casefold (not str.lower) maps the German eszett to "ss", so an eszett-keyed entry
    # still matches its uppercased spelling; str.lower would leave the two unequal.
    eszett = chr(0x00DF)
    lexicon = load(_write(tmp_path, f'[words]\n"stra{eszett}e" = "SHTRAH-se"\n'))
    assert lexicon.get("STRASSE") == "SHTRAH-se"
