"""Repair mojibake and canonicalize TTS-input text without touching the DOM.

`normalize` maps a sentence's display text to the text handed to the TTS engine.
The passes are ordered and the order is load-bearing:

1. ftfy `fix_text` repairs genuine mojibake. It runs first because U+00A0 (no-break
   space) and U+00AD (soft hyphen) are UTF-8 continuation bytes of common accented
   letters (a-grave is c3 a0, i-acute is c3 ad); touching them first would sever the
   byte pairs the encoding repair depends on.
2. Flatten no-break spaces to plain spaces. ftfy leaves a real U+00A0 untouched, so
   without this a no-break space reaches the synthesizer verbatim.
3. Strip soft-hyphen and zero-width marks (print hyphenation and invisible joiners)
   that would otherwise perturb pronunciation or the cache key.
4. Re-apply NFC. ftfy normalizes to NFC internally, but a zero-width mark between a
   base letter and a combining mark blocks composition; once step 3 removes it the
   now-adjacent pair must recompose so the output stays canonical.
5. Strip lone [N] footnote citations, repairing the spacing they leave behind.
6. Speak bare URLs and emails (their internals are dropped, so this runs before the
   roman pass, which would otherwise expand a numeral in a URL path).
7. Expand roman-numeral labels and headings (Chapter IV -> Chapter 4) to arabic digits.
8. Turn em-dashes into comma pauses (a digit range into "to") and canonicalize
   ellipses; last, so it also normalizes the dash a roman range leaves behind.

The config repairs mojibake without damaging correct text: HTML entities stay literal,
curly quotes stay curly, full-width forms and line breaks are preserved.
"""

import re
import unicodedata

import ftfy

__all__ = ["collapse_whitespace", "normalize"]

_NBSP = 0x00A0

# Invisible marks that must not reach the TTS engine: soft hyphen (print hyphenation)
# plus zero-width space, ZWNJ, ZWJ, and word joiner. None => translate deletes them.
_INVISIBLE_ORDS = (0x00AD, 0x200B, 0x200C, 0x200D, 0x2060)
_INVISIBLE_MARKS = str.maketrans("", "", "".join(map(chr, _INVISIBLE_ORDS)))

# Repair genuine mojibake (ftfy's default fix_encoding); leave correct text alone.
# NFC canonicalization stabilizes the synthesis cache key across publisher encodings.
_FIXER_CONFIG = ftfy.TextFixerConfig(
    unescape_html=False,  # keep entities literal (&lt; / &amp;), never decode
    uncurl_quotes=False,  # keep curly quotes curly
    fix_character_width=False,  # keep full-width forms (NFC, not NFKC)
    fix_line_breaks=False,  # keep the author's line breaks
    normalization="NFC",
    explain=False,
)

# Expand a roman numeral only next to a structural label (or as a whole-string heading),
# and emit arabic digits. The window is deliberately narrow: a >=2-character numeral
# after a label leaves the pronoun "I" and single-letter labels (Part C, Appendix D)
# alone; a strict canonical-form regex rejects loose spellings (IIII, VV, XM) and a
# small blocklist rejects valid romans that are English words.
_LABEL = (
    r"(?:[Cc]hapters?|[Pp]arts?|[Bb]ooks?|[Ss]ections?|[Aa]cts?|[Ss]cenes?"
    r"|[Vv]olumes?|[Vv]ols?\.?|[Cc]antos?|[Aa]ppendix|[Aa]ppendices"
    r"|[Aa]rticles?|[Tt]itles?|[Cc]hs?\.|[Bb]ks?\.|[Pp]ts?\.|[Ss]ecs?\.|[Aa]pps?\.)"
)
_ROMAN_CHARS = "[MDCLXVImdclxvi]"
# _ROMAN_STRICT accepts only the one canonical spelling of each value (loose forms like
# IIII/VV/XM fail it), _ROMAN_VALUES converts a validated numeral to its integer, and
# _ROMAN_WORDS blocklists romans that are ordinary words.
_ROMAN_STRICT = re.compile(
    r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# Canonical romans that are also common English words or codes -- never expanded.
_ROMAN_WORDS = frozenset(
    {"MIX", "DID", "DIM", "MID", "LID", "CIVIC", "MIC", "DI", "CI", "MI", "MM"}
)


def _roman_to_int(numeral: str) -> int:
    # `numeral` arrives uppercased and spelling-validated from `_roman_to_arabic`.
    total = previous = 0
    for char in reversed(numeral):
        value = _ROMAN_VALUES[char]
        total += -value if value < previous else value

        previous = value

    return total


def _roman_to_arabic(numeral: str) -> str | None:
    """Return the arabic form of a canonical, non-word roman numeral, else None.

    `_ROMAN_STRICT` matches only the single canonical spelling of each value, so a loose
    form (IIII, VV, XM) fails the guard outright -- no separate round-trip check needed.
    """
    upper = numeral.upper()
    if upper in _ROMAN_WORDS or not _ROMAN_STRICT.fullmatch(upper):
        return None

    return str(_roman_to_int(upper))


_ROMAN_HEADING = re.compile(r"^(" + _ROMAN_CHARS + r"{2,})(\.\s|\s*$)")
_ROMAN_RANGE = re.compile(
    r"\b("
    + _LABEL
    + r")(\.?\s+)("
    + _ROMAN_CHARS
    + r"+)([-\u2013\u2014])("
    + _ROMAN_CHARS
    + r"+)\b"
)
_ROMAN_LABELLED = re.compile(
    r"\b(" + _LABEL + r")(\.?\s+)(" + _ROMAN_CHARS + r"{2,})\b"
)


def _expand_heading(match: re.Match[str]) -> str:
    arabic = _roman_to_arabic(match.group(1))
    return arabic + match.group(2) if arabic else match.group(0)


def _expand_range(match: re.Match[str]) -> str:
    label, separator, first, dash, second = match.groups()
    left, right = _roman_to_arabic(first), _roman_to_arabic(second)
    if left and right:
        return label + separator + left + dash + right

    return match.group(0)


def _expand_labelled(match: re.Match[str]) -> str:
    label, separator, numeral = match.groups()
    arabic = _roman_to_arabic(numeral)
    return label + separator + arabic if arabic else match.group(0)


def _expand_roman_numerals(text: str) -> str:
    """Expand label-adjacent and bare-heading roman numerals to arabic digits."""
    text = _ROMAN_HEADING.sub(_expand_heading, text)
    text = _ROMAN_RANGE.sub(_expand_range, text)
    return _ROMAN_LABELLED.sub(_expand_labelled, text)


# Remove only unambiguous [N] citation brackets (1-3 digits, non-zero-lead, no ranges).
# Superscript digits and "(N)" are left untouched: with no DOM context they cannot be
# told apart from math exponents (cm2, E=mc2) and prose enumerations, and mangling those
# is worse than narrating a stray footnote number. Real noterefs are dropped upstream.
_LEADING_CITATION = re.compile(r"^\s*\[[1-9]\d{0,2}\]\s*")
_CITATION = re.compile(r"\s*\[[1-9]\d{0,2}\]")


def _strip_footnote_markers(text: str) -> str:
    """Remove lone [N] citations, repairing spacing only when one was removed."""
    text, leading = _LEADING_CITATION.subn("", text)
    text, others = _CITATION.subn("", text)
    if leading or others:
        text = re.sub(r"  +", " ", text)
        text = re.sub(r" +([,.;:])", r"\1", text)

    return text


# --- URL and email speaking
# Speak bare URLs and emails so the engine does not spell them out. A URL becomes its
# host, " dot "-joined (scheme/userinfo/www/path/query/port dropped, IPv4 kept literal);
# an email becomes "local at domain" (local dots -> spaces, domain -> " dot "). Matching
# is conservative: a scheme-less host must end in a common TLD, so "e.g." / "file.txt" /
# "3.14" / "read.me" stay literal. The STOP class stops a URL token at a quote, bracket,
# dash, or ellipsis, so it never swallows following prose or a sentence terminator.
_URL_STOP = r"[^\s<>()\[\]{}\"'\u2018\u2019\u201c\u201d\u2013\u2014\u2015\u2026]"
_URL_TLD = "com|org|net|edu|gov|mil|int"
_URL_RE = re.compile(
    r"(?:https?|ftp)://" + _URL_STOP + r"+"
    r"|www\.[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+|/)" + _URL_STOP + r"*"
    r"|\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+(?:"
    + _URL_TLD
    + r")\b(?::\d+)?(?:/"
    + _URL_STOP
    + r"*)?",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(
    r"(?<![:/\w.+%-])(?:mailto:)?"
    r"[A-Za-z0-9](?:[A-Za-z0-9._%+-]*[A-Za-z0-9])?@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
)
_TRAILING_PUNCT = ".,;:!?)]}\"'\u2018\u2019\u201c\u201d"
_IPV4 = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")


def _peel_trailing(token: str) -> tuple[str, str]:
    """Split a run of trailing sentence punctuation off a token, to be re-emitted."""
    trail = ""
    while token and token[-1] in _TRAILING_PUNCT:
        trail = token[-1] + trail
        token = token[:-1]

    return token, trail


def _speak_host(url: str) -> str:
    host = re.sub(r"(?i)^(?:https?|ftp)://", "", url)
    host = re.sub(r"^[^@/]*@", "", host)  # drop userinfo (no secret leaks to speech)
    host = re.split(r"[/?#]", host, maxsplit=1)[0]
    host = re.sub(r"(?i)^www\.", "", host)
    host = host.split(":", 1)[0]  # drop port

    if _IPV4.fullmatch(host):  # leave a dotted IPv4 literal for misaki to speak
        return host

    return " dot ".join(part for part in host.split(".") if part)


def _speak_url(match: re.Match[str]) -> str:
    token, trail = _peel_trailing(match.group(0))
    return _speak_host(token) + trail


def _speak_email(match: re.Match[str]) -> str:
    token = match.group(0)
    if token.lower().startswith("mailto:"):
        token = token[7:]

    token, trail = _peel_trailing(token)
    local, _, domain = token.partition("@")

    return local.replace(".", " ") + " at " + domain.replace(".", " dot ") + trail


def _speak_web_references(text: str) -> str:
    """Speak emails first (each embeds a host the URL rule would eat), then URLs."""
    text = _EMAIL_RE.sub(_speak_email, text)
    return _URL_RE.sub(_speak_url, text)


# --- Dash and ellipsis normalization
# em-dash (U+2014), horizontal bar (U+2015), and 2+ hyphens become a comma pause; a
# single hyphen and the en-dash (U+2013, a range mark) are left. A dash between digits
# becomes " to " (a range). Ellipses (U+2026, "...", ". . .") canonicalize to "...".
_DOT_RUN = re.compile(r"\.(?:[ \t]*\.){2,}")
_EM_DASH_DIGITS = re.compile(r"(\d)\s*(?:\u2014|\u2015|--+)\s*(\d)")
_EM_DASH = re.compile(r"\s*(?:\u2014|\u2015|--+)\s*")
_COMMA_BEFORE_PUNCT = re.compile(r",\s*([.,;:!?)\]])")
_LEADING_COMMA = re.compile(r"^\s*,\s*")
_TRAILING_COMMA = re.compile(r",\s*$")
_DOUBLED_COMMA = re.compile(r",(?:\s*,)+")


def _normalize_dashes(text: str) -> str:
    """Make em-dashes comma pauses (digit ranges ' to ') and canonicalize ellipses."""
    text = text.replace("\u2026", "...")
    text = _DOT_RUN.sub("...", text)
    text = _EM_DASH_DIGITS.sub(r"\1 to \2", text)
    text, converted = _EM_DASH.subn(", ", text)

    if converted:  # repair comma artifacts only where a dash actually became a comma
        text = _COMMA_BEFORE_PUNCT.sub(r"\1", text)
        text = _LEADING_COMMA.sub("", text)
        text = _TRAILING_COMMA.sub("", text)
        text = _DOUBLED_COMMA.sub(",", text)

    return text


def normalize(text: str) -> str:
    """Normalize a sentence's display text for TTS input; the source DOM is untouched.

    Repairs mojibake, flattens no-break spaces, strips soft-hyphen/zero-width marks and
    [N] footnotes, speaks URLs/emails, expands roman-numeral labels, and turns em-dashes
    and ellipses into speakable pauses -- returning NFC-normalized text.

    Args:
        text: One sentence's display text, as extracted from the content document.

    Returns:
        The normalized text handed to the TTS backend.
    """
    text = ftfy.fix_text(text, config=_FIXER_CONFIG)
    text = text.replace(chr(_NBSP), " ")
    text = text.translate(_INVISIBLE_MARKS)
    text = unicodedata.normalize("NFC", text)
    text = _strip_footnote_markers(text)
    text = _speak_web_references(text)
    text = _expand_roman_numerals(text)

    return _normalize_dashes(text)


# --- Whitespace collapsing
# A standalone primitive, separate from `normalize` (which preserves the author's
# whitespace): collapse runs of whitespace for display text and character estimates.
_WHITESPACE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    """Collapse each run of whitespace to a single space and strip the ends."""
    return _WHITESPACE.sub(" ", text).strip()
