"""Tests for the ftfy TTS-input normalization spine (`text_normalization.normalize`).

`CASES` is an empirically-derived adversarial corpus: `repair` rows are mojibake that
must be fixed, `survive` rows are correct text that must pass through unchanged,
`transform` rows are desired NFC/zero-width changes, and `ordering` rows prove the pass
order is load-bearing. Invisible code points are written as `\\uXXXX` escapes so every
row is auditable; printable non-ASCII is left readable.
"""

import unicodedata

import pytest

from spantrack.text_normalization import normalize


def moji(correct: str) -> str:
    """Return `correct` mojibaked: its UTF-8 bytes mis-decoded as latin-1."""
    return correct.encode("utf-8").decode("latin-1")


# (name, intent, raw, expected): every row validated against the pipeline.
# fmt: off
CASES = [
    ("repair-fr-cafe-acute", "repair", moji("café"), "café"),
    ("repair-fr-final-a-grave-nbsp-byte", "repair", moji("déjà"), "déjà"),
    ("repair-de-eszett", "repair", moji("Straße"), "Straße"),
    ("repair-es-enye", "repair", moji("niño"), "niño"),
    ("repair-latin-extended-lodz", "repair", moji("Łódź"), "Łódź"),
    ("repair-nordic-slash-ring-ae", "repair", moji("søndag år ære"), "søndag år ære"),
    ("repair-right-single-quote", "repair", moji("It’s"), "It’s"),
    ("repair-double-quoted-phrase", "repair", moji("“Hello, world.”"), "“Hello, world.”"),
    ("repair-em-dash-vs-en-dash", "repair", moji("pp. 12–34 wait—stop"), "pp. 12–34 wait, stop"),
    ("repair-ellipsis", "repair", moji("Well…"), "Well..."),
    ("repair-prime-and-degree", "repair", moji("40°26′46″N"), "40°26′46″N"),
    ("repair-guillemets", "repair", moji("«bonjour»"), "«bonjour»"),
    ("repair-mixed-typography-sentence", "repair", moji("“It’s here—now…”"), "“It’s here, now...”"),
    ("repair-moji-glued-to-ampersand", "repair", moji("Beyoncé") + "&Jay", "Beyoncé&Jay"),
    ("repair-moji-inside-angle-brackets", "repair", "x<" + moji("naïve") + ">y", "x<naïve>y"),
    ("repair-moji-beside-html-entity", "repair", moji("café") + " &amp; more", "café &amp; more"),
    ("survive-precomposed-accents", "survive", "résumé naïve", "résumé naïve"),
    ("survive-curly-quotes", "survive", "“Hello,” she said. ‘hi’ ’90s ’Twas", "“Hello,” she said. ‘hi’ ’90s ’Twas"),
    ("survive-german-low-quotes", "survive", "„Guten Tag“", "„Guten Tag“"),
    ("survive-prime-marks", "survive", "5′ 10″ tall", "5′ 10″ tall"),
    ("survive-dash-family-and-minus", "survive", "Wait—no; pages 12–18; −5 degrees", "Wait, no; pages 12–18; −5 degrees"),
    ("survive-real-ellipsis", "survive", "And then… nothing.", "And then... nothing."),
    ("survive-narrow-nbsp", "survive", "12\u202f345 items", "12\u202f345 items"),
    ("survive-html-entities", "survive", "a &lt;b&gt; c; Jack &amp; Jill; It&#8217;s; &quot;&#x2019;&quot;", "a &lt;b&gt; c; Jack &amp; Jill; It&#8217;s; &quot;&#x2019;&quot;"),
    ("survive-double-escaped-entity", "survive", "&amp;lt;", "&amp;lt;"),
    ("survive-literal-nbsp-entity", "survive", "a&nbsp;b", "a&nbsp;b"),
    ("survive-bare-ampersands-and-angles", "survive", "AT&T, R&D; if x < y and y > z", "AT&T, R&D; if x < y and y > z"),
    ("survive-url-percent-and-query", "survive", "See https://a.com/x?q=1&r=2#/%E2%9C%93", "See a dot com"),
    ("survive-code-generics-and-backslash-path", "survive", r"List<Map<String,int>> C:\Users\x\f.txt", r"List<Map<String,int>> C:\Users\x\f.txt"),
    ("survive-cjk-han-kana", "survive", "日本語 中文 한국어", "日本語 中文 한국어"),
    ("survive-fullwidth-and-halfwidth", "survive", "ＡＢＣ１２３ ｶﾀｶﾅ", "ＡＢＣ１２３ ｶﾀｶﾅ"),
    ("survive-greek-cyrillic-arabic", "survive", "Ελληνικά Русский العربية", "Ελληνικά Русский العربية"),
    ("survive-currency-math-half", "survive", "€ £ ¥ ± × ÷ ½", "€ £ ¥ ± × ÷ ½"),
    ("survive-basic-and-skintone-emoji", "survive", "\U0001f600 \U0001f44d\U0001f3fd \U0001f1ef\U0001f1f5", "\U0001f600 \U0001f44d\U0001f3fd \U0001f1ef\U0001f1f5"),
    ("transform-nbsp-to-space", "transform", "10\u00a0km on\u00a0Monday", "10 km on Monday"),
    ("transform-nfc-recompose-word", "transform", "cafe\u0301 man\u0303ana re\u0301sume\u0301", "café mañana résumé"),
    ("transform-nfc-hangul-jamo", "transform", "\u1112\u1161\u11ab", "한"),
    ("transform-nfc-vietnamese-stacked", "transform", "e\u0302\u0301", "ế"),
    ("survive-nfc-no-precomposed-form", "survive", "t\u0301", "t\u0301"),
    ("transform-soft-hyphen-strip", "transform", "hy\u00adphen\u00adation encyclo\u00ad\npedia", "hyphenation encyclo\npedia"),
    ("transform-zero-width-all-five", "transform", "a\u200bb\u200cc\u200dd\u2060e\u00adf", "abcdef"),
    ("transform-zwsp-after-mojibake-repair", "transform", moji("It’s") + "\u200b", "It’s"),
    ("ordering-nbsp-mojibake-continuation-byte", "ordering", moji("M. Dupont"), "M. Dupont"),
    ("ordering-nbsp-mojibake-adjacent-accent", "ordering", moji("à la carte"), "à la carte"),
    ("ordering-zw-strip-after-i-acute", "ordering", moji("Martín"), "Martín"),
    ("ordering-zw-strip-after-adios-control", "ordering", moji("adiós"), "adiós"),
    ("ordering-zwsp-multibyte-mojibake-strip-after", "ordering", moji("Noël\u200bmarket"), "Noëlmarket"),
    ("ordering-literal-shy-order-independent", "ordering", "cafe\u00adteria", "cafeteria"),
    ("transform-nfc-zerowidth-between-base-and-mark", "transform", "e\u200b\u0301", "é"),
    ("transform-nfc-softhyphen-between-base-and-mark", "transform", "e\u00ad\u0301", "é"),
    ("transform-zwj-emoji-splits", "transform", "\U0001f468\u200d\U0001f469", "\U0001f468\U0001f469"),
]
# fmt: on

_IDS = [row[0] for row in CASES]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(row[2], row[3]) for row in CASES],
    ids=_IDS,
)
def test_normalize_matches_corpus(raw: str, expected: str):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", [row[2] for row in CASES], ids=_IDS)
def test_normalize_output_is_nfc(raw: str):
    out = normalize(raw)
    assert out == unicodedata.normalize("NFC", out)


@pytest.mark.parametrize("raw", [row[2] for row in CASES], ids=_IDS)
def test_normalize_is_idempotent(raw: str):
    once = normalize(raw)
    assert normalize(once) == once


def test_corpus_has_expected_size():
    # Guards against a row being dropped when the table is edited.
    assert len(CASES) == 51


def test_pass_order_is_load_bearing():
    """The spine must repair mojibake BEFORE the flatten/strip passes run.

    U+00A0 and U+00AD are UTF-8 continuation bytes (a-grave c3 a0, i-acute c3 ad).
    Flattening or stripping them before ftfy would sever the byte pairs it needs,
    so reordering the passes breaks these witnesses.
    """
    assert normalize(moji("Martín")) == "Martín"
    assert normalize(moji("déjà")) == "déjà"
    assert normalize(moji("M. Dupont")) == "M. Dupont"


def test_normalize_empty_string():
    assert normalize("") == ""


def test_normalize_whitespace_only_is_preserved():
    # normalize does not trim; the segmenter owns sentence boundaries.
    assert normalize("  \n\t ") == "  \n\t "


def test_normalize_all_invisible_collapses_to_empty():
    assert normalize("\u200b\u2060\u00ad\u200c\u200d") == ""


def test_normalize_preserves_crlf():
    # fix_line_breaks=False keeps the author's newlines.
    assert normalize("a\r\nb\nc") == "a\r\nb\nc"


def test_normalize_handles_paragraph_length_input():
    para = (moji("café") + " ") * 500
    assert normalize(para) == "café " * 500


# roman-numeral expansion + footnote-[N] stripping
# fmt: off
ROMAN_FOOTNOTE_CASES = [
    ("chapter_iv", "roman-expand", "Chapter IV", "Chapter 4"),
    ("ch_abbrev", "roman-expand", "Ch. IV", "Ch. 4"),
    ("sec_abbrev", "roman-expand", "Sec. IX", "Sec. 9"),
    ("vol_dot", "roman-expand", "Vol. XII", "Vol. 12"),
    ("chapters_plural_range", "roman-expand", "Chapters IV-VI", "Chapters 4-6"),
    ("acts_single_start_range", "roman-expand", "Acts I-III", "Acts 1-3"),
    ("chapter_hyphen_range", "roman-expand", "Chapter IV-VI", "Chapter 4-6"),
    ("act_endash_range", "roman-expand", "Act I\u2013III", "Act 1\u20133"),
    ("bare_heading", "roman-expand", "XIV", "14"),
    ("leading_numeral_title", "roman-expand", "IV. The Return", "4. The Return"),
    ("possessive", "roman-expand", "Chapter IV's themes", "Chapter 4's themes"),
    ("chapter_xiv", "roman-expand", "Chapter XIV", "Chapter 14"),
    ("chapter_I_read", "roman-leave", "the chapter I read was long", "the chapter I read was long"),
    ("no_I_interjection", "roman-leave", "no I did not go", "no I did not go"),
    ("act_I_regret", "roman-leave", "an act I regret to this day", "an act I regret to this day"),
    ("part_I_played", "roman-leave", "a part I played in the war", "a part I played in the war"),
    ("part_C_medicare", "roman-leave", "Part C Medicare Advantage", "Part C Medicare Advantage"),
    ("appendix_C", "roman-leave", "Appendix C contains the tables", "Appendix C contains the tables"),
    ("act_V_of", "roman-leave", "the act V of kindness", "the act V of kindness"),
    ("chapter_MIX_word", "roman-leave", "chapter MIX", "chapter MIX"),
    ("book_X_placeholder", "roman-leave", "book X marks the spot", "book X marks the spot"),
    ("chapter_I_opened", "roman-leave", "Chapter I opened the door", "Chapter I opened the door"),
    ("title_I_funding", "roman-leave", "Title I funding for schools", "Title I funding for schools"),
    ("title_I_gave", "roman-leave", "the title I gave it was wrong", "the title I gave it was wrong"),
    ("volume_I_borrowed", "roman-leave", "the volume I borrowed", "the volume I borrowed"),
    ("henry_viii", "roman-leave", "Henry VIII", "Henry VIII"),
    ("world_war_ii", "roman-leave", "World War II", "World War II"),
    ("no_C_in_word", "roman-leave", "There is no C in that word.", "There is no C in that word."),
    ("article_I_editor", "roman-leave", "He gave article I to the editor.", "He gave article I to the editor."),
    ("section_C_conclude", "roman-leave", "In section C we conclude.", "In section C we conclude."),
    ("for_my_part_I", "roman-leave", "For my part I disagree.", "For my part I disagree."),
    ("section_I_am", "roman-leave", "In section I am confident we succeed.", "In section I am confident we succeed."),
    ("section_DI_code", "roman-leave", "section DI was revised", "section DI was revised"),
    ("no_I_index", "roman-leave", "The book has no I in its index.", "The book has no I in its index."),
    ("no_M_name", "roman-leave", "I have no M in my name.", "I have no M in my name."),
    ("part_I_and_I_alone", "roman-leave", "Part I, and I alone, will decide.", "Part I, and I alone, will decide."),
    ("cite_before_period", "footnote-strip", "The theory holds [12].", "The theory holds."),
    ("cite_mid_sentence", "footnote-strip", "the result [1] was clear", "the result was clear"),
    ("cite_leading", "footnote-strip", "[1] This is the first note.", "This is the first note."),
    ("cite_proven", "footnote-strip", "The claim was proven [12].", "The claim was proven."),
    ("cite_disputed", "footnote-strip", "It was disputed [42] by others.", "It was disputed by others."),
    ("cite_before_comma", "footnote-strip", "The theory [5], though old, holds.", "The theory, though old, holds."),
    ("cite_before_colon", "footnote-strip", "the result [9]: a breakthrough.", "the result: a breakthrough."),
    ("cite_final_nopunct", "footnote-strip", "Ends with citation [7]", "Ends with citation"),
    ("cite_fused_word", "footnote-strip", "word[3] fused citation.", "word fused citation."),
    ("cite_before_semicolon", "footnote-strip", "well established [8]; however, doubts remain.", "well established; however, doubts remain."),
    ("unit_cm2", "footnote-leave", "cm\u00b2", "cm\u00b2"),
    ("emc2", "footnote-leave", "E=mc\u00b2", "E=mc\u00b2"),
    ("pythag", "footnote-leave", "x\u00b2 + y\u00b2 = z\u00b2", "x\u00b2 + y\u00b2 = z\u00b2"),
    ("digit_exponent", "footnote-leave", "2\u00b3 equals 8", "2\u00b3 equals 8"),
    ("bracket_year", "footnote-leave", "Published [1997] in the journal.", "Published [1997] in the journal."),
    ("bracket_year_range", "footnote-leave", "The war [1914\u20131918] reshaped Europe.", "The war [1914\u20131918] reshaped Europe."),
    ("bracket_zeropad", "footnote-leave", "Sample [007] was flagged.", "Sample [007] was flagged."),
    ("interval_pair", "footnote-leave", "the set [1,2] is closed", "the set [1,2] is closed"),
    ("vector_triple", "footnote-leave", "coordinates [1, 2, 3]", "coordinates [1, 2, 3]"),
    ("co2_subscript", "footnote-leave", "CO\u2082 emissions rose", "CO\u2082 emissions rose"),
    ("superscript_index_i", "footnote-leave", "the term x\u2071 vanishes", "the term x\u2071 vanishes"),
    ("letter_exponent_f2", "footnote-leave", "the function f\u00b2 is smooth", "the function f\u00b2 is smooth"),
    ("letter_exponent_p3", "footnote-leave", "the polynomial p\u00b3 term", "the polynomial p\u00b3 term"),
    ("scientific_10e6", "footnote-leave", "about 10\u2076 stars", "about 10\u2076 stars"),
    ("true_footnote_left", "footnote-leave", "end of sentence.\u00b3", "end of sentence.\u00b3"),
    ("page_range_bracket", "footnote-leave", "Turn to [15-30] in the reader.", "Turn to [15-30] in the reader."),
    ("interaction_strip_then_expand", "interaction", "See Chapter IV [3] for details.", "See Chapter 4 for details."),
    ("guard-3d-movie", "roman-leave", "3-D movie", "3-D movie"),
    ("guard-5xl-shirt", "roman-leave", "size 5-XL shirt", "size 5-XL shirt"),
    ("guard-tx-model", "roman-leave", "the T-X model", "the T-X model"),
    ("guard-vitamin-d3", "roman-leave", "vitamin D-3", "vitamin D-3"),
    ("guard-section-8-arabic", "roman-leave", "Section 8 housing", "Section 8 housing"),
    ("guard-part-time", "roman-leave", "part-time job", "part-time job"),
]
# fmt: on

_RF_IDS = [row[0] for row in ROMAN_FOOTNOTE_CASES]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(row[2], row[3]) for row in ROMAN_FOOTNOTE_CASES],
    ids=_RF_IDS,
)
def test_normalize_roman_and_footnote_corpus(raw: str, expected: str):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", [row[2] for row in ROMAN_FOOTNOTE_CASES], ids=_RF_IDS)
def test_normalize_roman_footnote_idempotent(raw: str):
    once = normalize(raw)
    assert normalize(once) == once


def test_roman_footnote_guards_are_load_bearing():
    """The passes are scoped so they never mangle prose or math.

    A label-adjacent numeral must be >=2 chars (so the pronoun "I" is safe), superscript
    digits are left (cm2 / E=mc2 are exponents, not footnotes), and there is no global
    "<digit>-<roman>" rule (3-D, 5-XL must not become 3-500, 5-40).
    """
    assert normalize("the part I played") == "the part I played"
    assert normalize("cm² and E=mc²") == "cm² and E=mc²"
    assert normalize("a 3-D printer, size 5-XL") == "a 3-D printer, size 5-XL"


# URL/email speaking + em-dash/ellipsis
# fmt: off
URL_DASH_CASES = [
    ("scheme_quoted_inline", "url-speak", 'He typed "http://example.com/login" and waited.', 'He typed "example dot com" and waited.'),
    ("scheme_ends_quoted_sentence", "url-speak", '"Visit https://example.com."', '"Visit example dot com."'),
    ("scheme_then_emdash", "url-speak", "Visit https://example.com/p\u2014the best.", "Visit example dot com, the best."),
    ("bare_path_ends_sentence", "url-speak", "See gutenberg.org/ebooks/1342. It is free.", "See gutenberg dot org. It is free."),
    ("scheme_ends_sentence", "url-speak", "Visit https://example.com. Bye.", "Visit example dot com. Bye."),
    ("scheme_midsentence_comma", "url-speak", "At https://example.com, we meet.", "At example dot com, we meet."),
    ("bare_www_comma", "url-speak", "Go to www.example.com, please.", "Go to example dot com, please."),
    ("scheme_creds_dropped", "url-speak", "https://user:pass@example.com/secret", "example dot com"),
    ("bare_host_port_path", "url-speak", "The API lives at api.example.com:8443/v1.", "The API lives at api dot example dot com."),
    ("ipv4_left_literal", "url-speak", "http://127.0.0.1:5000/", "127.0.0.1"),
    ("scheme_query_ends_sentence", "url-speak", "Search bing.com/search?q=cats. Done.", "Search bing dot com. Done."),
    ("doi_paren_period", "url-speak", "(Retrieved from https://doi.org/10.1000/xyz.)", "(Retrieved from doi dot org.)"),
    ("email_basic", "email-speak", "a@b.com", "a at b dot com"),
    ("email_firstlast_local", "email-speak", "Best regards, John.Smith@example.com", "Best regards, John Smith at example dot com"),
    ("email_multidot_local_ccsld", "email-speak", "Contact: mary.jane.watson@newspaper.co.uk", "Contact: mary jane watson at newspaper dot co dot uk"),
    ("email_mailto_prefix", "email-speak", "Send to mailto:x@y.com now.", "Send to x at y dot com now."),
    ("tld_me_filename", "url-leave", "Read read.me before installing.", "Read read.me before installing."),
    ("tld_me_prose", "url-leave", "She whispered, forgive.me for this.", "She whispered, forgive.me for this."),
    ("tld_co_abbrev", "url-leave", "Acme Inc.co-founded the firm.", "Acme Inc.co-founded the firm."),
    ("tld_dev_config", "url-leave", "The config.dev file loads.", "The config.dev file loads."),
    ("tld_io_data", "url-leave", "Save data.io to disk.", "Save data.io to disk."),
    ("tld_io_abbrev", "url-leave", "vs.io? no.", "vs.io? no."),
    ("tld_au_abbrev", "url-leave", "cf.au for Australia.", "cf.au for Australia."),
    ("tld_dev_command", "url-leave", "the go.dev command builds Go.", "the go.dev command builds Go."),
    ("digit_label_version", "url-leave", "1.co is not a version.", "1.co is not a version."),
    ("sentence_boundary_fusion", "url-leave", "It was a tie.Co-workers cheered.", "It was a tie.Co-workers cheered."),
    ("boundary_us_me", "url-leave", "They defeated us.Me and my brother watched.", "They defeated us.Me and my brother watched."),
    ("www_no_tld_placeholder", "url-leave", "Our site is www.example and nothing more.", "Our site is www.example and nothing more."),
    ("abbrev_eg", "leave", "e.g. this", "e.g. this"),
    ("decimal_number", "leave", "3.14", "3.14"),
    ("filename_txt", "leave", "file.txt", "file.txt"),
    ("initials_usa", "leave", "U.S.A.", "U.S.A."),
    ("version_token", "leave", "v2.0", "v2.0"),
    ("emdash_before_period", "dash", "the end\u2014.", "the end."),
    ("emdash_before_question", "dash", "really\u2014?", "really?"),
    ("emdash_before_comma", "dash", "wait\u2014, then", "wait, then"),
    ("double_emdash_between_words", "dash", "done\u2014\u2014done", "done, done"),
    ("emdash_number_range", "dash", "1914\u20141918", "1914 to 1918"),
    ("emdash_page_range", "dash", "See pp. 34\u201437 for details.", "See pp. 34 to 37 for details."),
    ("emdash_parenthetical", "dash", 'He said\u2014and I quote\u2014"no."', 'He said, and I quote, "no."'),
    ("horizontal_bar_dash", "dash", "He said \u2015 nothing.", "He said, nothing."),
    ("double_hyphen", "dash", "wait--then", "wait, then"),
    ("leading_emdash_attribution", "dash", "\u2014Anonymous", "Anonymous"),
    ("trailing_emdash", "dash", "Wait\u2014", "Wait"),
    ("endash_range_untouched", "leave", "1914\u20131918", "1914\u20131918"),
    ("single_hyphen_compound", "leave", "pages 12-18", "pages 12-18"),
    ("ellipsis_char", "ellipsis", "wait\u2026 what", "wait... what"),
    ("ellipsis_spaced_dots", "ellipsis", "Well . . . . maybe", "Well ... maybe"),
    ("emdash_before_ellipsis", "ellipsis", "So it goes\u2014\u2026", "So it goes..."),
    ("plain_prose_url_control", "interaction", "The theory was debunked. See rebuttal.org for details.", "The theory was debunked. See rebuttal dot org for details."),
    ("scheme_url_then_dash_then_ellipsis_dialogue", "interaction", "\u201cI thought\u2014\u2019 she paused \u2014that you knew\u2026\u201d", "\u201cI thought, \u2019 she paused, that you knew...\u201d"),
    ("url_subdomain_full_host", "url-speak", "See en.wikipedia.org now.", "See en dot wikipedia dot org now."),
    ("url_simple_scheme", "url-speak", "Visit https://example.com.", "Visit example dot com."),
    ("email_plus_tag", "email-speak", "Write a+b@x.org today.", "Write a+b at x dot org today."),
    ("url_two_in_one", "url-speak", "Try example.com and test.org.", "Try example dot com and test dot org."),
    ("leave_ratio", "leave", "The score was 3:2.", "The score was 3:2."),
    ("leave_time_am", "leave", "Meet at 9 a.m. sharp.", "Meet at 9 a.m. sharp."),
]
# fmt: on

_UD_IDS = [row[0] for row in URL_DASH_CASES]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(row[2], row[3]) for row in URL_DASH_CASES],
    ids=_UD_IDS,
)
def test_normalize_url_and_dash_corpus(raw: str, expected: str):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", [row[2] for row in URL_DASH_CASES], ids=_UD_IDS)
def test_normalize_url_dash_idempotent(raw: str):
    once = normalize(raw)
    assert normalize(once) == once


def test_url_dash_guards_are_load_bearing():
    """URLs/emails are spoken and em-dashes pause without mangling prose or math.

    A scheme-less host must end in a common TLD (so "e.g."/"file.txt"/"read.me" stay), a
    single hyphen and en-dash are untouched (ranges), and a trailing em-dash before a
    period drops rather than leaving a stray ", .".
    """
    assert normalize("e.g. read.me and file.txt") == "e.g. read.me and file.txt"
    assert normalize("Email a@b.com now.") == "Email a at b dot com now."
    assert normalize("pages 12-18 stay") == "pages 12-18 stay"
