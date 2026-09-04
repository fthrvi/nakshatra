"""Reading a container's self-report — including when it never made one."""
import pytest

from reportparse import parse_report

GOOD = ('###NAKSHATRA-REPORT###{"exit_code":0,"daemon_pid":4242,'
        '"running_argv":["llama-worker","--n-gpu-layers","99"],'
        '"detected_accel":"cuda","http_status":200}###END###')


def test_a_clean_report_after_a_long_log():
    r, p = parse_report("apt output\n" * 200 + GOOD)
    assert p == [] and r["daemon_pid"] == 4242


def test_a_decoy_marker_earlier_in_the_log_does_not_win():
    """⚠️ A build log is thousands of lines; a package name or echoed script containing the
    marker must not beat the real report at the bottom."""
    decoy = '###NAKSHATRA-REPORT###{"exit_code":99}###END###\n'
    r, _ = parse_report(decoy + "more log\n" + GOOD)
    assert r["exit_code"] == 0 and r["daemon_pid"] == 4242


def test_no_report_is_distinct_from_a_failing_report():
    """⚠️ The container died before it could speak — that points at the harness, not the
    installer, and conflating them costs an operator a day."""
    _, p = parse_report("Reading package lists...\nKilled\n")
    assert p and "no report" in p[0]


@pytest.mark.parametrize("blob,expect", [
    ("###NAKSHATRA-REPORT###{oops###END###", "not valid JSON"),
    ("###NAKSHATRA-REPORT###[1,2]###END###", "not an object"),
    ('###NAKSHATRA-REPORT###"a string"###END###', "not an object"),
])
def test_malformed_content_between_the_markers(blob, expect):
    r, p = parse_report(blob)
    assert r == {} and any(expect in x for x in p)


def test_a_wrong_typed_field_is_named_and_nulled():
    r, p = parse_report('###NAKSHATRA-REPORT###{"exit_code":"zero"}###END###')
    assert r["exit_code"] is None
    assert any("'exit_code'" in x and "str" in x for x in p)


def test_a_missing_field_is_noted_not_fatal():
    r, p = parse_report('###NAKSHATRA-REPORT###{"exit_code":0}###END###')
    assert r["exit_code"] == 0 and r["daemon_pid"] is None
    assert any("missing 'daemon_pid'" in x for x in p)


def test_booleans_are_not_ints():
    r, p = parse_report('###NAKSHATRA-REPORT###{"exit_code":true}###END###')
    assert r["exit_code"] is None and any("bool" in x for x in p)


def test_unexpected_fields_are_noted_but_not_fatal():
    r, p = parse_report('###NAKSHATRA-REPORT###{"exit_code":0,"future":1}###END###')
    assert r["exit_code"] == 0
    assert any("unexpected field" in x for x in p)


def test_a_megabyte_of_noise_before_a_valid_report():
    r, _ = parse_report("x" * 1_000_000 + GOOD)
    assert r["daemon_pid"] == 4242


@pytest.mark.parametrize("bad", [None, 42, b"bytes", [], {}])
def test_non_text_never_raises(bad):
    r, p = parse_report(bad)
    assert r == {} and p


def test_unterminated_markers():
    for blob in ("###NAKSHATRA-REPORT###{}", "{}###END###", "###END###abc###NAKSHATRA-REPORT###"):
        r, p = parse_report(blob)
        assert r == {} and p
