"""Tests for app.utils.refang."""

from app.utils.refang import refang, refang_artifacts


class TestRefangSingle:
    def test_bracketed_dot_domain(self):
        assert refang("evil[.]com") == "evil.com"
        assert refang("sub.evil[.]com") == "sub.evil.com"

    def test_bracketed_dot_ipv4(self):
        assert refang("192[.]168[.]1[.]1") == "192.168.1.1"

    def test_bracketed_dot_with_whitespace(self):
        assert refang("evil [.] com") == "evil . com"  # preserves spaces around the dot
        assert refang("evil[ . ]com") == "evil.com"

    def test_bracketed_dot_round_brackets(self):
        assert refang("evil(.)com") == "evil.com"

    def test_bracketed_dot_curly_brackets(self):
        assert refang("evil{.}com") == "evil.com"

    def test_hxxp_scheme(self):
        assert refang("hxxp://evil.com/path") == "http://evil.com/path"
        assert refang("hxxps://evil.com") == "https://evil.com"
        assert refang("hXXps://EvilCorp.com") == "https://EvilCorp.com"

    def test_combined_url_defang(self):
        assert refang("hxxps://evil[.]com[/]path") == "https://evil.com/path"

    def test_bracketed_at_email(self):
        assert refang("user[at]evil[.]com") == "user@evil.com"
        assert refang("user(at)evil(.)com") == "user@evil.com"
        # When defang brackets have surrounding whitespace in the source,
        # only the bracket itself is replaced; the surrounding whitespace
        # is preserved as-is.
        assert refang("user [at] evil [.] com") == "user @ evil . com"

    def test_bracketed_colon_port(self):
        assert refang("evil[.]com[:]443") == "evil.com:443"

    def test_backslash_dot(self):
        assert refang("evil\\.com") == "evil.com"

    def test_idempotent_on_canonical(self):
        assert refang("evil.com") == "evil.com"
        assert refang("https://evil.com/path") == "https://evil.com/path"
        assert refang("user@evil.com") == "user@evil.com"

    def test_empty_string(self):
        assert refang("") == ""

    def test_whitespace_stripped(self):
        assert refang("  evil[.]com  ") == "evil.com"

    def test_non_string_passthrough(self):
        # Defensive: values that aren't strings should pass through.
        assert refang(None) is None
        assert refang(12345) == 12345

    def test_does_not_mangle_unrelated_brackets(self):
        # `[ATTRIBUTION]` style tags should not be touched (no `.` between brackets).
        assert refang("APT29 [ATTRIBUTION]") == "APT29 [ATTRIBUTION]"

    def test_does_not_touch_hxxp_substring(self):
        # `hxxp` not followed by `://` is left alone.
        assert refang("the hxxp prefix is a defang") == "the hxxp prefix is a defang"


class TestRefangArtifacts:
    def test_empty_dict(self):
        assert refang_artifacts({}) == {}

    def test_passthrough_categories(self):
        # file_hashes is not in the refang list — values pass through.
        out = refang_artifacts({"file_hashes": ["abc123def456"]})
        assert out == {"file_hashes": ["abc123def456"]}

    def test_refangs_domains(self):
        out = refang_artifacts({"c2_domains": ["evil[.]com", "another[.]net"]})
        assert out == {"c2_domains": ["evil.com", "another.net"]}

    def test_refangs_urls(self):
        out = refang_artifacts({"urls": ["hxxps://evil[.]com/payload"]})
        assert out == {"urls": ["https://evil.com/payload"]}

    def test_refangs_ips(self):
        out = refang_artifacts({"c2_ips": ["192[.]168[.]1[.]1", "10[.]0[.]0[.]1"]})
        assert out == {"c2_ips": ["192.168.1.1", "10.0.0.1"]}

    def test_dedup_after_refang(self):
        # Defanged + fanged forms of the same value collapse.
        out = refang_artifacts({"c2_domains": ["evil[.]com", "evil.com"]})
        assert out == {"c2_domains": ["evil.com"]}

    def test_drops_empty_categories(self):
        out = refang_artifacts({"c2_domains": [], "c2_ips": ["192[.]168[.]1[.]1"]})
        assert out == {"c2_ips": ["192.168.1.1"]}

    def test_skips_non_list_values(self):
        out = refang_artifacts({"c2_domains": "not-a-list"})
        assert out == {}

    def test_skips_non_string_entries(self):
        out = refang_artifacts({"c2_domains": [None, 12345, "evil[.]com"]})
        assert out == {"c2_domains": ["evil.com"]}
