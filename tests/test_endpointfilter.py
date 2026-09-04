import pytest
from endpointfilter import usable_endpoints


class TestUsableEndpoints:
    def test_public_ipv4_kept(self):
        endpoints = [("8.8.8.8", 53)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == [("8.8.8.8", 53)]
        assert problems == []
    
    def test_public_ipv6_kept(self):
        endpoints = [("2001:4860:4860::8888", 53)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == [("2001:4860:4860::8888", 53)]
        assert problems == []
    
    def test_loopback_dropped(self):
        endpoints = [("127.0.0.1", 8080)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "loopback address: 127.0.0.1" in problems
    
    def test_ipv6_loopback_dropped(self):
        endpoints = [("::1", 8080)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "loopback address: ::1" in problems
    
    def test_unspecified_ipv4_dropped(self):
        endpoints = [("0.0.0.0", 8080)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "unspecified address: 0.0.0.0" in problems
    
    def test_unspecified_ipv6_dropped(self):
        endpoints = [("::", 8080)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "unspecified address: ::" in problems
    
    def test_multicast_dropped(self):
        endpoints = [("224.0.0.1", 8080)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "multicast address: 224.0.0.1" in problems
    
    def test_link_local_ipv4_dropped(self):
        endpoints = [("169.254.1.1", 8080)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "link-local address: 169.254.1.1" in problems
    
    def test_link_local_ipv6_dropped(self):
        endpoints = [("fe80::1", 8080)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "link-local address: fe80::1" in problems
    
    def test_private_ipv4_with_allow_lan_true(self):
        endpoints = [("192.168.1.1", 8080)]
        usable, problems = usable_endpoints(endpoints, allow_lan=True)
        assert usable == [("192.168.1.1", 8080)]
        assert problems == []
    
    def test_private_ipv4_with_allow_lan_false(self):
        endpoints = [("192.168.1.1", 8080)]
        usable, problems = usable_endpoints(endpoints, allow_lan=False)
        assert usable == []
        assert "private address: 192.168.1.1" in problems
    
    def test_private_ipv6_with_allow_lan_true(self):
        endpoints = [("fc00::1", 8080)]
        usable, problems = usable_endpoints(endpoints, allow_lan=True)
        assert usable == [("fc00::1", 8080)]
        assert problems == []
    
    def test_private_ipv6_with_allow_lan_false(self):
        endpoints = [("fc00::1", 8080)]
        usable, problems = usable_endpoints(endpoints, allow_lan=False)
        assert usable == []
        assert "private address: fc00::1" in problems
    
    def test_dns_name_dropped(self):
        endpoints = [("example.com", 8080)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "DNS name not allowed: example.com" in problems
    
    def test_port_zero_dropped(self):
        endpoints = [("8.8.8.8", 0)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "invalid port 0 for endpoint ('8.8.8.8', 0)" in problems
    
    def test_port_70000_dropped(self):
        endpoints = [("8.8.8.8", 70000)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "invalid port 70000 for endpoint ('8.8.8.8', 70000)" in problems
    
    def test_malformed_shape_dropped(self):
        endpoints = [("8.8.8.8",)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "malformed entry: ('8.8.8.8',)" in problems
    
    def test_order_preserved(self):
        endpoints = [
            ("8.8.8.8", 53),
            ("1.1.1.1", 53),
            ("9.9.9.9", 53),
        ]
        usable, problems = usable_endpoints(endpoints)
        assert usable == endpoints
    
    def test_exact_duplicates_collapsed(self):
        endpoints = [
            ("8.8.8.8", 53),
            ("8.8.8.8", 53),
            ("1.1.1.1", 53),
        ]
        usable, problems = usable_endpoints(endpoints)
        assert usable == [("8.8.8.8", 53), ("1.1.1.1", 53)]
        assert "duplicate endpoint: 8.8.8.8:53" in problems
    
    def test_empty_list(self):
        endpoints = []
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert problems == []
    
    def test_non_list_input(self):
        endpoints = "not a list"
        usable, problems = usable_endpoints(endpoints)
        assert usable == []
        assert "non-list input" in problems
    
    def test_none_entries(self):
        endpoints = [("8.8.8.8", 53), None, ("1.1.1.1", 53)]
        usable, problems = usable_endpoints(endpoints)
        assert usable == [("8.8.8.8", 53), ("1.1.1.1", 53)]
        assert "None entry" in problems
    
    def test_never_raises(self):
        # Test various malformed inputs that shouldn't raise
        malformed_inputs = [
            [("8.8.8.8", "not an int")],
            [("not a string", 53)],
            [("8.8.8.8", 53, "extra")],
            [123],
            [[8.8, 8.8]],
            [()],
            [()],
            [("", 0)],
            [("127.0.0.1", 0)],
        ]
        
        for endpoints in malformed_inputs:
            try:
                usable, problems = usable_endpoints(endpoints)
                # Should not raise, just return empty or partial results
            except Exception:
                pytest.fail(f"Should not raise for input: {endpoints}")