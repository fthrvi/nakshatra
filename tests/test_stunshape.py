import unittest
from stunshape import shape_observations


class TestShapeObservations(unittest.TestCase):
    def test_two_clean_ipv4_replies(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50:40000", "error": None},
            {"server": "stun2.example.com", "local_port": 50001, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.51:40001", "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 2)
        self.assertEqual(len(problems), 0)
        self.assertEqual(observations[0]["server"], "stun1.example.com")
        self.assertEqual(observations[0]["mapped_ip"], "203.0.113.50")
        self.assertEqual(observations[0]["mapped_port"], 40000)
        self.assertEqual(observations[1]["server"], "stun2.example.com")
        self.assertEqual(observations[1]["mapped_ip"], "203.0.113.51")
        self.assertEqual(observations[1]["mapped_port"], 40001)
    
    def test_ipv6_mapped_address(self):
        replies = [
            {"server": "stun6.example.com", "local_port": 50002, "local_ip": "2001:db8::1",
             "mapped": "[2001:db8::1]:40002", "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(problems), 0)
        self.assertEqual(observations[0]["mapped_ip"], "2001:db8::1")
        self.assertEqual(observations[0]["mapped_port"], 40002)
    
    def test_reply_with_error_dropped(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50:40000", "error": None},
            {"server": "stun2.example.com", "local_port": 50001, "local_ip": "192.168.1.100",
             "mapped": None, "error": "timeout"}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("error from stun2.example.com", problems)
    
    def test_mapped_none_dropped(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": None, "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 0)
        self.assertEqual(len(problems), 1)
        self.assertIn("no mapped address from stun1.example.com", problems)
    
    def test_mapped_no_colon(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50", "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 0)
        self.assertEqual(len(problems), 1)
        self.assertIn("malformed mapped address from stun1.example.com", problems)
    
    def test_non_numeric_port(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50:abc", "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 0)
        self.assertEqual(len(problems), 1)
        self.assertIn("malformed mapped address from stun1.example.com", problems)
    
    def test_port_zero(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50:0", "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 0)
        self.assertEqual(len(problems), 1)
        self.assertIn("malformed mapped address from stun1.example.com", problems)
    
    def test_port_99999(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50:99999", "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 0)
        self.assertEqual(len(problems), 1)
        self.assertIn("malformed mapped address from stun1.example.com", problems)
    
    def test_duplicate_server_collapses(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50:40000", "error": None},
            {"server": "stun1.example.com", "local_port": 50001, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.51:40001", "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("duplicate server stun1.example.com", problems)
        # First reply should be kept
        self.assertEqual(observations[0]["mapped_port"], 40000)
    
    def test_output_contains_exactly_five_keys(self):
        replies = [
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50:40000", "error": None}
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 1)
        self.assertEqual(set(observations[0].keys()), 
                         {"server", "local_port", "local_ip", "mapped_ip", "mapped_port"})
    
    def test_empty_list(self):
        observations, problems = shape_observations([])
        self.assertEqual(observations, [])
        self.assertEqual(problems, [])
    
    def test_non_list_input(self):
        observations, problems = shape_observations("not a list")
        self.assertEqual(observations, [])
        self.assertEqual(problems, [])
    
    def test_none_entries(self):
        replies = [
            None,
            {"server": "stun1.example.com", "local_port": 50000, "local_ip": "192.168.1.100",
             "mapped": "203.0.113.50:40000", "error": None},
            None
        ]
        observations, problems = shape_observations(replies)
        self.assertEqual(len(observations), 1)
        self.assertEqual(len(problems), 0)
    
    def test_never_raises(self):
        # Test various malformed inputs that could cause issues
        malformed_inputs = [
            [123],  # non-dict in list
            [{"server": "test", "local_port": "not an int"}],  # wrong types
            [{"server": "test", "local_port": 50000, "local_ip": "192.168.1.100",
              "mapped": "203.0.113.50:40000", "error": None}],
            [{"server": "test", "local_port": 50000, "local_ip": "192.168.1.100",
              "mapped": "[2001:db8::1]:40000", "error": None}],
        ]
        for inp in malformed_inputs:
            try:
                shape_observations(inp)
            except Exception:
                self.fail(f"shape_observations raised on input: {inp}")


if __name__ == "__main__":
    unittest.main()