import unittest
from natclass import classify_nat

class TestClassifyNAT(unittest.TestCase):
    
    def test_two_servers_agreeing_port_restricted(self):
        observations = [
            {"local_port": 1234, "server": "stun1.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000},
            {"local_port": 1234, "server": "stun2.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("port-restricted", "same local_port, two servers, same mapped_port"))
    
    def test_two_servers_disagreeing_symmetric(self):
        observations = [
            {"local_port": 1234, "server": "stun1.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000},
            {"local_port": 1234, "server": "stun2.example.com", "mapped_ip": "203.0.113.2", "mapped_port": 6000}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("symmetric", "same local_port, two servers, different mapped_ports"))
    
    def test_one_observation_unknown(self):
        observations = [
            {"local_port": 1234, "server": "stun1.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("unknown", "fewer than two observations from different servers"))
    
    def test_two_observations_same_server_unknown(self):
        observations = [
            {"local_port": 1234, "server": "stun1.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000},
            {"local_port": 1234, "server": "stun1.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5001}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("unknown", "fewer than two observations from different servers"))
    
    def test_mapped_ip_equals_local_ip_none(self):
        observations = [
            {"local_port": 1234, "server": "stun1.example.com", "mapped_ip": "192.168.1.100", "mapped_port": 5000, "local_ip": "192.168.1.100"}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("none", "mapped_ip equals local_ip"))
    
    def test_three_with_one_outlier_symmetric(self):
        observations = [
            {"local_port": 1234, "server": "stun1.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000},
            {"local_port": 1234, "server": "stun2.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000},
            {"local_port": 1234, "server": "stun3.example.com", "mapped_ip": "203.0.113.2", "mapped_port": 6000}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("symmetric", "same local_port, two servers, different mapped_ports"))
    
    def test_empty_list_unknown(self):
        observations = []
        result = classify_nat(observations)
        self.assertEqual(result, ("unknown", "no observations"))
    
    def test_non_list_input_unknown(self):
        result = classify_nat("not a list")
        self.assertEqual(result, ("unknown", "non-list input"))
    
    def test_non_dict_entry_unknown(self):
        observations = [{"local_port": 1234, "server": "stun1.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000}, "not a dict"]
        result = classify_nat(observations)
        self.assertEqual(result, ("unknown", "non-dict entry"))
    
    def test_missing_keys_unknown(self):
        observations = [{"local_port": 1234, "server": "stun1.example.com"}]  # missing mapped_ip and mapped_port
        result = classify_nat(observations)
        self.assertEqual(result, ("unknown", "missing required keys"))
    
    def test_non_int_ports_unknown(self):
        observations = [
            {"local_port": "1234", "server": "stun1.example.com", "mapped_ip": "203.0.113.1", "mapped_port": 5000}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("unknown", "non-int ports or non-string fields"))
    
    def test_non_string_server_unknown(self):
        observations = [
            {"local_port": 1234, "server": 12345, "mapped_ip": "203.0.113.1", "mapped_port": 5000}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("unknown", "non-int ports or non-string fields"))
    
    def test_non_string_mapped_ip_unknown(self):
        observations = [
            {"local_port": 1234, "server": "stun1.example.com", "mapped_ip": 12345, "mapped_port": 5000}
        ]
        result = classify_nat(observations)
        self.assertEqual(result, ("unknown", "non-int ports or non-string fields"))

if __name__ == "__main__":
    unittest.main()