import unittest
from probeparse import parse_probe

class TestParseProbe(unittest.TestCase):
    
    def test_shape1_parsed(self):
        body = '{"tokens": [1, 2, 3], "layers": {"start": 0, "end": 32}}'
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [1, 2, 3])
        self.assertEqual(result["probe_layers"], [0, 32])
        self.assertIsNone(result["error"])
    
    def test_shape2_zeros_and_none_layers(self):
        body = '{"choices":[{"text":"hello"}], "usage":{"completion_tokens":3}}'
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [0, 0, 0])
        self.assertIsNone(result["probe_layers"])
        self.assertIsNone(result["error"])
    
    def test_invalid_json(self):
        body = 'not valid json'
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [])
        self.assertIsNone(result["probe_layers"])
        self.assertIn("invalid JSON", result["error"])
    
    def test_empty_body(self):
        body = ""
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [])
        self.assertIsNone(result["probe_layers"])
        self.assertIn("empty", result["error"])
    
    def test_top_level_array(self):
        body = '[1, 2, 3]'
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [])
        self.assertIsNone(result["probe_layers"])
        self.assertIn("top-level is not an object", result["error"])
    
    def test_string_body(self):
        body = '"just a string"'
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [])
        self.assertIsNone(result["probe_layers"])
        self.assertIn("top-level is not an object", result["error"])
    
    def test_non_int_tokens_dropped_with_error(self):
        body = '{"tokens": [1, "two", 3], "layers": {"start": 0, "end": 32}}'
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [1, 3])
        self.assertEqual(result["probe_layers"], [0, 32])
        self.assertIn("non-int token dropped", result["error"])
    
    def test_missing_layers_gives_none(self):
        body = '{"tokens": [1, 2, 3]}'
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [1, 2, 3])
        self.assertIsNone(result["probe_layers"])
        self.assertIn("layers is not a 2-field object", result["error"])
    
    def test_layers_not_2_field_object(self):
        body = '{"tokens": [1, 2, 3], "layers": {"start": 0}}'
        result = parse_probe(body)
        self.assertEqual(result["probe_tokens"], [1, 2, 3])
        self.assertIsNone(result["probe_layers"])
        self.assertIn("layers is not a 2-field object", result["error"])
    
    def test_non_string_input(self):
        result = parse_probe(123)
        self.assertEqual(result["probe_tokens"], [])
        self.assertIsNone(result["probe_layers"])
        self.assertIn("not a string", result["error"])
    
    def test_never_raises(self):
        # Test various malformed inputs that could cause crashes
        inputs = [
            None,
            123,
            45.67,
            [1, 2, 3],
            {"tokens": "not a list", "layers": {"start": 0, "end": 32}},
            {"tokens": [1, 2, 3], "layers": "not an object"},
            {"tokens": [1, 2, 3], "layers": {"start": "not int", "end": 32}},
            {"tokens": [1, 2, 3], "layers": {"start": 0, "end": "not int"}},
            {"choices": "not a list", "usage": {"completion_tokens": 3}},
            {"choices": [{"text": "hello"}], "usage": "not an object"},
            {"choices": [{"text": "hello"}], "usage": {"completion_tokens": "not int"}},
        ]
        
        for inp in inputs:
            try:
                result = parse_probe(inp)
                # Should always return a dict with expected keys
                self.assertIn("probe_tokens", result)
                self.assertIn("probe_layers", result)
                self.assertIn("error", result)
            except Exception:
                self.fail(f"parse_probe raised on input: {inp}")

if __name__ == "__main__":
    unittest.main()