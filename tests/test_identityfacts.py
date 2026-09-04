import pytest
from identityfacts import identity_facts


class TestIdentityFacts:
    def test_registered_node(self):
        # Node exists, key matches roster entry
        result = identity_facts(
            key_existed=True,
            pubkey="a" * 64,
            roster={"node1": "a" * 64},
            node_id="node1"
        )
        assert result == {
            "identity_pubkey": "a" * 64,
            "identity_is_new": False,
            "registered": True,
            "roster_pubkey": "a" * 64
        }

    def test_brand_new_key(self):
        # New node with new key
        result = identity_facts(
            key_existed=False,
            pubkey="b" * 64,
            roster={},
            node_id="node2"
        )
        assert result == {
            "identity_pubkey": "b" * 64,
            "identity_is_new": True,
            "registered": False,
            "roster_pubkey": None
        }

    def test_node_absent_from_roster(self):
        # Node not in roster
        result = identity_facts(
            key_existed=True,
            pubkey="c" * 64,
            roster={"node1": "d" * 64},
            node_id="node3"
        )
        assert result == {
            "identity_pubkey": "c" * 64,
            "identity_is_new": False,
            "registered": False,
            "roster_pubkey": None
        }

    def test_roster_entry_with_different_key(self):
        # Roster entry exists but with different key (re-key scenario)
        result = identity_facts(
            key_existed=True,
            pubkey="e" * 64,
            roster={"node4": "f" * 64},
            node_id="node4"
        )
        assert result == {
            "identity_pubkey": "e" * 64,
            "identity_is_new": False,
            "registered": False,
            "roster_pubkey": "f" * 64
        }

    def test_mixed_case_pubkey_rejected_not_lowercased(self):
        # Mixed case pubkey should be rejected (not lowercased)
        result = identity_facts(
            key_existed=True,
            pubkey="A" * 32 + "a" * 32,  # Mixed case
            roster={"node5": "A" * 32 + "a" * 32},
            node_id="node5"
        )
        assert result == {
            "identity_pubkey": "",
            "identity_is_new": False,
            "registered": False,
            "roster_pubkey": "A" * 32 + "a" * 32
        }

    def test_malformed_pubkey_gives_empty_string(self):
        # Malformed pubkey (wrong length or non-hex)
        result = identity_facts(
            key_existed=True,
            pubkey="g" * 63,  # Wrong length
            roster={},
            node_id="node6"
        )
        assert result == {
            "identity_pubkey": "",
            "identity_is_new": False,
            "registered": False,
            "roster_pubkey": None
        }

    def test_private_key_in_roster_never_reaches_output(self):
        # Test with private key in roster under three different names
        private_key = "h" * 64  # Simulating a private key
        
        result = identity_facts(
            key_existed=True,
            pubkey="i" * 64,
            roster={
                "private_key_field": private_key,
                "secret_key": private_key,
                "secret": private_key
            },
            node_id="node7"
        )
        
        # Verify none of the private key values appear in the output
        assert private_key not in result.values()
        assert result["identity_pubkey"] == ""
        assert result["roster_pubkey"] is None

    def test_non_dict_roster(self):
        # Non-dict roster should be handled gracefully
        result = identity_facts(
            key_existed=True,
            pubkey="j" * 64,
            roster="not a dict",
            node_id="node8"
        )
        assert result == {
            "identity_pubkey": "",
            "identity_is_new": False,
            "registered": False,
            "roster_pubkey": None
        }

    def test_non_string_inputs(self):
        # Non-string inputs should be handled gracefully
        result = identity_facts(
            key_existed=True,
            pubkey=123,  # Non-string
            roster={"node9": "k" * 64},
            node_id=456  # Non-string
        )
        assert result == {
            "identity_pubkey": "",
            "identity_is_new": False,
            "registered": False,
            "roster_pubkey": None
        }

    def test_exact_key_set_returned(self):
        # Verify the exact keys are returned
        result = identity_facts(
            key_existed=True,
            pubkey="l" * 64,
            roster={"node10": "l" * 64},
            node_id="node10"
        )
        assert set(result.keys()) == {"identity_pubkey", "identity_is_new", "registered", "roster_pubkey"}