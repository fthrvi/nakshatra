import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from pathlib import Path
from procargs import serve_args, gpu_layers


def test_serve_args_basic():
    """Test basic command line parsing."""
    args = ["llama-server", "--n-gpu-layers", "99", "--port", "8080"]
    expected = ["llama-server", "--n-gpu-layers", "99", "--port", "8080"]
    
    assert serve_args(1234, proc_root="/nonexistent") == []
    assert gpu_layers(1234, proc_root="/nonexistent") is None


def test_gpu_layers_two_arg_form(tmp_path):
    """Test --n-gpu-layers 99 form."""
    args = ["llama-server", "--n-gpu-layers", "99", "--port", "8080"]
    (tmp_path / "1234").mkdir()
    (tmp_path / "1234" / "cmdline").write_bytes(b"\x00".join(a.encode() for a in args) + b"\x00")
    
    assert gpu_layers(1234, proc_root=str(tmp_path)) == 99


def test_gpu_layers_one_arg_form(tmp_path):
    """Test --n-gpu-layers=99 form."""
    args = ["llama-server", "--n-gpu-layers=42", "--port", "8080"]
    (tmp_path / "5678").mkdir()
    (tmp_path / "5678" / "cmdline").write_bytes(b"\x00".join(a.encode() for a in args) + b"\x00")
    
    assert gpu_layers(5678, proc_root=str(tmp_path)) == 42


def test_gpu_layers_absent(tmp_path):
    """Test when flag is absent."""
    args = ["llama-server", "--port", "8080"]
    (tmp_path / "9999").mkdir()
    (tmp_path / "9999" / "cmdline").write_bytes(b"\x00".join(a.encode() for a in args) + b"\x00")
    
    assert gpu_layers(9999, proc_root=str(tmp_path)) is None


def test_gpu_layers_trailing_nul(tmp_path):
    """Test that trailing NUL is handled correctly."""
    args = ["llama-server", "--n-gpu-layers", "5"]
    (tmp_path / "1111").mkdir()
    # Write with trailing NUL as per cmdline format
    (tmp_path / "1111" / "cmdline").write_bytes(b"\x00".join(a.encode() for a in args) + b"\x00")
    
    assert gpu_layers(1111, proc_root=str(tmp_path)) == 5
    assert serve_args(1111, proc_root=str(tmp_path)) == args


def test_gpu_layers_empty_cmdline(tmp_path):
    """Test empty cmdline (zombie process)."""
    (tmp_path / "2222").mkdir()
    (tmp_path / "2222" / "cmdline").write_bytes(b"")
    
    assert gpu_layers(2222, proc_root=str(tmp_path)) is None
    assert serve_args(2222, proc_root=str(tmp_path)) == []


def test_gpu_layers_missing_pid(tmp_path):
    """Test missing pid directory."""
    assert gpu_layers(9999, proc_root=str(tmp_path)) is None
    assert serve_args(9999, proc_root=str(tmp_path)) == []


def test_gpu_layers_non_integer_value(tmp_path):
    """Test when value is not an integer."""
    args = ["llama-server", "--n-gpu-layers", "notanumber"]
    (tmp_path / "3333").mkdir()
    (tmp_path / "3333" / "cmdline").write_bytes(b"\x00".join(a.encode() for a in args) + b"\x00")
    
    assert gpu_layers(3333, proc_root=str(tmp_path)) is None


def test_gpu_layers_contains_flag_name(tmp_path):
    """Test that --n-gpu-layers-max is not mistaken for --n-gpu-layers."""
    args = ["llama-server", "--n-gpu-layers-max", "100", "--n-gpu-layers", "50"]
    (tmp_path / "4444").mkdir()
    (tmp_path / "4444" / "cmdline").write_bytes(b"\x00".join(a.encode() for a in args) + b"\x00")
    
    assert gpu_layers(4444, proc_root=str(tmp_path)) == 50


def test_gpu_layers_invalid_one_arg_form(tmp_path):
    """Test invalid value in one-argument form."""
    args = ["llama-server", "--n-gpu-layers=abc"]
    (tmp_path / "5555").mkdir()
    (tmp_path / "5555" / "cmdline").write_bytes(b"\x00".join(a.encode() for a in args) + b"\x00")
    
    assert gpu_layers(5555, proc_root=str(tmp_path)) is None


def test_serve_args_unicode_error_handling(tmp_path):
    """Test handling of invalid UTF-8."""
    (tmp_path / "6666").mkdir()
    (tmp_path / "6666" / "cmdline").write_bytes(b"\xff\xfe\x00test\x00")
    
    result = serve_args(6666, proc_root=str(tmp_path))
    # \xff\xfe decoded with errors='replace' becomes '\ufffd\ufffd'
    assert result == ["\ufffd\ufffd", "test"]