"""Shared GGUF key-value helpers — copying metadata KVs between GGUFs.

Factored out of experiments/v0.0/partial_gguf.py so the packager (which splits
a GGUF) and the assembler (which recombines fragments) use byte-identical KV
handling. Keep this in lock-step with partial_gguf.py's copy logic — the
assembled sub-GGUF must carry the same metadata a partial_gguf cut would.
"""
from __future__ import annotations

from gguf import GGUFReader, GGUFWriter, GGUFValueType

# KVs the writer manages itself / that name the file rather than the model.
SKIP_KV = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
           "general.architecture"}

TYPED_ADDERS = {
    GGUFValueType.UINT8:   "add_uint8",
    GGUFValueType.INT8:    "add_int8",
    GGUFValueType.UINT16:  "add_uint16",
    GGUFValueType.INT16:   "add_int16",
    GGUFValueType.UINT32:  "add_uint32",
    GGUFValueType.INT32:   "add_int32",
    GGUFValueType.FLOAT32: "add_float32",
    GGUFValueType.BOOL:    "add_bool",
    GGUFValueType.STRING:  "add_string",
    GGUFValueType.UINT64:  "add_uint64",
    GGUFValueType.INT64:   "add_int64",
    GGUFValueType.FLOAT64: "add_float64",
}


def field_value(field):
    """Best-effort decode of a GGUFReader field into a Python value."""
    types = field.types
    primary = types[0]
    if primary == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    if primary == GGUFValueType.ARRAY:
        elem_type = types[1] if len(types) > 1 else None
        if elem_type == GGUFValueType.STRING:
            return [bytes(field.parts[i]).decode("utf-8", errors="replace") for i in field.data]
        return [field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist") else field.parts[i][0]
                for i in field.data]
    return field.parts[field.data[0]].tolist()[0] if hasattr(field.parts[field.data[0]], "tolist") else field.parts[field.data[0]][0]


def read_arch_and_block_count(reader: GGUFReader) -> tuple[str, int | None]:
    arch = "llama"
    block_count = None
    for f in reader.fields.values():
        if f.name == "general.architecture":
            arch = field_value(f)
        if f.name.endswith(".block_count"):
            block_count = int(field_value(f))
    return arch, block_count


def copy_kvs(reader: GGUFReader, writer: GGUFWriter) -> tuple[int, list]:
    """Copy every model KV (except SKIP_KV) from reader into writer.
    Returns (copied_count, skipped[(name, reason)])."""
    copied = 0
    skipped: list = []
    for f in reader.fields.values():
        if f.name in SKIP_KV:
            continue
        primary = f.types[0]
        try:
            if primary == GGUFValueType.ARRAY:
                writer.add_array(f.name, field_value(f))
            elif primary in TYPED_ADDERS:
                getattr(writer, TYPED_ADDERS[primary])(f.name, field_value(f))
            else:
                skipped.append((f.name, str(primary)))
                continue
            copied += 1
        except Exception as e:  # noqa: BLE001 — best-effort, mirror partial_gguf
            skipped.append((f.name, f"{type(e).__name__}: {e}"))
    return copied, skipped
