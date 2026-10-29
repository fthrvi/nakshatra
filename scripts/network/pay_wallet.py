"""Local, non-custodial Solana wallet created during Nakshatra onboarding.

This module deliberately has no RPC code.  Creating and loading the wallet are
filesystem-only operations; network access belongs at the CLI boundary.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519


BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_BASE58_INDEX = {char: index for index, char in enumerate(BASE58_ALPHABET)}
_BIP39_WORDLIST_SHA256 = "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda"


def base58_encode(value: bytes) -> str:
    """Encode bytes with the Bitcoin/Solana base58 alphabet."""
    raw = bytes(value)
    zeroes = len(raw) - len(raw.lstrip(b"\0"))
    number = int.from_bytes(raw, "big")
    encoded = []
    while number:
        number, remainder = divmod(number, 58)
        encoded.append(BASE58_ALPHABET[remainder])
    return "1" * zeroes + "".join(reversed(encoded))


def base58_decode(value: str) -> bytes:
    """Decode Bitcoin/Solana base58 text, rejecting non-alphabet characters."""
    if not isinstance(value, str):
        raise TypeError("base58 input must be text")
    number = 0
    for char in value:
        try:
            digit = _BASE58_INDEX[char]
        except KeyError as exc:
            raise ValueError(f"invalid base58 character {char!r}") from exc
        number = number * 58 + digit
    zeroes = len(value) - len(value.lstrip("1"))
    body = (number.to_bytes((number.bit_length() + 7) // 8, "big") if number else b"")
    return b"\0" * zeroes + body


def bip39_mnemonic(entropy: bytes) -> str:
    """Encode standard-length entropy with the official BIP39 English list."""
    raw = bytes(entropy)
    if len(raw) not in {16, 20, 24, 28, 32}:
        raise ValueError("BIP39 entropy must contain 16, 20, 24, 28, or 32 bytes")
    wordlist_path = Path(__file__).with_name("bip39_english.txt")
    try:
        wordlist_bytes = wordlist_path.read_bytes()
        words = wordlist_bytes.decode("ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read the BIP39 English wordlist: {exc}") from exc
    if (hashlib.sha256(wordlist_bytes).hexdigest() != _BIP39_WORDLIST_SHA256
            or len(words) != 2048):
        raise ValueError("the BIP39 English wordlist failed its integrity check")

    entropy_bits = len(raw) * 8
    checksum_bits = entropy_bits // 32
    checksum = hashlib.sha256(raw).digest()[0] >> (8 - checksum_bits)
    encoded = (int.from_bytes(raw, "big") << checksum_bits) | checksum
    word_count = (entropy_bits + checksum_bits) // 11
    return " ".join(
        words[(encoded >> (11 * (word_count - index - 1))) & 0x7ff]
        for index in range(word_count)
    )


def _paths(home: Path) -> tuple[Path, Path, Path, Path]:
    pay_dir = Path(home) / ".nakshatra" / "pay"
    return pay_dir, pay_dir / "wallet.ed25519", pay_dir / "wallet.pub", pay_dir / "custody"


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _check_pay_dir(pay_dir: Path) -> None:
    try:
        info = pay_dir.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise SystemExit(f"cannot inspect pay wallet directory {pay_dir}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SystemExit(f"refusing {pay_dir}: pay wallet directory must be a regular directory")
    if info.st_uid != os.geteuid():
        raise SystemExit(f"refusing {pay_dir}: pay wallet directory is not owned by this user")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise SystemExit(f"refusing {pay_dir}: pay wallet directory must have mode 0700")


def _read_seed(path: Path) -> bytes:
    # O_NONBLOCK keeps a hostile FIFO/device at this path from hanging onboarding before fstat.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SystemExit(f"refusing unreadable wallet seed {path}: {exc}") from exc
    try:
        info = os.fstat(fd)
        mode = stat.S_IMODE(info.st_mode)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"refusing {path}: wallet seed must be a regular file")
        if info.st_uid != os.geteuid():
            raise SystemExit(f"refusing {path}: wallet seed is not owned by this user")
        if mode != 0o600:
            raise SystemExit(f"refusing {path}: wallet seed must be mode 0600 (found {mode:04o})")
        chunks = []
        while True:
            chunk = os.read(fd, 33)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(map(len, chunks)) > 32:
                break
        seed = b"".join(chunks)
    finally:
        os.close(fd)
    if len(seed) != 32:
        raise SystemExit(f"refusing {path}: wallet seed must be exactly 32 raw bytes")
    return seed


def _wallet_values(seed: bytes) -> tuple[bytes, str]:
    public_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
    return public_key, base58_encode(public_key)


def _read_public(path: Path, expected: str) -> None:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"refusing {path}: wallet address must be a regular file")
        stored = path.read_text(encoding="ascii").strip()
        decoded = base58_decode(stored)
    except SystemExit:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"refusing invalid wallet address file {path}: {exc}") from exc
    if len(decoded) != 32 or stored != expected:
        raise SystemExit(f"refusing {path}: wallet address does not match wallet.ed25519")


def _read_custody(path: Path) -> str:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"refusing {path}: wallet custody marker must be a regular file")
        mode = path.read_text(encoding="ascii").strip()
    except SystemExit:
        raise
    except (OSError, UnicodeError) as exc:
        raise SystemExit(f"refusing unreadable wallet custody marker {path}: {exc}") from exc
    if mode not in {"file", "broker"}:
        raise SystemExit(f"refusing {path}: wallet custody must be one word: file or broker")
    return mode


def preflight_wallet(home: Path) -> None:
    """Validate all wallet state that already exists, without writing anything."""
    pay_dir, seed_path, public_path, custody_path = _paths(home)
    _check_pay_dir(pay_dir)
    if not _path_present(seed_path):
        if _path_present(public_path) or _path_present(custody_path):
            raise SystemExit(f"refusing incomplete pay wallet at {pay_dir}: wallet.ed25519 is missing")
        return
    seed = _read_seed(seed_path)
    _public_key, address = _wallet_values(seed)
    if _path_present(public_path):
        _read_public(public_path, address)
    if _path_present(custody_path):
        _read_custody(custody_path)


def _write_new(path: Path, data: bytes, mode: int) -> None:
    """Atomically create a file without replacing a concurrent or existing file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    linked = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            os.fchmod(fd, mode)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(tmp, path)
        linked = True
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            if not linked:
                raise


def ensure_wallet(home: Path) -> dict:
    """Create the local pay wallet once, or validate and return the existing wallet."""
    home = Path(home)
    pay_dir, seed_path, public_path, custody_path = _paths(home)

    # A corrupt existing seed must stop the operation before mkdir, chmod, or any other write.
    preflight_wallet(home)
    if not _path_present(pay_dir):
        pay_dir.mkdir(parents=True, mode=0o700)
        os.chmod(pay_dir, 0o700)
    _check_pay_dir(pay_dir)

    if not _path_present(seed_path):
        seed = os.urandom(32)
        try:
            fd = os.open(seed_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            seed = _read_seed(seed_path)
        else:
            try:
                os.fchmod(fd, 0o600)
                view = memoryview(seed)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
    else:
        seed = _read_seed(seed_path)

    public_key, address = _wallet_values(seed)
    if _path_present(public_path):
        _read_public(public_path, address)
    else:
        try:
            _write_new(public_path, (address + "\n").encode("ascii"), 0o644)
        except FileExistsError:
            _read_public(public_path, address)

    if _path_present(custody_path):
        custody = _read_custody(custody_path)
    else:
        try:
            _write_new(custody_path, b"file\n", 0o644)
            custody = "file"
        except FileExistsError:
            custody = _read_custody(custody_path)

    return {"seed": seed, "public_key": public_key, "address": address, "custody": custody}


def load_wallet(home: Path) -> dict:
    """Load a complete wallet for the CLI with the same validation as onboarding."""
    home = Path(home)
    pay_dir, seed_path, public_path, custody_path = _paths(home)
    _check_pay_dir(pay_dir)
    if not _path_present(seed_path):
        raise SystemExit(f"pay wallet is missing at {seed_path}; run the signed join setup first")
    seed = _read_seed(seed_path)
    public_key, address = _wallet_values(seed)
    if not _path_present(public_path):
        raise SystemExit(f"pay wallet address is missing at {public_path}; re-run the signed join setup")
    _read_public(public_path, address)
    if not _path_present(custody_path):
        raise SystemExit(f"pay wallet custody marker is missing at {custody_path}; re-run the signed join setup")
    custody = _read_custody(custody_path)
    return {"seed": seed, "public_key": public_key, "address": address, "custody": custody}
