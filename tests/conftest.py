import asyncio
import gc
import logging
from contextlib import suppress

import pytest

# 2026-05-26 drive-by — import-tolerant conftest so the hardening +
# SPKI test files (which need none of the petals-derived fixtures
# below) can be collected on a stripped-down venv without --noconftest.
# When psutil / hivemind are present, the cleanup_children fixture
# does its full work; when absent, the fixture no-ops and tests still
# pass. Keeps the existing tests/ layout (no need to migrate files
# into a tests/hardened/ subdir, which wouldn't actually escape this
# conftest anyway — pytest walks up the directory tree).
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

try:
    from hivemind.utils.crypto import RSAPrivateKey
    from hivemind.utils.logging import get_logger
    from hivemind.utils.mpfuture import MPFuture
    _HIVEMIND_AVAILABLE = True
    logger = get_logger(__name__)
except ImportError:
    RSAPrivateKey = None
    MPFuture = None
    _HIVEMIND_AVAILABLE = False
    logger = logging.getLogger(__name__)


@pytest.fixture
def event_loop():
    """
    This overrides the ``event_loop`` fixture from pytest-asyncio
    (e.g. to make it compatible with ``asyncio.subprocess``).

    This fixture is identical to the original one but does not call ``loop.close()`` in the end.
    Indeed, at this point, the loop is already stopped (i.e. next tests are free to create new loops).
    However, finalizers of objects created in the current test may reference the current loop and fail if it is closed.
    For example, this happens while using ``asyncio.subprocess`` (the ``asyncio.subprocess.Process`` finalizer
    fails if the loop is closed, but works if the loop is only stopped).
    """

    yield asyncio.get_event_loop()


@pytest.fixture(autouse=True, scope="session")
def cleanup_children():
    yield

    if _HIVEMIND_AVAILABLE:
        with RSAPrivateKey._process_wide_key_lock:
            RSAPrivateKey._process_wide_key = None

    gc.collect()  # Call .__del__() for removed objects

    if _PSUTIL_AVAILABLE:
        children = psutil.Process().children(recursive=True)
        if children:
            logger.info(f"Cleaning up {len(children)} leftover child processes")
            for child in children:
                with suppress(psutil.NoSuchProcess):
                    child.terminate()
            psutil.wait_procs(children, timeout=1)
            for child in children:
                with suppress(psutil.NoSuchProcess):
                    child.kill()

    if _HIVEMIND_AVAILABLE:
        MPFuture.reset_backend()
