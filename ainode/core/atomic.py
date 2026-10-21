"""Private, durable replacement for credential-bearing configuration files.

The installer embeds this stdlib-only module verbatim; its regression test pins
that copy so installations can use even an older, explicitly pinned image.
"""

import os
import stat
import tempfile
from pathlib import Path


def atomic_private_write(path, text):
    """Exclusively create a 0600 inode, sync it, then atomically publish it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            if stat.S_IMODE(os.fstat(stream.fileno()).st_mode) != 0o600:
                raise OSError("Private configuration mode is not 0600")
        os.replace(temporary, path)
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise OSError("Published configuration mode is not 0600")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
