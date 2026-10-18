import contextlib
import fcntl
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory


class BaseCompiler:
    # what a built unit is, a library or a module, by its file's suffix
    suffix = None

    def __init__(self):
        self.cache = Path(os.environ.get("PEREGRINE_CACHE", Path.home() / ".cache" / "peregrine"))
        # the device architecture the kernels are compiled for; none on a host
        self.arch = None
        # debug info and the kernels' asserts, for looking into a problem
        self.debug = bool(os.environ.get("PEREGRINE_JIT_DEBUG"))

    @property
    def key(self):
        raise NotImplementedError

    def compile(self, text, out):
        raise NotImplementedError

    # written beside the build so the errors of a unit that did not compile point at a file that stays
    def source(self, text, out):
        path = self.cache / f"{out.stem}.cpp"
        self._write(path, text)
        return path

    def run(self, command, text, out):
        source = self.source(text, out)
        result = subprocess.run(command(source, out), capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{source} did not compile:\n{result.stderr}")
        source.unlink()

    def build(self, stem, text):
        # a unit includes no file of the package, so its text and the toolchain are a build's whole input
        key = hashlib.sha256(text.encode())
        key.update(self.key.encode())
        out = self.cache / f"{stem}-{key.hexdigest()[:16]}{self.suffix}"
        if out.exists():
            return out
        self.cache.mkdir(parents=True, exist_ok=True)
        try:
            with open(out.with_suffix(".lock"), "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not out.exists():
                    # built aside and moved in whole, so a reader never sees a partial file
                    with TemporaryDirectory(dir=self.cache) as tmp:
                        built = Path(tmp) / out.name
                        self.compile(text, built)
                        shutil.move(built, out)
        finally:
            # whoever held the lock removes it, built or not
            with contextlib.suppress(FileNotFoundError):
                os.remove(out.with_suffix(".lock"))
        return out

    # written aside and moved in whole, so a reader never sees a partial file
    @staticmethod
    def _write(path, text):
        if path.exists():
            return
        with NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
            f.write(text)
        shutil.move(f.name, path)
