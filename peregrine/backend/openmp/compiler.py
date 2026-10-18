import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from peregrine.backend.base.compiler import BaseCompiler


class OpenMPCompiler(BaseCompiler):
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    # the compile and link flags the probe found, per compiler, once a process
    openmp = {}

    def __init__(self, cxx):
        super().__init__()
        self.compiler = cxx
        compile, link = self._probe()
        self.flags = [
            "-O3",
            *(["-g"] if self.debug else ["-DNDEBUG"]),
            "-std=gnu++20",
            "-fPIC",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            *compile,
        ]
        self.link = ["-shared", *link]
        if sys.platform == "darwin":
            self.link += ["-undefined", "dynamic_lookup"]

    @property
    def key(self):
        return repr((self.compiler, self.flags, self.link))

    def command(self, source, out):
        return [self.compiler, *self.flags, str(source), *self.link, "-o", str(out)]

    def compile(self, text, out):
        self.run(self.command, text, out)

    def _probe(self):
        if self.compiler in self.openmp:
            return self.openmp[self.compiler]
        candidates = [(["-fopenmp"], ["-fopenmp"])]
        brew = shutil.which("brew")
        # Apple's clang takes OpenMP only through the front end, with Homebrew's libomp
        if sys.platform == "darwin" and brew:
            cmd = [brew, "--prefix", "libomp"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            prefix = result.stdout.strip()
            cflags = ["-Xclang", "-fopenmp", "-isystem", f"{prefix}/include"]
            ldflags = [f"-L{prefix}/lib", "-lomp"]
            candidates.append((cflags, ldflags))
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "openmp.cpp"
            probe.write_text(
                "#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n"
            )
            out = f"{tmp}/openmp"
            for compile, link in candidates:
                cmd = [self.compiler, *compile, *link, str(probe), "-o", out]
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0:
                    self.openmp[self.compiler] = (compile, link)
                    return compile, link
        raise EnvironmentError(f"{self.compiler} does not compile OpenMP")
