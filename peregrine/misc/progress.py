import shutil
import sys


class Progress:
    length = 31
    dude = "¯\\_(ツ)_/¯"

    def __init__(self, total, quiet=False):
        self.total, self.quiet, self.done = total, quiet, 0
        self.drawn = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if not self.quiet and self.done < self.total:
            sys.stdout.write("\n")

    def step(self, note=""):
        self.at(self.done + 1, note)

    def at(self, done, note=""):
        self.done = done
        if self.quiet:
            return
        fraction = min(done / float(self.total), 1.0)
        completed = int(round(self.length * fraction))
        bar = "_" * completed + self.dude + "_" * (self.length - completed)
        line = f"[{bar}] {round(100.0 * fraction, 1)}% ...{note}"
        # a line reaching the last column wraps and \r returns to the wrapped tail, so stay short of the edge; ツ is two columns wide
        line = line[: shutil.get_terminal_size().columns - 2]
        # blank whatever a longer previous draw left past the end of this one
        pad = " " * max(self.drawn - len(line), 0)
        self.drawn = len(line)
        end = "\n" if fraction >= 1.0 else "\r"
        sys.stdout.write(f"{line}{pad}{end}")
        sys.stdout.flush()
