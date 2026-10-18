class Struct:
    def __init__(self, name, fields, arrays=None, base=None):
        self.name, self.fields = name, tuple(fields)
        self.arrays, self.base = dict(arrays or {}), base
