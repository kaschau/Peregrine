class IoVars:
    def __init__(self, primitives, exported=None, vectors=None, derive=None):
        # a primitive's value is what a result lacking it starts at; None marks one a result must hold
        self.primitives = dict(primitives)
        self.exported = list(self.primitives if exported is None else exported)
        self.vectors = {} if vectors is None else dict(vectors)
        self.derive = derive

    @property
    def privars(self):
        return list(self.primitives)

    @property
    def minimal(self):
        return IoVars(self.primitives, vectors=self.vectors)

    def data(self, blk, names):
        if self.derive:
            return self.derive(blk, names)
        prims = blk.q.get()
        return {n: prims[..., self.privars.index(n)] for n in names}

    def vectors_among(self, names):
        vectors = self.vectors.items()
        return {v: list(c) for v, c in vectors if all(n in names for n in c)}

    def scalars_among(self, names):
        vectors = self.vectors_among(names).values()
        invectors = {n for components in vectors for n in components}
        return [n for n in names if n not in invectors]
