from peregrine.misc import subclass_where


class BasePlugin:
    name = None

    def __init__(self, cfgsect):
        self.cfgsect = cfgsect

    @classmethod
    def claim(cls, cfgsect, owner):
        plugins = {}
        for name, entry in cfgsect.items():
            kind = subclass_where(BasePlugin, name=name)
            # the section lists both levels' plugins; each level takes its own and leaves the rest to the other
            if issubclass(kind, cls):
                plugins[name] = kind(entry)
        return plugins
