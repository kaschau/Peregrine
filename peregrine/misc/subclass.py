def subclasses(cls):
    direct = cls.__subclasses__()
    return direct + [g for s in direct for g in subclasses(s)]


def subclass_where(cls, **kwargs):
    for s in subclasses(cls):
        if all(getattr(s, k, None) == v for k, v in kwargs.items()):
            return s
    attrs = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    raise KeyError(f"No subclass of {cls.__name__} with {attrs}.")
