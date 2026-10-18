# not selectable by the config: each eos names its own
class BaseCaloricModel:
    def __init__(self, cfgsect):
        self.cfgsect = cfgsect
        # what it reads off each species
        self.needs = ()
