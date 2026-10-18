from types import SimpleNamespace


# a reaction of a mechanism, its class its kind
class BaseReaction(SimpleNamespace):
    name = None
    # whether a third body's concentration enters the rate, whether through a falloff blend, and whether that blend is Troe's
    thirdbody = falloff = troe = False
    # the third body's share of the rate constant's order
    bodyorder = 0

    # the most the kind adds to ln k beyond its Arrhenius rate, every concentration at most e^logcmax
    @staticmethod
    def boost(logcmax):
        return 0.0


class ElementaryReaction(BaseReaction):
    name = "elementary"


class ThreeBodyReaction(BaseReaction):
    name = "three-body"
    thirdbody = True
    bodyorder = 1

    @staticmethod
    def boost(logcmax):
        return logcmax + 2.0


class LindemannReaction(BaseReaction):
    name = "lindemann"
    thirdbody = falloff = True

    @staticmethod
    def boost(logcmax):
        return 1.0


class TroeReaction(LindemannReaction):
    name = "troe"
    troe = True
