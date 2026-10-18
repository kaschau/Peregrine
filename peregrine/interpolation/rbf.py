from peregrine.interpolation.base import BaseInterpolator


class RbfInterpolator(BaseInterpolator):
    name = "rbf"

    def __init__(self, basis="linear", smooth=0.5, **kwargs):
        super().__init__(**kwargs)
        self.basis = basis
        self.smooth = smooth

    def mapper(self, spts, tpts):
        def onto(values):
            # scipy is a tool's dependency, not the solver's
            from scipy import interpolate

            rbf = interpolate.Rbf(
                *spts.T, values, function=self.basis, smooth=self.smooth
            )
            return rbf(*tpts.T)

        return onto
