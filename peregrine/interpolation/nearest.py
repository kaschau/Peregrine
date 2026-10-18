from peregrine.interpolation.base import BaseInterpolator


class NearestInterpolator(BaseInterpolator):
    name = "nearest"
    # every value is one it was given, so there is no new extremum to clip
    overshoots = False

    def mapper(self, spts, tpts):
        # built once and thrown away, so the cheap build beats the tighter tree, and the query threads
        from scipy.spatial import cKDTree

        tree = cKDTree(spts, balanced_tree=False, compact_nodes=False)
        _, nearest = tree.query(tpts, workers=-1)
        return lambda values: values[nearest]
