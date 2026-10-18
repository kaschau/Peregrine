def merge_disjoint(*dicts):
    merged = {}
    for d in dicts:
        taken = merged.keys() & d.keys()
        if taken:
            raise ValueError(f"{sorted(taken)} given twice")
        merged.update(d)
    return merged
