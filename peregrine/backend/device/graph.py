from peregrine.backend.base.graph import Graph


class DeviceGraph(Graph):
    def __init__(self, name, backend):
        super().__init__(name, backend)
        # the device graph built of each stretch, by its place in the sequence, once it has run
        self.built = {}

    def stretches(self):
        items, nodes = [], []
        for item in self.close().sequence:
            if not item.host:
                nodes.append(item)
                continue
            if nodes:
                items.append(nodes)
                nodes = []
            items.append(item)
        if nodes:
            items.append(nodes)
        return items

    def run(self):
        for n, item in enumerate(self.stretches()):
            if not isinstance(item, list):
                item.run(self)
                continue
            if n not in self.built:
                self.built[n] = self.build(item)
            self.built[n].launch(self.backend.kernels)

    # a node named from an earlier stretch is done before this one launches on the same stream, so only this stretch's ends are edges
    def build(self, nodes):
        graph = self.backend.driver.graph()
        # what each node ends with: its last group's launches, or what it followed when it launched nothing
        ends = {}
        for node in nodes:
            # each end once, in order: two named nodes may end in the same launches
            deps = list(dict.fromkeys(end for named in self.after[node] if named in ends for end in ends[named]))
            for group in node.groups:
                # the group's launches as nodes, those that launch anything
                made = [n for n in (k.node(graph, deps, table, tiling) for k, table, tiling in group) if n is not None]
                if made:
                    deps = made
            ends[node] = deps
        return graph.instantiate()
