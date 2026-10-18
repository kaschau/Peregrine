from peregrine.backend.base.nodes import BcsNode, LaunchNode, RecvNode, SendNode


class Graph:
    def __init__(self, name, backend):
        self.name, self.backend = name, backend
        # every node in the order added, by name, and the edges by node
        self.nodes, self.names, self.after = [], {}, {}
        # the send of each bank, which its landing follows
        self.sends = {}
        # the nodes in the order they run, and the sequence of nodes and host steps run, once closed
        self.order = self.sequence = None

    def add_launch(self, name, kernel, over, after=()):
        return self._add(LaunchNode(name, kernel, over), after)

    def add_bcs(self, hook, Q, after=()):
        return self._add(BcsNode(hook, Q), after)

    def add_send(self, name, bank, after=()):
        if bank in self.sends:
            raise ValueError(f"{self.name}: {bank!r} is sent twice")
        self.sends[bank] = node = self._add(SendNode(name, bank), after)
        return node

    def add_recv(self, name, bank, after=()):
        return self._add(RecvNode(name, bank), (*after, self.sends[bank].name))

    def _add(self, node, after):
        if self.order is not None:
            raise ValueError(f"{self.name}: closed, nothing more runs in it")
        if node.name in self.names:
            raise ValueError(f"{self.name}: two nodes named {node.name!r}")
        self.after[node] = tuple(self.names[n] for n in after)
        self.names[node.name] = node
        self.nodes.append(node)
        return node

    def ancestors(self, node):
        found, pending = set(), list(self.after[node])
        while pending:
            a = pending.pop()
            if a not in found:
                found.add(a)
                pending += self.after[a]
        return found

    def sinks(self):
        followed = {a for after in self.after.values() for a in after}
        return tuple(n.name for n in self.nodes if n not in followed)

    def bind(self, to):
        for node in self.nodes:
            node.bind(to, self)

    @property
    def kernels(self):
        return [kernel for node in self.nodes for group in node.groups for kernel, _, _ in group]

    # a prestep that does not wait is hoisted before the first node and a poststep that waits sunk after the last, so a message is in flight for the whole graph
    def close(self):
        if self.order is not None:
            return self
        self.order = self._order()
        early = [s for node in self.order for s in node.presteps if not s.wait]
        late = [s for node in self.order for s in node.poststeps if s.wait]
        self.sequence = list(early)
        for node in self.order:
            self.sequence += [s for s in node.presteps if s.wait]
            self.sequence.append(node)
            self.sequence += [s for s in node.poststeps if not s.wait]
        self.sequence += late
        return self

    def _order(self):
        index = {node: n for n, node in enumerate(self.nodes)}
        done, order, pending = set(), [], list(self.nodes)
        while pending:
            ready = [n for n in pending if all(a in done for a in self.after[n])]
            if not ready:
                raise ValueError(f"{self.name}: a node follows one that never runs before it")
            node = min(ready, key=lambda n: (n.wait, index[n]))
            order.append(node)
            done.add(node)
            pending.remove(node)
        return order

    def run(self):
        return self.once()

    # the sequence run directly, outside any device graph a backend builds in run()
    def once(self):
        self.close()
        result = None
        for item in self.sequence:
            result = item.run(self)
        return result

    def _launch(self, nodes):
        result = None
        for node in nodes:
            for group in node.groups:
                for kernel, table, tiling in group:
                    result = kernel(table, tiling)
        return result

    def __repr__(self):
        lines = [f"{self.name}:"]
        for item in self.close().sequence:
            if item.host:
                lines.append(f"  host: {item}")
                continue
            launches = ", ".join(f"{k} over {t.name}" for group in item.groups for k, _, t in group)
            lines.append(f"  {item.name}: {launches}" if launches else f"  {item.name}")
        return "\n".join(lines)
