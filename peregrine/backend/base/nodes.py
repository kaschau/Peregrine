class Step:
    host = True

    def __init__(self, step, wait, node):
        self.step, self.wait, self.node = step, wait, node

    def __str__(self):
        return f"{self.node.name}: {self.step.__name__}"

    def run(self, g):
        self.step()


class BaseNode:
    # the bank whose halos this node sends or lands, for a launch that follows it
    bank = None
    # whether the halos are those the message brought, or those the exchange filled on this rank
    landed = False
    # the hook this node runs the boundaries at, for a launch that follows it
    hook = None
    # True for a node whose steps wait on the host, placed as late as its edges allow
    wait = False
    # a node is launches, not a host step
    host = False

    def __init__(self, name):
        self.name = name
        self.groups = []
        self.presteps, self.poststeps = [], []

    def pre(self, step, wait=False):
        self.presteps.append(Step(step, wait, self))

    def post(self, step, wait=False):
        self.poststeps.append(Step(step, wait, self))

    def bind(self, to, g):
        raise NotImplementedError

    def run(self, g):
        return g._launch([self])


class LaunchNode(BaseNode):
    def __init__(self, name, kernel, over):
        super().__init__(name)
        self.kernel, self.over = kernel, over

    def bind(self, to, g):
        messages = [a for a in g.after[self] if a.bank is not None]
        if not messages:
            self.groups = to.blkcoll.launches(self.kernel, self.over)
            return
        (message,) = messages
        spec = to.spec(message.bank)
        if message.landed:
            self.groups = to.blkcoll.launches(self.kernel, self.over, spec, behind=to.mpicolls)
            return
        # the boundaries' halos are valid only after a hook has run before this node
        hooked = any(a.hook is not None for a in g.ancestors(self))
        at = [*to.mpicolls, *([] if hooked else to.bccolls.values())]
        self.groups = to.blkcoll.launches(self.kernel, self.over, spec, less=at)


class BcsNode(BaseNode):
    def __init__(self, hook, Q):
        super().__init__(hook)
        self.hook, self.Q = hook, Q

    def bind(self, to, g):
        # one group: the boundaries' interfaces are disjoint
        group = [launch for c in to.bccolls.values() for launch in c.launches(self.hook, self.Q)]
        self.groups = [group] if group else []


class SendNode(BaseNode):
    def __init__(self, name, bank):
        super().__init__(name)
        self.bank = bank

    def bind(self, to, g):
        msg = to.message(self.bank)
        self.groups = [*(group for c in to.mpicolls for group in c.packs(self.bank)), *to.interiorcoll.lands("fill", self.bank)]
        self.pre(msg.expect)
        self.post(msg.copy_out)


class RecvNode(BaseNode):
    landed = True
    wait = True

    def __init__(self, name, bank):
        super().__init__(name)
        self.bank = bank

    def bind(self, to, g):
        msg = to.message(self.bank)
        self.groups = [group for c in to.mpicolls for group in c.lands("unpack", self.bank)]
        self.pre(msg.send, wait=True)
        self.pre(msg.recv, wait=True)
        self.post(msg.sent, wait=True)
