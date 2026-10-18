from functools import cache

from mpi4py import MPI

from peregrine.multiblock.collections import (
    BaseMessage,
    BcInterfaceCollection,
    BlockCollection,
    InteriorInterfaceCollection,
    MpiInterfaceCollection,
    RecvPool,
    SendPool,
)
from peregrine.multiblock.restart import Restart
from peregrine.multiblock.block import SolverBlock
from peregrine.backend import Scalar
from peregrine.launch.arrays import ScalarSpec
from peregrine.codegen import KernelProvider
from peregrine.misc import get_comm_rank_size, merge_disjoint, subclass_where
from peregrine.plugins.solver import BaseSolverPlugin


class Solver(Restart):
    ###########################################################################
    # Making a case
    ###########################################################################
    def __init__(self, cfg, mesh, backend, sim, integ, state=None):
        super().__init__(len(sim.iovars.privars), backend=backend)
        self.cfg = cfg
        # a mesher, or a grid reader with the partition it took, which the results name
        self.mesh = mesh
        self.sim, self.integ = sim, integ
        # the clock: step count, time and step size, the last two held where the kernels run
        self.nrt = 0
        self.tme = Scalar(backend, "tme", 0.0)
        self.dt = Scalar(backend, "dt")
        # the registers readied for a right-hand side not yet taken, each with the derivative begun into, None when nothing was
        self.readied = {}
        self._decl_plugins()
        self._decl_arrays()
        self._decl_tplargs()
        self._decl_kerns()
        self._decl_halo()
        self._decl_bckinds()
        mesh.fill(self)
        self._check_bcs()
        self._decl_colls()
        self._decl_comm()
        self._make_halo()
        self.make_metrics()
        # after the metrics, which a boundary's values may be made from
        for collection in self.bccolls.values():
            collection.start(self.tme)
        self._make_graphs()
        self._compile()
        self._set_state(state)
        for plugin in self.plugins.values():
            plugin.start(self)

    def _new_blk(self, nblki):
        return SolverBlock(nblki, self)

    ###########################################################################
    # What the simulator and the integrator ask for
    ###########################################################################
    def _decl_plugins(self):
        cfgsect = self.cfg["plugins"]
        self.plugins = BaseSolverPlugin.claim(cfgsect, self)
        taken = set(self.plugins) | set(self.sim.plugins)
        unclaimed = sorted(set(cfgsect) - taken)
        if unclaimed:
            raise ValueError(f"{unclaimed}: no part of this case takes these plugins")

    def _decl_arrays(self):
        theirs = (plugin.arrays(self) for plugin in self.plugins.values())
        arrays = merge_disjoint(self.sim.arrays(), *theirs)
        soln = self.sim.solution
        arrays[soln]["nregs"] = self.integ.nregs
        for name, spec in arrays.items():
            self.decl_array(name, **spec)
        self.regs = tuple(self.arrays[soln].banks)

    def _decl_kerns(self):
        theirs = (self.sim.decl_kerns(self.provider), self.integ.decl_kerns(self.provider))
        # memoized in place, in the dicts the recipes read: a tag asked for a register gives every graph the same kernel, built once
        for kernels in theirs:
            for tag, maker in kernels.items():
                kernels[tag] = cache(maker)
        self.kernels = {**theirs[0], **theirs[1]}

    def _decl_tplargs(self):
        theirs = (self.backend.tplargs(), self.sim.tplargs())
        scalars = {s.name: ScalarSpec(s) for s in (self.dt, self.tme)}
        columns = merge_disjoint(self.columns, scalars)
        # what the kernels declare locals of, beyond what the arrays are made of, and the constants their bodies name
        mine = {"arrays": columns, "structs": self.sim.structs(), "constants": self.sim.constants()}
        self.tplargs = merge_disjoint(*theirs, mine)
        self.provider = KernelProvider(self.backend, self.tplargs)

    def _decl_halo(self):
        reach = 0
        for maker in self.sim.kernels.values():
            for group in maker(self.regs[0]).groups:
                for k in group:
                    reach = max(reach, k.stencil)
        # one layer at least, for the boundaries' own cells; from here the provider refuses a deeper reach
        self.ng = self.provider.ng = max(1, reach)

    def _make_graphs(self):
        self.graphs = {}
        sim, integ = self.sim, self.integ
        solution, derivative = self.regs[0], self.regs[-1]
        self._fill(sim, "prestep", solution, derivative)
        self._fill(sim, "poststep")
        self._fill(sim, "initialize")
        self._fill(sim, "postrhs", solution)
        for n, stage in enumerate(integ.stageregs(self.regs)):
            self._fill(sim, "prerhs", stage.Q)
            self._fill(sim, "prerhs", stage.Q, stage.dQ)
            self._fill(sim, "begin_dQ", stage.Q, stage.dQ)
            # the apply adds to what the begin slot wrote, or writes when it is empty
            begun = bool(self.graphs["begin_dQ", stage.Q, stage.dQ].nodes)
            self._fill(sim, "rhs", stage.Q, stage.dQ, begun=begun)
            self._fill(integ, "advance", self.regs, n)
            self._fill(sim, "postrhs", stage.Qout)
        # the apply adds to the derivative only what the physics begins, so a plugin's own source goes after the apply, in rhs
        names = {name for name, *_ in self.graphs} - {"begin_dQ"}
        for plugin in self.sim.plugins.values():
            adds = plugin.launches(self.provider)
            unknown = sorted(adds.keys() - names)
            if unknown:
                msg = f"{plugin.name}: no slot {unknown}"
                raise ValueError(f"{msg}; the slots are {sorted(names)}")
            for (name, *_), g in self.graphs.items():
                for n, (kernel, over) in enumerate(adds.get(name, ())):
                    g.add_launch(f"{plugin.name}-{n}", kernel, over, after=g.sinks())
        for g in self.graphs.values():
            g.bind(self)
            g.close()

    def _fill(self, owner, name, *registers, **context):
        if (name, *registers) in self.graphs:
            return
        g = self.backend.graph(name)
        getattr(owner, f"{name}_graph")(g, *registers, **context)
        self.graphs[name, *registers] = g

    def _compile(self):
        kernels = [k for g in self.graphs.values() for k in g.kernels]
        exchanges = (self.interiorcoll, *self.mpicolls)
        kernels += [k for c in exchanges for k in c.placed.values()]
        self.provider.build(kernels)

    ###########################################################################
    # The grid, once every block is on its rank
    ###########################################################################

    # a boundary no entry of bcvals names is selected by no collection, and would silently be of no kind
    def _check_bcs(self):
        for blk, interface in self.intfaces():
            if interface.nbr is not None:
                continue
            where = f"block {blk.nblki} interface {interface.nface}"
            if interface.bcname is None:
                raise ValueError(
                    f"{where} is a boundary the grid did not name; name it "
                    "-- a mesher's bcnames or defaultbc, a translator's "
                    "names -- and say what it is in bcvals"
                )
            if interface.bcname not in self.cfg["bcvals"]:
                raise ValueError(
                    f"{where} is named {interface.bcname!r}, which bcvals "
                    f"does not have; it has {sorted(self.cfg['bcvals'])}"
                )

    def _make_halo(self):
        self.gen_halo()
        g = self.backend.graph("unify")
        g.add_send("nodes-sent", "nodes")
        g.add_recv("nodes-landed", "nodes")
        g.bind(self)
        g.run()
        self.move_periodics()

    ###########################################################################
    # Communication: a halo exchange per exchanged array
    ###########################################################################
    def _decl_comm(self):
        kind = self.cfg["halo-exchange"]["kind"]
        msgcls = subclass_where(BaseMessage, name=kind)
        mpis = self.mpicolls
        self.msgs = {}
        # the tag is the family's place among the exchanged, which every rank declares in the same order
        for tag, spec in enumerate(self.xchgarrays):
            send = SendPool(spec, mpis, self.backend)
            recv = RecvPool(spec, mpis, self.backend)
            self.msgs[spec.name] = msgcls(self.backend, send, recv, tag)

    ###########################################################################
    # The boundaries: the physics' kind on each entry of bcvals
    ###########################################################################
    def _decl_bckinds(self):
        self.bckinds = []
        # every rank makes every kind, those it has no interface of too, so a bad entry is refused on every rank at once
        for name, given in sorted(self.cfg["bcvals"].items()):
            kindcls = subclass_where(self.sim.bcscls, kind=given["kind"])
            self.bckinds.append(kindcls(name, given, self.sim))

    ###########################################################################
    # The collections: the blocks, and their interfaces by kind
    ###########################################################################
    def _decl_colls(self):
        blocks, backend = self.blocks, self.backend
        self.blkcoll = BlockCollection(blocks, backend)
        bcs = [BcInterfaceCollection(k, blocks, backend) for k in self.bckinds]
        self.bccolls = {c.name: c for c in bcs}
        args = (blocks, backend, self.provider, self.xchgarrays)
        self.interiorcoll = InteriorInterfaceCollection(*args)
        ranks = sorted({f.nbrrank for _, f in self.intfaces() if f.offrank})
        self.mpicolls = [MpiInterfaceCollection(rank, *args) for rank in ranks]

    @property
    def xchgarrays(self):
        return [spec for spec in self.arrays.values() if spec.exchanged]

    ###########################################################################
    # The slots: the graphs the physics and the integrator filled, run here
    ###########################################################################
    def spec(self, bank):
        return self.columns[bank]

    def message(self, bank):
        return self.msgs[self.columns[bank].name]

    def prestep(self):
        Q, dQ = self.regs[0], self.regs[-1]
        self._ready(Q, dQ)
        self.graphs["prestep", Q, dQ].run()

    def poststep(self):
        self.graphs["poststep",].run()

    def initialize(self):
        self.graphs["initialize",].run()

    def prerhs(self, Q, dQ=None):
        self._update_bcs()
        key = ("prerhs", Q) if dQ is None else ("prerhs", Q, dQ)
        self.graphs[key].run()
        self._begun(Q, dQ)

    def _begun(self, Q, dQ):
        self._spent(dQ)
        self.readied[Q] = dQ

    def _spent(self, dQ):
        if dQ is not None:
            self.readied = {q: d for q, d in self.readied.items() if d != dQ}

    def _ready(self, Q, dQ):
        standing = self.readied.get(Q, False)
        if standing == dQ:
            return
        if standing is None:
            self.graphs["begin_dQ", Q, dQ].run()
            self._begun(Q, dQ)
        else:
            self.prerhs(Q, dQ)

    def rhs(self, *, Q, dQ):
        self._ready(Q, dQ)
        self.graphs["rhs", Q, dQ].run()
        self._spent(dQ)

    def advance(self, n):
        Qout = self.integ.stageregs(self.regs)[n].Qout
        self.graphs["advance", self.regs, n].run()
        self.readied.pop(Qout, None)
        self.graphs["postrhs", Qout].run()

    def postrhs(self, Q=None):
        Q = Q or self.regs[0]
        self.graphs["postrhs", Q].run()
        self.prerhs(Q)

    def launch(self, kernel, over):
        g = self.backend.graph(str(kernel))
        g.add_launch("launch", kernel, over)
        g.bind(self)
        return g.once()

    def cflmax(self):
        kernel = self.kernels["cflmax"](Q=self.regs[0])
        speeds = self.launch(kernel, "interior")
        get_comm_rank_size()[0].Allreduce(MPI.IN_PLACE, speeds, op=MPI.MAX)
        return dict(zip(self.sim.cflspeeds, speeds))

    def from_prims(self):
        self.launch(self.kernels["fromprims"](Q=self.regs[0]), "all")
        self.initialize()
        self.readied.clear()
        self.postrhs()
        for blk in self.blocks:
            blk.q = None
        self._forget("q")

    ###########################################################################
    # The state
    ###########################################################################
    def _set_state(self, state):
        if state is None:
            values = self.sim.ics()
            for blk in self.blocks:
                q = blk.q.get()
                q[...] = values
                blk.q.set(q)
        else:
            state.fill(self, self.sim.iovars)
            self.nrt = state.nrt
            self.tme.set(state.tme)
            # a result holds the interior; a halo starts as the nearest cell
            for blk in self.blocks:
                blk.pad_halo("q")
        self.from_prims()

    def set_prims(self, prims):
        for blk, values in zip(self.blocks, prims):
            blk.replace("q", values)
        self._forget("q")
        self.from_prims()

    def _forget(self, name):
        exchanges = (self.interiorcoll, *self.mpicolls)
        bcs = self.bccolls.values()
        for collection in (self.blkcoll, *bcs, *exchanges):
            collection.forget(name)

    ###########################################################################
    # The boundaries, on the interfaces
    ###########################################################################
    def apply_bcs(self, hook, Q=None):
        self._update_bcs()
        g = self.backend.graph(f"{hook} bcs")
        g.add_bcs(hook, Q or self.regs[0])
        g.bind(self)
        g.once()

    def _update_bcs(self):
        for kind in self.bckinds:
            kind.update()

    ###########################################################################
    # Running: the steps, the loop, the counts and the banner
    ###########################################################################
    def step(self, dt):
        # the stages move the clock to their own times; the step ends it at start + dt
        start = self.tme.get()
        # queued behind the last step's kernels, ahead of this one's
        self.dt.set(dt)
        self.prestep()
        self.integ.step(self, dt)
        self.poststep()
        self.nrt += 1
        self.tme.set(start + dt)

    def run(self):
        plugins = self.plugins.values()
        try:
            for _ in range(self.cfg["simulation"]["niter"]):
                self.step(self.integ.controller.stepsize(self))
                for plugin in plugins:
                    if plugin.due(self):
                        plugin(self)
        finally:
            for plugin in plugins:
                plugin.finalize(self)

    @property
    def ncells(self):
        return sum(b.ncells for b in self.blocks)

    def __repr__(self):
        string = f"  Blocks: {len(self.blocks)} of {self.nblocks}\n"
        pname = self.mesh.pname
        if pname:
            string += f"  Partition: {pname} (ranks x ranks per node)\n"
        string += self.sim.report() + self.integ.report()
        for graph in self.graphs.values():
            lines = repr(graph).split("\n")
            string += "\n".join("  " + line for line in lines) + "\n"
        return string
