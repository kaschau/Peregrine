import numpy as np
import pytest

import peregrine as pg
from peregrine.backend.base.graph import Graph

from tests.gases import configure, primitives


def solver(backend):
    config = pg.Config()
    configure(config, "air", "navier-stokes")
    config["bcvals"]["walls"] = {"kind": "no-slp-adia-wall"}
    config["bcvals"]["slip"] = {"kind": "slp-adia-wall"}
    config["bcvals"]["unused"] = {"kind": "slp-adia-wall"}
    mesh = pg.mesher.CubeMesher(mbdims=[2, 1, 1], blkdims=[5, 4, 3], lengths=[2, 1, 1], bcnames={1: "walls"}, defaultbc="slip")
    return pg.get_solver(config, mesh, backend(config))


def bump(mb, rng):
    prims = []
    for blk in mb.blocks:
        q = primitives(mb, blk)
        q[..., 0] *= 1 + 0.05 * rng.random(q.shape[:3])
        q[..., 1:4] = 30 * rng.random(q.shape[:3] + (3,))
        q[..., 4] *= 1 + 0.1 * rng.random(q.shape[:3])
        prims.append(q)
    mb.set_prims(prims)


def held(mb, register):
    return [getattr(blk, register).get().copy() for blk in mb.blocks]


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_a_run_of_the_solvers_graphs_is_their_one_off_launch_to_the_bit(backend):
    mb = solver(backend)
    g = mb.graphs["rhs", "Q", "Q2"]
    # the viscous right-hand side runs around the gradients' message: its steps in the message's order
    steps = [str(item) for item in g.close().sequence if item.host]
    assert steps == ["grads-sent: expect", "grads-sent: copy_out", "grads-landed: send", "grads-landed: recv", "grads-landed: sent"]
    bump(mb, np.random.default_rng(5))
    mb.rhs(Q="Q", dQ="Q2")
    first = held(mb, "Q2")
    mb.rhs(Q="Q", dQ="Q2")
    assert same(held(mb, "Q2"), first)
    # the state moved under the graphs: their run is the one-off launch, to the bit
    for blk in mb.blocks:
        Q = blk.Q.get()
        Q[blk.interior + (0,)] *= 1.01
        Q[blk.interior + (4,)] *= 1.01
        blk.Q.set(Q)
    mb.postrhs()
    mb.graphs["prerhs", "Q", "Q2"].run()
    g.run()
    replay = held(mb, "Q2")
    mb.graphs["prerhs", "Q", "Q2"].once()
    g.once()
    assert same(replay, held(mb, "Q2")) and not same(replay, first)


def test_a_device_builds_each_stretch_once_and_replays_it(backend):
    mb = solver(backend)
    g = mb.graphs["rhs", "Q", "Q2"]
    if mb.backend.graphcls is Graph:
        pytest.skip("a host launches each time and builds nothing")
    assert g.built == {}
    bump(mb, np.random.default_rng(5))
    mb.rhs(Q="Q", dQ="Q2")
    built = dict(g.built)
    # three stretches: before the copy out, while the message flies, once landed
    assert len(built) == 3
    mb.rhs(Q="Q", dQ="Q2")
    assert g.built == built
    # a one-off launch stays outside the device graphs
    g.once()
    assert g.built == built


def test_a_rerun_advance_reads_the_step_where_the_kernels_run(backend):
    mb = solver(backend)
    bump(mb, np.random.default_rng(5))
    mb.rhs(Q="Q", dQ="Q2")
    adv = mb.graphs["advance", mb.regs, 0]
    mb.dt.set(5e-6)
    adv.run()
    # the first stage of tvd-rk3: Q1 = Q + dt Q2 over the interior, exact here, within contraction's rounding on a device
    for blk in mb.blocks:
        want = blk.Q.get()[blk.interior] + 5e-6 * blk.Q2.get()[blk.interior]
        assert np.allclose(blk.Q1.get()[blk.interior], want, rtol=1e-15, atol=0.0)
    five = held(mb, "Q1")
    mb.dt.set(7e-6)
    adv.run()
    seven = held(mb, "Q1")
    adv.once()
    assert same(seven, held(mb, "Q1")) and not same(seven, five)


def test_a_launch_is_one_kernel_over_every_block(backend):
    mb = solver(backend)
    g = mb.backend.graph("t")
    k = mb.kernels["fromQ"]("Q")
    g.add_launch("fromq", k, "interior")
    g.bind(mb)
    g.close()
    ((kernel, table, tiling),) = g.names["fromq"].groups[0]
    # the tag's maker is memoized: one kernel per register, however many graphs launch it
    assert kernel is k and mb.kernels["fromQ"]("Q") is k
    # a tiling is keyed by what makes a policy the same: another cells kernel over the interior shares it
    assert table.tiling(mb.kernels["fromQ"]("Q1"), "interior") is tiling
    assert table is mb.blkcoll.table and tiling.count == len(mb.blocks) == 2
    # a face flux is three kernels, one per axis, in one group
    g = mb.backend.graph("t")
    g.add_launch("flux", mb.kernels["advflux"]("Q"), "interior")
    g.bind(mb)
    g.close()
    (group,) = g.names["flux"].groups
    assert [str(kernel) for kernel, _, _ in group] == ["single.flux i", "single.flux j", "single.flux k"]


def test_the_boundaries_are_one_group_tiled_per_kind(backend):
    mb = solver(backend)
    g = mb.backend.graph("t")
    g.add_bcs("preadvflux", "Q")
    g.bind(mb)
    g.close()
    (group,) = g.names["preadvflux"].groups
    # one kernel per axis per entry, over that entry's interfaces of the axis, one range per halo layer
    counts, kinds = {}, {}
    for kernel, table, tiling in group:
        name = table.collection.name
        counts[name] = counts.get(name, 0) + tiling.count
        kinds.setdefault(name, set()).add(str(kernel))
    assert counts == {"walls": 1 * mb.ng, "slip": 9 * mb.ng, "unused": 0}
    assert kinds["walls"] == {f"no-slp-adia-wall.preadvflux {a}" for a in "ijk"}
    assert kinds["slip"] == kinds["unused"] == {f"slp-adia-wall.preadvflux {a}" for a in "ijk"}
    # an entry on no face launches nothing, and the group runs
    g.run()


def test_a_launch_after_a_send_or_a_landing_covers_what_that_made_valid(backend):
    mb = solver(backend)
    k = mb.kernels["fromQ"]("Q")
    g = mb.backend.graph("x")
    sent = g.add_send("q-sent", "Q")
    landed = g.add_recv("q-landed", "Q")
    # added in the opposite order to the one they run in: the edges, not the order of adding, place them
    far = g.add_launch("far", k, "all", after=("q-landed",))
    near = g.add_launch("near", k, "all", after=("q-sent",))
    g.bind(mb)
    g.close()
    # the message's steps by the two rules: the receives expected first and our sends waited for last, the copy out at the pack, the send and the receive at the landing, which runs as late as the edges allow
    sequence = [str(item) if item.host else item for item in g.sequence]
    assert sequence == ["q-sent: expect", sent, "q-sent: copy_out", near, "q-landed: send", "q-landed: recv", landed, far, "q-landed: sent"]
    # after the send the launch covers what the exchange filled, after the landing what the message brought: nothing, on one rank
    ((kernel, table, tiling),) = near.groups[0]
    assert kernel is k and tiling.name == "all filled by Q"
    ((kernel, table, tiling),) = far.groups[0]
    assert kernel is k and tiling.name == "all landed by Q" and tiling.count == 0 and tiling.tiles == 0
    assert mb.mpicolls == []
    # and the run exchanged the halos: the first block's high i halo is the second's first interior planes
    bump(mb, np.random.default_rng(3))
    a, b = mb.blocks
    Q = a.Q.get()
    Q[-1] = 0.0
    a.Q.set(Q)
    g.run()
    ng = mb.ng
    assert np.array_equal(a.Q.get()[-ng:, ng:-ng, ng:-ng], b.Q.get()[ng : 2 * ng, ng:-ng, ng:-ng])


def test_a_graph_orders_the_ready_nodes_earliest_added_first(backend):
    mb = solver(backend)
    k = mb.kernels["fromQ"]("Q")
    g = mb.backend.graph("order")
    a = g.add_launch("a", k, "interior")
    b = g.add_launch("b", k, "interior", after=("a",))
    c = g.add_launch("c", k, "interior")
    d = g.add_launch("d", k, "interior", after=("b", "c"))
    g.bind(mb)
    g.close()
    # the sequence takes the earliest added among the ready
    assert g.sequence == [a, b, c, d]
    # the graph is the backend's own, a kind of the base's
    assert type(g) is mb.backend.graphcls and isinstance(g, Graph)
    assert repr(g) == "order:\n" + "\n".join(f"  {n}: fromQ over interior" for n in "abcd")


def test_a_name_used_twice_or_followed_before_it_is_added_is_refused(backend):
    mb = solver(backend)
    k = mb.kernels["fromQ"]("Q")
    g = mb.backend.graph("names")
    g.add_launch("a", k, "interior")
    with pytest.raises(ValueError):
        g.add_launch("a", k, "interior")
    with pytest.raises(KeyError):
        g.add_launch("b", k, "interior", after=("c",))
    # a landing follows its send, so a bank is sent before it lands, and once
    with pytest.raises(KeyError):
        g.add_recv("q-landed", "Q")
    g.add_send("q-sent", "Q")
    with pytest.raises(ValueError):
        g.add_send("again", "Q")


def test_a_rerun_launch_reads_the_arrays_as_they_are(backend):
    mb = solver(backend)
    k = mb.integ.combine("Q1", ("Q", 1.0), step=False)
    g = mb.backend.graph("copy")
    g.add_launch("copy", k, "interior")
    g.bind(mb)
    g.run()
    blk = mb.blocks[0]
    assert np.array_equal(blk.Q1.get()[blk.interior], blk.Q.get()[blk.interior])
    Q = blk.Q.get()
    blk.Q.set(2.0 * Q)
    g.run()
    assert np.array_equal(blk.Q1.get()[blk.interior], 2.0 * Q[blk.interior])


def test_a_launch_of_a_reduction_gives_its_values(backend):
    mb = solver(backend)
    values = mb.launch(mb.kernels["cflmax"]("Q"), "interior")
    assert isinstance(values, np.ndarray) and values.shape == (3,)
    speeds = mb.cflmax()
    assert list(speeds) == ["acoustic", "convective", "combined"]
    assert np.array_equal(values, list(speeds.values()))
    # at rest: the acoustic speed over the oblique length of the fastest cell, nothing convective
    c = list(mb.sim.statefields).index("c")
    acoustic = max(
        (blk.state.get()[blk.interior][..., c] * np.sqrt((1.0 / blk.dijk.get()[blk.interior] ** 2).sum(axis=-1))).max()
        for blk in mb.blocks
    )
    assert speeds["acoustic"] == acoustic
    assert speeds["convective"] == 0.0 and speeds["combined"] == speeds["acoustic"]
