import pytest

import peregrine as pg

from tests.gases import configure


def solver(backend, physics, kind, **values):
    config = pg.Config()
    configure(config, "air", physics)
    config["bcvals"]["outer"] = {"kind": kind, **values}
    mesh = pg.mesher.CubeMesher(mbdims=[1, 1, 1], blkdims=[6, 6, 6], lengths=[1, 1, 1], defaultbc="outer")
    return pg.get_solver(config, mesh, backend(config))


def layout(mb, key):
    graph = mb.graphs[key]
    return [str(item) if item.host else [[str(k) for k, _, _ in grp] for grp in item.groups] for item in graph.sequence]


def ranges(mb, key, name):
    node = mb.graphs[key].names[name]
    return {str(kernel): tiling.name for group in node.groups for kernel, _, tiling in group}


def bcs(kind, hook):
    return [f"{kind}.{hook} {a}" for a in "ijk"]


def flux(template):
    return [f"{template}.flux {a}" for a in "ijk"]


def assert_readied(mb, key, Q, kind, after=()):
    # after: the launches following the boundaries, each a (tag, launch names) pair
    # the message's steps by the two rules: the receives expected first and our sends waited for last, the copy out at the pack; the send and the receive at the landing, as late as the edges allow
    assert layout(mb, key) == [
        f"{Q}-sent: expect",
        [],
        f"{Q}-sent: copy_out",
        [["fromQ"]],
        [bcs(kind, "preadvflux")],
        *[[group] for _, group in after],
        f"{Q}-landed: send",
        f"{Q}-landed: recv",
        [],
        [["fromQ"]],
        *[[group] for _, group in after],
        f"{Q}-landed: sent",
    ]
    # a cut is named for the family, whichever bank flies
    family = mb.columns[Q].name
    assert ranges(mb, key, "near") == {"fromQ": f"halos filled by {family}"}
    assert ranges(mb, key, "far") == {"fromQ": f"halos landed by {family}"}
    for tag, group in after:
        assert ranges(mb, key, f"{tag}-near") == {name: f"shallow less {family}" for name in group}
        assert ranges(mb, key, f"{tag}-far") == {name: f"shallow landed by {family}" for name in group}


def test_euler_lays_out_its_slots(backend):
    mb = solver(backend, "euler", "slp-adia-wall")
    for stage in mb.integ.stageregs(mb.regs):
        rhs = ("rhs", stage.Q, stage.dQ)
        # the walls write their own face flux between the flux and the apply
        assert layout(mb, rhs) == [[flux("single")], [bcs("slp-adia-wall", "postflux")], [["applyflux"]]]
        assert ranges(mb, rhs, "apply") == {"applyflux": "interior"}
        assert ranges(mb, rhs, "postflux") == {k: "postflux" for k in bcs("slp-adia-wall", "postflux")}
        assert_readied(mb, ("prerhs", stage.Q, stage.dQ), stage.Q, "slp-adia-wall")
        # a readying with nothing begun lays out the same, and nothing begins the derivative on its own for this physics
        assert_readied(mb, ("prerhs", stage.Q), stage.Q, "slp-adia-wall")
        assert layout(mb, ("begin_dQ", stage.Q, stage.dQ)) == []
        # after an advance the register's state is derived over the interior alone
        assert layout(mb, ("postrhs", stage.Qout)) == [[["fromQ"]]]
        assert ranges(mb, ("postrhs", stage.Qout), "fromq") == {"fromQ": "interior"}
    # a step begins with nothing of its own for this physics
    assert layout(mb, ("prestep", mb.regs[0], mb.regs[-1])) == []


@pytest.mark.parametrize("kind,values", [("no-slp-adia-wall", {}), ("slp-adia-wall", {}), ("sub-out-fp", {"p": 101325.0})])
def test_navier_stokes_lays_out_its_slots(backend, kind, values):
    mb = solver(backend, "navier-stokes", kind, **values)
    # a wall writes its faces' flux once the gradients have landed and fills no gradient hook; an exit sets the halo's gradients while they fly; a hook nothing fills launches nothing
    wall = kind.endswith("wall")
    postgrads = [] if wall else [bcs(kind, "postgrads")]
    postflux = [bcs(kind, "postflux")] if wall else []
    for stage in mb.integ.stageregs(mb.regs):
        rhs = ("rhs", stage.Q, stage.dQ)
        assert layout(mb, rhs) == [
            "grads-sent: expect",
            [],
            [["dqdxyz"]],
            postgrads,
            [],
            "grads-sent: copy_out",
            [flux("single")],
            [flux("alpha-damping")],
            "grads-landed: send",
            "grads-landed: recv",
            [],
            [flux("single")],
            [flux("alpha-damping")],
            postflux,
            [["applyflux"]],
            "grads-landed: sent",
        ]
        assert_readied(mb, ("prerhs", stage.Q, stage.dQ), stage.Q, kind, after=[("trans", ["transport"])])
        assert layout(mb, ("postrhs", stage.Qout)) == [[["fromQ"]]]
        # both fluxes alongside the gradients' flight: the faces the message does not reach, then the faces it brought
        for f, near, far in (("single", "adv", "advl"), ("alpha-damping", "diff", "diffl")):
            assert ranges(mb, rhs, near) == {f"{f}.flux {a}": "interior less grads" for a in "ijk"}
            assert ranges(mb, rhs, far) == {f"{f}.flux {a}": "interior landed by grads" for a in "ijk"}
        assert ranges(mb, rhs, "apply") == {"applyflux": "interior"}
        assert ranges(mb, rhs, "postflux") == ({k: "postflux" for k in bcs(kind, "postflux")} if wall else {})


def test_a_hook_outside_the_physics_is_refused(backend):
    mb = solver(backend, "euler", "slp-adia-wall")
    assert not any("grads" in str(k) for g in mb.graphs.values() for k in g.kernels)
    with pytest.raises(AttributeError):
        mb.apply_bcs("pregrads")
