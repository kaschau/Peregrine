import itertools

import pytest

import peregrine as pg

from peregrine.config import ConfigError
from peregrine.simulator import get_simulator
from peregrine.simulator.mixture.chemistry import FiniteRateModel
from peregrine.simulator.mixture import (
    BaseEosModel,
    BaseMixingRuleModel,
    BaseSpeciesDiffusionModel,
    BaseTransportModel,
    Mixture,
    Species,
    database,
    get_mixture,
)
from peregrine.misc import subclass_where, subclasses
from peregrine.simulator.mixture.eos import RealGasModel
from peregrine.simulator.mixture.parser import CanteraParser

air = ["N2", "O2", "AR"]


def mechanismPath(name):
    import peregrine.simulator.mixture as mix
    from pathlib import Path

    return str(Path(mix.__file__).parent / "database" / "mechanisms" / name)


# the range every fit in these tests is made over
T = (300.0, 3000.0)


def section(mixture, **choices):
    sect = dict(pg.Config()["mixture"])
    sect.update({"species": mixture, "T-range": T, **choices})
    return sect


def test_species_list_from_library():
    m = Mixture(section(air, eos="tpg"))
    assert m.spnames == air
    assert m.ns == 3
    assert m.reactions == []
    # molecular weight is derived from the composition, never stored
    assert m.species["N2"]["MW"] == pytest.approx(28.014, rel=1e-4)


def test_every_eos_and_transport_is_selectable():
    for name in ("cpg", "tpg", "real-gas"):
        assert subclass_where(BaseEosModel, name=name).name == name
    for name in ("constant", "kinetic-theory", "chung", "entropy-scaling"):
        assert subclass_where(BaseTransportModel, name=name).name == name


def test_the_library_carries_every_entropy_scaling_row():
    for name, entry in database("species").items():
        assert len(entry["resmu"]) == 3, name
        assert len(entry["reskappa"]) == 4, name


# the dense transport models subclass kinetic theory, so binary takes them; takahashi needs the real gas as well
def test_a_dense_model_takes_the_binary_diffusion():
    Mixture(section(air, eos="real-gas", trans="entropy-scaling", diffusion="binary"))
    Mixture(section(air, eos="real-gas", trans="chung", diffusion="takahashi"))
    with pytest.raises(ValueError, match="takahashi needs RealGasModel"):
        Mixture(section(air, eos="tpg", trans="kinetic-theory", diffusion="takahashi"))
    with pytest.raises(ValueError, match="binary needs KineticTheoryModel"):
        Mixture(section(air, eos="cpg", trans="constant", diffusion="binary"))


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError):
        Mixture(section(air, eos="notAnEos"))
    with pytest.raises(KeyError):
        Mixture(section(air, eos="tpg", trans="notATransport"))


def test_missing_property_names_the_species():
    m = Mixture(section(air, eos="tpg"))
    broken = {k: Species(k, dict(sp.data)) for k, sp in m.species.items()}
    broken["N2"].data.pop("janaf")
    with pytest.raises(ValueError, match="janaf|NASA7 for N2"):
        subclass_where(BaseEosModel, name="tpg")(section(air)).check(broken)


def test_mechanism_wins_over_library():
    m = Mixture(section("GRI30.yaml", eos="tpg"))
    lib = Species.library()
    gri = CanteraParser(mechanismPath("GRI30.yaml")).species
    assert gri["H2O"]["well"] != lib["H2O"]["well"]
    assert m.species["H2O"]["well"] == gri["H2O"]["well"]


def names(base):
    return sorted({cls.name for cls in subclasses(base) if cls.name})


def test_every_combination_of_models_merges_its_constants_once():
    kinds = (BaseEosModel, BaseTransportModel, BaseSpeciesDiffusionModel, BaseMixingRuleModel)
    # what the library never carries, which the perfectly constant models need stated
    stated = {
        "N2": dict(cp0=1040.0, mu0=1.76e-5, kappa0=0.0259),
        "O2": dict(cp0=918.0, mu0=2.04e-5, kappa0=0.0264),
        "AR": dict(cp0=520.0, mu0=2.23e-5, kappa0=0.0177),
    }
    allowed = set()
    for eos, trans, diffusion, rule in itertools.product(*(names(k) for k in kinds)):
        choices = dict(eos=eos, trans=trans, diffusion=diffusion)
        choices["mixing-rule"] = rule
        try:
            m = Mixture(section(stated, **choices))
        except ValueError as e:
            assert " needs " in str(e) and "Model" in str(e)
            continue
        own = {"Ru", "MW", "MWinv", "Rsp"}
        theirs = [key for model in m.models for key in model.constants(m.species)]
        assert len(theirs) == len(set(theirs)) and not own & set(theirs)
        assert set(m.constants()) == own | set(theirs)
        allowed.add((eos, trans, diffusion, rule))
    assert ("real-gas", "entropy-scaling", "takahashi", "wilke") in allowed
    assert ("tpg", "kinetic-theory", "binary", "herning") in allowed
    assert ("cpg", "constant", "lewis", "wilke") in allowed
    # kinetic theory fits over a cp curve, the dense models and takahashi need the critical point of a real gas, the plain one or the double flux's, binary the collision parameters
    realgas = {name for name in names(BaseEosModel) if issubclass(subclass_where(BaseEosModel, name=name), RealGasModel)}
    assert realgas > {"real-gas"}
    assert not {combo for combo in allowed if combo[0] == "cpg" and combo[1] != "constant"}
    assert not {combo for combo in allowed if combo[1] in ("chung", "entropy-scaling") and combo[0] not in realgas}
    assert not {combo for combo in allowed if combo[2] == "takahashi" and combo[0] not in realgas}
    assert not {combo for combo in allowed if combo[2] == "binary" and combo[1] == "constant"}


def test_a_reacting_mixture_needs_reactions():
    config = pg.Config()
    config["mixture"].update(section("GRI30.yaml", eos="tpg"))
    config["mixture"]["chemistry"] = "finite-rate"
    m = get_mixture(config)
    assert isinstance(m.chemistry, FiniteRateModel)
    r = m.chemistry.rxnconsts(m.species)
    assert int(r["nr"]) == 325 and r["logcfloor"] < 0.0
    config["mixture"]["chemistry"] = None
    assert get_mixture(config).chemistry is None
    config["mixture"].update(section(air, eos="tpg"))
    config["mixture"]["chemistry"] = "finite-rate"
    with pytest.raises(ValueError):
        get_mixture(config)


def test_the_chemistry_model_needs_the_thermally_perfect_caloric():
    with pytest.raises(ValueError, match="finite-rate needs TempDepCpModel"):
        Mixture(section("GRI30.yaml", eos="cpg", chemistry="finite-rate"))


def test_the_chemistry_model_is_one_of_the_mixtures_models():
    for eos in ("tpg", "real-gas"):
        m = Mixture(section("CH4_O2_FFCMY.yaml", eos=eos, chemistry="finite-rate"))
        assert m.chemistry in m.models and "RateState" in m.structs()
        theirs = [key for model in m.models for key in model.constants(m.species)]
        assert len(theirs) == len(set(theirs)) and {"rxnlogA", "tbdefeff", "troeuc"} <= set(theirs)
        assert set(m.constants()) == {"Ru", "MW", "MWinv", "Rsp"} | set(theirs)


def test_a_reacting_kind_needs_the_chemistry_model():
    config = pg.Config()
    config["mixture"].update(section("CH4_O2_FFCMY.yaml", eos="tpg"))
    config["chem-integration"]["kind"] = "coupled"
    with pytest.raises(ConfigError, match="needs mixture.chemistry"):
        get_simulator(config)
