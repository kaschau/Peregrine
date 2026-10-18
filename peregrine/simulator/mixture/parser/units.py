from peregrine.simulator.mixture.constants import Ru, avogadro, debye, kB

# J/kmol per unit of an activation energy or a molar energy
energy = {
    "cal/mol": 4184.0,
    "kcal/mol": 4.184e6,
    "J/mol": 1.0e3,
    "kJ/mol": 1.0e6,
    "J/kmol": 1.0,
    "K": Ru,
}
# m per unit of length
length = {"cm": 1.0e-2, "m": 1.0, "angstrom": 1.0e-10}
# kmol per unit of amount
quantity = {"mol": 1.0e-3, "kmol": 1.0, "molec": 1.0 / avogadro}
# J per unit of a well depth stated as a temperature
tempenergy = {"K": kB}
# C m per unit of dipole moment
dipole = {"debye": debye}
# m^3 per unit of volume
volume = {"angstrom^3": 1.0e-30}
# the spellings CHEMKIN's REACTIONS line uses for Cantera's unit names
chemkin = {
    "CAL/MOLE": "cal/mol",
    "KCAL/MOLE": "kcal/mol",
    "JOULES/MOLE": "J/mol",
    "KJOULES/MOLE": "kJ/mol",
    "KELVINS": "K",
    "MOLES": "mol",
    "MOLECULES": "molec",
}
# what Cantera's yaml leaves implicit for a species' transport data
transport = {
    "well-depth": ("K", tempenergy),
    "diameter": ("angstrom", length),
    "dipole": ("debye", dipole),
    "polarizability": ("angstrom^3", volume),
}


def factor(table, unit):
    try:
        return table[unit]
    except KeyError:
        raise KeyError(f"unit {unit!r}; one of {', '.join(table)}") from None


def si(value, unit, table):
    return float(value) * factor(table, unit)


# only the amount in the denominator converts; the numerator is taken as SI
def molar(text):
    value, unit = str(text).split()
    amount = unit.split("/")[1]
    return float(value) / factor(quantity, amount)


class ReactionUnits:
    def __init__(self, units):
        self.energy = factor(energy, units["activation-energy"])
        self.length = factor(length, units["length"])
        self.quantity = factor(quantity, units["quantity"])

    def arrhenius(self, sect, order):
        A = float(sect["A"]) * (self.length**3 / self.quantity) ** (order - 1)
        return A, float(sect["b"]), float(sect["Ea"]) * self.energy / Ru
