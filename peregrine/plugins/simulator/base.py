import re

from peregrine.plugins.base import BasePlugin


class BaseSimulatorPlugin(BasePlugin):
    # the physics this plugin is valid for: a regex their config names match whole, 'euler|navier-stokes.*', '.*' for any
    simulators = ""

    @classmethod
    def claim(cls, cfgsect, sim):
        plugins = super().claim(cfgsect, sim)
        for name, plugin in plugins.items():
            if not re.fullmatch(plugin.simulators, sim.name):
                what = f"{name} is valid for {plugin.simulators}"
                raise ValueError(f"{what}, not {sim.name}")
        return plugins

    # slot -> [(kernel, range)], asked once when the graphs are made and launched in every graph of that slot, so a kernel binds array names, not a register
    def launches(self, provider):
        return {}
