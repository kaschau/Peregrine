from peregrine.simulator.mixture.base import BaseModel


class BaseChemistryModel(BaseModel):
    def __init__(self, cfgsect, reactions):
        super().__init__(cfgsect)
        if not reactions:
            raise ValueError(f"{cfgsect['species']} has no reactions to react by")
        self.reactions = reactions
