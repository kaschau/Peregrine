from importlib import resources

from mako.lookup import TemplateLookup
from mako.template import Template


class DottedTemplateLookup(TemplateLookup):
    def __init__(self):
        super().__init__(strict_undefined=True, input_encoding="utf-8")
        self.templates = {}

    # a dotted name is absolute; Mako's default resolves a uri as a path relative to the including template
    def adjust_uri(self, uri, relativeto):
        return uri

    def get_template(self, uri):
        if uri not in self.templates:
            package, _, name = uri.rpartition(".")
            path = resources.files(package).joinpath(f"{name}.mako")
            self.templates[uri] = Template(
                path.read_text(), filename=str(path), uri=uri, lookup=self, strict_undefined=True
            )
        return self.templates[uri]


class Templates:
    lookup = DottedTemplateLookup()
    base = "peregrine.backend.base.kernels"

    @classmethod
    def get(cls, template):
        return cls.lookup.get_template(template)
