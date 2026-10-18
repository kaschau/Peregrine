import peregrine as pg
from peregrine.tools.base import BaseTool


class ConfigTool(BaseTool):
    name = "config"
    help = "write a config file of every default, to start a case from"

    def add_cli(self, parser):
        parser.add_argument("out", help="the yaml to write")

    def main(self, args):
        pg.write_cfg(pg.Config(), args.out)
        print(
            f"Wrote the defaults to {args.out};"
            " docs/config.md says what each key does."
        )
