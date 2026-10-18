import argparse

from peregrine.misc import subclasses
from peregrine.tools import BaseTool


def main(argv=None):
    desc = "Run a PEREGRINE case, or work on its files."
    parser = argparse.ArgumentParser(prog="peregrine", description=desc)
    subparsers = parser.add_subparsers(dest="tool", required=True, metavar="tool")
    for tool in (cls() for cls in subclasses(BaseTool)):
        text = tool.help
        sub = subparsers.add_parser(tool.name, help=text, description=text)
        tool.add_cli(sub)
        sub.set_defaults(main=tool.main)
    args = parser.parse_args(argv)
    args.main(args)


if __name__ == "__main__":
    main()
