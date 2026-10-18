class BaseTool:
    name = None
    help = None

    def add_cli(self, parser):
        raise NotImplementedError

    def main(self, args):
        raise NotImplementedError
