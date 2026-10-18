class BaseMessage:
    # what the config calls this kind; None for a kind the config does not choose
    name = None

    def __init__(self, backend, spool, rpool):
        self.backend = backend
        self.spool, self.rpool = spool, rpool

    def expect(self):
        raise NotImplementedError

    def copy_out(self):
        raise NotImplementedError

    def send(self):
        raise NotImplementedError

    def recv(self):
        raise NotImplementedError

    def sent(self):
        raise NotImplementedError
