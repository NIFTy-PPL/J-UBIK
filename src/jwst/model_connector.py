import nifty8.re as jft


class ConnectModels(jft.Model):
    def __init__(self, models: list[jft.Model]):
        domain = {}
        target = {}
        for m in models:
            domain = domain | m.domain
            target = target | m.target
        self.models = models

        super().__init__(domain=domain, target=target)

    def __call__(self, x):
        out = {}
        for m in self.models:
            out = out | m(x)
        return out
