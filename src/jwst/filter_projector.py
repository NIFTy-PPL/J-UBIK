import nifty8.re as jft


class FilterProjector(jft.Model):
    def __init__(self, sky_domain: jft.ShapeWithDtype, key_and_index: dict):
        self.key_and_index = key_and_index
        super().__init__(
            domain=sky_domain
        )

    def __call__(self, x):
        return {key: x[index] for key, index in self.key_and_index.items()}
