import nifty8.re as jft


class FilterProjector(jft.Model):
    def __init__(self, sky_domain: jft.ShapeWithDtype, keys_and_colors: dict):
        self.keys_and_colors = keys_and_colors
        self.keys_and_index = {key: index for
                               index, key in enumerate(keys_and_colors.keys())}
        super().__init__(
            domain=sky_domain
        )

    def get_key(self, color):
        out_key = ''
        for k, v in self.keys_and_colors.items():
            if color in v:
                if out_key != '':
                    raise IndexError(
                        f'{color} fits into multiple keys of the '
                        'FilterProjector')
                out_key = k
        if out_key == '':
            raise IndexError(
                f"{color} doesn't fit in the bounds of the FilterProjector.")

        return out_key

    def __call__(self, x):
        return {key: x[index] for key, index in self.keys_and_index.items()}
