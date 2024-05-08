def build_mask_apply(mask):

    def mask_apply(field):
        return field[mask]

    return mask_apply
