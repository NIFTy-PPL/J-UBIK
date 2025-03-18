from ..grid import Grid
from ..color import ColorRange


def _spectral_bins_within_range(
    grid: Grid,
    srange: tuple[float]
) -> tuple[int]:
    sbins = []
    for ii, cr in enumerate(grid.spectral.color_ranges):
        if cr.start in srange and cr.end in srange:
            sbins.append(ii)
        elif cr.start in srange or cr.end in srange:
            raise ValueError('Please set up your spectrum such that the bins '
                             'are completely within the spectral range.'
                             f'{cr} should be in {srange}.')
    return sbins


def last_spectral_bin(grid: Grid, spectral_range: ColorRange) -> int:
    spectral_bins = _spectral_bins_within_range(grid, spectral_range)
    if spectral_bins == []:
        raise ValueError('Could not find any spectral bin.')
    return max(spectral_bins)


def spectral_slices(
    grid: Grid,
    spectral_ranges: list[ColorRange]
) -> list[tuple[int | None]]:
    """
    Calculate spectral slices for multiple spectral ranges.

    Args:
        grid: Grid object containing spectral information
        spectral_ranges: List of ColorRange objects defining the spectral regions

    Returns:
        List of tuples containing (start_bin, end_bin) pairs, where None
        represents the beginning or end of the spectrum.
    """

    lsbs = [last_spectral_bin(grid, spectral_range) + 1
            for spectral_range in spectral_ranges]

    slices = []
    slices.append((None, lsbs[0]))
    for i in range(len(lsbs) - 1):
        slices.append((lsbs[i], lsbs[i + 1]))
    slices.append((lsbs[-1], None))

    return slices
