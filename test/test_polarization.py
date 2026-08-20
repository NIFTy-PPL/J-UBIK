import pytest

from jubik.polarization import (
    POLARIZATION_INVTABLE,
    POLARIZATION_TABLE,
    Polarization,
    PolarizationType,
)

pmp = pytest.mark.parametrize

CIRCULAR = [5, 6, 7, 8]
LINEAR = [9, 10, 11, 12]


def test_tables_are_inverse_of_each_other():
    assert set(POLARIZATION_TABLE) == set(CIRCULAR + LINEAR)
    for index, label in POLARIZATION_TABLE.items():
        assert POLARIZATION_INVTABLE[label] == index


@pmp("indices", ([], [5], [5, 8], CIRCULAR, LINEAR))
def test_roundtrip_through_list(indices):
    pol = Polarization(indices)
    assert pol.to_list() == list(indices)
    assert Polarization.from_list(pol.to_list()) == pol


def test_to_str_list():
    assert Polarization(CIRCULAR).to_str_list() == ["RR", "RL", "LR", "LL"]
    assert Polarization(LINEAR).to_str_list() == ["XX", "XY", "YX", "YY"]
    assert Polarization.trivial().to_str_list() == []


def test_more_than_four_polarizations_rejected():
    with pytest.raises(AssertionError):
        Polarization([5, 6, 7, 8, 9])


def test_trivial_has_length_one():
    # A trivial polarization still occupies one axis entry (Stokes I).
    assert len(Polarization.trivial()) == 1
    assert Polarization.trivial().to_list() == []


@pmp("indices", ([5], [5, 8], CIRCULAR, LINEAR))
def test_length_equals_number_of_indices(indices):
    assert len(Polarization(indices)) == len(indices)


@pmp("indices,expected", (([5, 8], True), (CIRCULAR, True), (LINEAR, False)))
def test_circular(indices, expected):
    assert Polarization(indices).circular() is expected


def test_circular_raises_for_trivial():
    with pytest.raises(RuntimeError):
        Polarization.trivial().circular()


def test_circular_raises_for_mixed_feeds():
    with pytest.raises(RuntimeError):
        Polarization([5, 9]).circular()


@pmp(
    "indices,expected",
    (
        ([5, 8], False),
        ([9, 12], False),
        (CIRCULAR, True),
        (LINEAR, True),
        ([6], True),
        ([11], True),
        ([], False),
    ),
)
def test_has_crosshanded(indices, expected):
    assert Polarization(indices).has_crosshanded() is expected


def test_restrict_to_stokes_i():
    # Convention: (LL, RR) for circular feeds, (XX, YY) for linear ones.
    assert Polarization(CIRCULAR).restrict_to_stokes_i() == Polarization([8, 5])
    assert Polarization(LINEAR).restrict_to_stokes_i() == Polarization([9, 12])


def test_restrict_to_stokes_i_is_idempotent():
    for indices in ([8, 5], [9, 12]):
        pol = Polarization(indices)
        assert pol.restrict_to_stokes_i() == pol


def test_restrict_by_name():
    pol = Polarization.trivial().restrict_by_name(["XX", "YY"])
    assert pol == Polarization([9, 12])


@pmp(
    "indices,expected",
    (
        (CIRCULAR, [3, 0]),
        ([5, 8], [1, 0]),
        ([8, 5], [0, 1]),
        (LINEAR, [0, 3]),
        ([9, 12], [0, 1]),
    ),
)
def test_stokes_i_indices(indices, expected):
    assert Polarization(indices).stokes_i_indices() == expected


def test_stokes_i_indices_raises_for_trivial():
    with pytest.raises(RuntimeError):
        Polarization.trivial().stokes_i_indices()


def test_stokes_i_indices_raises_if_parallel_hand_missing():
    # RR and RL only: LL is not part of the data set.
    with pytest.raises(ValueError):
        Polarization([5, 6]).stokes_i_indices()


def test_equality():
    assert Polarization([5, 8]) == Polarization([5, 8])
    assert Polarization([5, 8]) != Polarization([8, 5])
    assert Polarization.trivial() == Polarization([])
    assert Polarization([5]) != [5]
    assert Polarization([5]) != None  # noqa: E711


def test_repr():
    assert repr(Polarization([5, 8])) == "Polarization((5, 8))"


def test_space_of_trivial_is_stokes_i():
    assert Polarization.trivial().space.labels == ("I",)


def test_space_labels_follow_indices():
    assert Polarization(CIRCULAR).space.labels == ("RR", "RL", "LR", "LL")
    assert Polarization([9, 12]).space.labels == ("XX", "YY")


@pmp(
    "indices,expected",
    (
        ([], PolarizationType.I),
        ([5], PolarizationType.RR),
        ([5, 8], PolarizationType.RR_LL),
        ([8, 5], PolarizationType.LL_RR),
        ([9, 12], PolarizationType.XX_YY),
        (CIRCULAR, PolarizationType.RR_RL_LR_LL),
        (LINEAR, PolarizationType.XX_XY_YX_YY),
    ),
)
def test_polarization_type_from_polarization_object(indices, expected):
    assert PolarizationType.from_polarization_object(Polarization(indices)) is expected


def test_polarization_type_from_unknown_combination_raises():
    with pytest.raises(ValueError):
        PolarizationType.from_polarization_object(Polarization([5, 6]))


@pmp("indices", ([], [5], [5, 8], [8, 5], [9, 12], CIRCULAR, LINEAR))
def test_polarization_type_legacy_roundtrip(indices):
    pol = Polarization(indices)
    typ = PolarizationType.from_polarization_object(pol)
    assert typ.get_legacy_polarization() == pol


def test_stokes_i_type_maps_to_trivial_polarization():
    assert PolarizationType.I.get_legacy_polarization() == Polarization.trivial()


def test_iquv_has_no_legacy_representation():
    # I, Q, U, V are not part of the correlator-product table.
    with pytest.raises(KeyError):
        PolarizationType.IQUV.get_legacy_polarization()


@pmp(
    "typ",
    (
        PolarizationType.I,
        PolarizationType.RR,
        PolarizationType.LL,
        PolarizationType.XX,
        PolarizationType.YY,
    ),
)
def test_is_single_feed_true(typ):
    assert typ.is_single_feed


@pmp(
    "typ",
    (
        PolarizationType.RR_LL,
        PolarizationType.LL_RR,
        PolarizationType.XX_YY,
        PolarizationType.IQUV,
        PolarizationType.RR_RL_LR_LL,
        PolarizationType.XX_XY_YX_YY,
    ),
)
def test_is_single_feed_false(typ):
    assert not typ.is_single_feed


@pmp("typ", tuple(PolarizationType))
def test_polarization_type_length_and_shape(typ):
    assert len(typ) == len(typ.value)
    assert typ.shape == (len(typ.value),)
