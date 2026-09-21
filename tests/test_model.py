"""Tests voor de profielberekeningen."""
import pandas as pd
import pytest

from leggerprofiel.model import (
    ProfileValidationError,
    build_profile,
    calculate_layers,
    default_boundaries,
    default_layers,
    default_profile,
    interpolate_y,
)


def test_default_layers_match_excel_distances() -> None:
    """De standaard grondopbouw reproduceert de afstanden uit Excel."""
    result = calculate_layers(default_layers(), 0.0, -1.77, "outside")
    assert result["Afstand"].tolist() == pytest.approx([12.92, 9.0, 9.3, 3.6])
    assert result.iloc[-1]["X"] == pytest.approx(-34.82)


def test_build_profile_interpolates_boundaries() -> None:
    """Zoneringsgrenzen krijgen een geinterpoleerde Y-waarde."""
    result = build_profile(default_profile(), default_layers(), default_boundaries())
    core = result.boundaries[result.boundaries["Omschrijving"] == "Grens kernzone"]
    assert core.iloc[0]["Y"] == pytest.approx(-1.77)
    assert core.iloc[1]["Y"] == pytest.approx(-1.77)


def test_empty_profile_is_rejected() -> None:
    """Een leeg profiel geeft een domeinfout."""
    with pytest.raises(ProfileValidationError, match="minimaal twee"):
        build_profile(pd.DataFrame(columns=["Omschrijving", "X", "Y", "Code"]), default_layers(), default_boundaries())


def test_duplicate_x_is_rejected() -> None:
    """Dubbele X-waarden voorkomen ambigue interpolatie."""
    profile = default_profile()
    profile.loc[1, "X"] = profile.loc[0, "X"]
    with pytest.raises(ProfileValidationError, match="uniek"):
        build_profile(profile, default_layers(), default_boundaries())


def test_decreasing_layer_order_is_required() -> None:
    """Een omhoog lopende laagscheiding wordt afgewezen."""
    layers = default_layers()
    layers.loc[1, "Laagscheiding"] = -4.0
    with pytest.raises(ProfileValidationError, match="hoog naar laag"):
        calculate_layers(layers, 0.0, -1.77, "inside")


def test_interpolation_outside_range_is_rejected() -> None:
    """Extrapolatie buiten het profiel gebeurt niet stilzwijgend."""
    profile = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, -1.0]})
    with pytest.raises(ProfileValidationError, match="buiten"):
        interpolate_y(profile, 2.0)
