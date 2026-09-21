"""Domeinlogica voor het berekenen van een leggerprofiel."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd

PROFILE_COLUMNS = ["Omschrijving", "X", "Y", "Code"]
LAYER_COLUMNS = ["Grondlaag", "Laagscheiding", "Helling"]


class ProfileValidationError(ValueError):
    """Meld een invoerfout die begrijpelijk aan de gebruiker kan worden getoond."""


@dataclass(frozen=True)
class ProfileResult:
    """Bevat de berekende profielpunten en afgeleide tabellen."""

    combined: pd.DataFrame
    outside_layers: pd.DataFrame
    inside_layers: pd.DataFrame
    boundaries: pd.DataFrame


def default_profile() -> pd.DataFrame:
    """Geef de profielinvoer uit het oorspronkelijke rekenblad terug."""
    return pd.DataFrame(
        [
            ("1. Onderhoudsdiepte", -4.5, -2.6, "99"),
            ("2. Buitenteen", -0.25, -2.6, "99"),
            ("3. Referentielijn", 0.0, -1.77, "90"),
            ("4. Binnenkruin", 3.0, -1.77, "99"),
            ("5. Binnenteen", 5.8, -3.6, "99"),
            ("6. Insteek sloot", 6.7, -3.6, "99"),
            ("7. Teen sloot", 15.7, -4.4, "99"),
            ("8. Teen sloot", 16.4, -4.4, "99"),
            ("9. Uitsteek sloot", 17.0, -3.6, "99"),
            ("X. Uittredepunt glijcirkel", 10.0, -3.2, "@"),
        ],
        columns=PROFILE_COLUMNS,
    )


def default_layers() -> pd.DataFrame:
    """Geef de standaard grondopbouw uit het rekenblad terug."""
    return pd.DataFrame(
        [
            ("Onderkant zand", -5.0, 4.0),
            ("Onderkant veen", -6.5, 6.0),
            ("Onderkant klei", -9.6, 3.0),
            ("Onderkant veen", -10.2, 6.0),
        ],
        columns=LAYER_COLUMNS,
    )


def default_boundaries() -> pd.DataFrame:
    """Geef de standaard zoneringsgrenzen terug."""
    return pd.DataFrame(
        [
            ("Grens kernzone", 0.0, "Buitendijks", "25"),
            ("Grens beschermingszone", -10.0, "Buitendijks", "25"),
            ("Grens buitenbeschermingszone", -20.0, "Buitendijks", "25"),
            ("Grens kernzone", 3.0, "Binnendijks", "25"),
            ("Grens beschermingszone", 13.0, "Binnendijks", "25"),
            ("Grens buitenbeschermingszone", 23.0, "Binnendijks", "25"),
        ],
        columns=["Omschrijving", "X", "Zijde", "Code"],
    )


def _numeric(frame: pd.DataFrame, columns: Iterable[str], name: str) -> pd.DataFrame:
    """Converteer vereiste kolommen naar getallen of geef een gerichte fout."""
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(columns)].isna().any().any():
        raise ProfileValidationError(f"{name} bevat lege of ongeldige getallen.")
    return result


def validate_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Valideer profielpunten en sorteer ze op X."""
    if profile.empty:
        raise ProfileValidationError("Voeg minimaal twee profielpunten toe.")
    missing = set(PROFILE_COLUMNS) - set(profile.columns)
    if missing:
        raise ProfileValidationError(f"Profiel mist kolommen: {', '.join(sorted(missing))}.")
    result = _numeric(profile, ["X", "Y"], "Het profiel")
    result = result.dropna(subset=["Omschrijving"]).copy()
    if len(result) < 2:
        raise ProfileValidationError("Voeg minimaal twee geldige profielpunten toe.")
    if result["X"].duplicated().any():
        raise ProfileValidationError("X-coordinaten van profielpunten moeten uniek zijn.")
    reference = result[result["Omschrijving"].astype(str).str.contains("Referentielijn", case=False)]
    if len(reference) != 1:
        raise ProfileValidationError("Neem precies een punt 'Referentielijn' op.")
    return result.sort_values("X", kind="stable").reset_index(drop=True)


def calculate_layers(
    layers: pd.DataFrame,
    start_x: float,
    reference_y: float,
    side: str,
) -> pd.DataFrame:
    """Bereken X-posities van laagscheidingen vanuit dikte maal helling."""
    if side not in {"outside", "inside"}:
        raise ValueError("side moet 'outside' of 'inside' zijn.")
    if layers.empty:
        return pd.DataFrame(columns=[*LAYER_COLUMNS, "Laagdikte", "Afstand", "X", "Y", "Code"])
    missing = set(LAYER_COLUMNS) - set(layers.columns)
    if missing:
        raise ProfileValidationError(f"Grondopbouw mist kolommen: {', '.join(sorted(missing))}.")
    result = _numeric(layers.dropna(how="all"), ["Laagscheiding", "Helling"], "De grondopbouw")
    if (result["Helling"] < 0).any():
        raise ProfileValidationError("Hellingen van grondlagen mogen niet negatief zijn.")
    y_values = result["Laagscheiding"].to_numpy(dtype=float)
    previous = np.r_[reference_y, y_values[:-1]]
    thickness = previous - y_values
    if (thickness <= 0).any():
        raise ProfileValidationError("Laagscheidingen moeten van hoog naar laag zijn ingevoerd.")
    distance = thickness * result["Helling"].to_numpy(dtype=float)
    cumulative = np.cumsum(distance)
    result["Laagdikte"] = thickness
    result["Afstand"] = distance
    result["X"] = start_x - cumulative if side == "outside" else start_x + cumulative
    result["Y"] = y_values
    result["Code"] = "99"
    return result[["Grondlaag", "Laagscheiding", "Laagdikte", "Helling", "Afstand", "X", "Y", "Code"]]


def interpolate_y(profile: pd.DataFrame, x: float) -> float:
    """Interpoleer de profielhoogte lineair op een X-coordinaat."""
    ordered = profile.sort_values("X")
    minimum, maximum = ordered["X"].min(), ordered["X"].max()
    if x < minimum or x > maximum:
        raise ProfileValidationError(
            f"Zoneringsgrens X={x:g} ligt buiten het berekende bereik {minimum:g} tot {maximum:g}."
        )
    return float(np.interp(x, ordered["X"], ordered["Y"]))


def build_profile(
    profile: pd.DataFrame,
    layers: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> ProfileResult:
    """Bereken grondlagen, zoneringshoogten en de gecombineerde uitvoer."""
    surface = validate_profile(profile)
    reference = surface[surface["Omschrijving"].astype(str).str.contains("Referentielijn", case=False)].iloc[0]
    inside_start = surface[surface["Omschrijving"].astype(str).str.contains("Binnenkruin", case=False)]
    inside_x = float(inside_start.iloc[0]["X"]) if not inside_start.empty else float(reference["X"])
    outside = calculate_layers(layers, float(reference["X"]), float(reference["Y"]), "outside")
    inside = calculate_layers(layers, inside_x, float(reference["Y"]), "inside")

    layer_points = pd.concat(
        [
            outside[["Grondlaag", "X", "Y", "Code"]].rename(columns={"Grondlaag": "Omschrijving"}),
            inside[["Grondlaag", "X", "Y", "Code"]].rename(columns={"Grondlaag": "Omschrijving"}),
        ],
        ignore_index=True,
    )
    interpolation = pd.concat([surface[PROFILE_COLUMNS], layer_points], ignore_index=True)
    interpolation = interpolation.sort_values("X").drop_duplicates("X", keep="last")

    boundary_result = boundaries.dropna(how="all").copy()
    if not boundary_result.empty:
        boundary_result = _numeric(boundary_result, ["X"], "De zonering")
        boundary_result["Y"] = [interpolate_y(interpolation, x) for x in boundary_result["X"]]
    else:
        boundary_result["Y"] = pd.Series(dtype=float)

    combined = pd.concat(
        [
            surface[PROFILE_COLUMNS],
            layer_points[PROFILE_COLUMNS],
            boundary_result[["Omschrijving", "X", "Y", "Code"]],
        ],
        ignore_index=True,
    ).sort_values(["X", "Y"], kind="stable").reset_index(drop=True)
    return ProfileResult(combined, outside, inside, boundary_result)


def to_excel_bytes(result: ProfileResult, profile: pd.DataFrame, layers: pd.DataFrame) -> bytes:
    """Maak een downloadbaar Excel-bestand met invoer en resultaten."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        profile.to_excel(writer, sheet_name="Profielinvoer", index=False)
        layers.to_excel(writer, sheet_name="Grondopbouw", index=False)
        result.boundaries.to_excel(writer, sheet_name="Zonering", index=False)
        result.combined.to_excel(writer, sheet_name="Gecombineerd profiel", index=False)
        result.outside_layers.to_excel(writer, sheet_name="Buitendijks", index=False)
        result.inside_layers.to_excel(writer, sheet_name="Binnendijks", index=False)
    return output.getvalue()
