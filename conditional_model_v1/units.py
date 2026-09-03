"""Solution-chemistry unit conversion for professor-generated PHREEQC inputs.

Why this module exists
----------------------
The four legacy rocks and Sandstone were generated from `.phr` templates whose
SOLUTION block says ``units mol/kgw``. The three schist templates instead say
``units mg/L``. PHREEQC converts mg/L to mol/kgw internally, so every OUTPUT file
is already consistent across datasets; only the INPUT water-chemistry scalars
differ. Feeding raw mg/L and mol/kgw into one model column would make the same
physical concentration look ~1000x different and turn the column into a hidden
rock label. This module normalizes every dataset's solution scalars to mol/kgw
before they reach the condition vector.

How PHREEQC does the conversion (recovered from `phreeqc.dat` + the `.phr` files)
---------------------------------------------------------------------------------
``mol/kgw = (mg/L / 1000) / gfw`` where ``gfw`` is the gram formula weight
(g/mol) listed in ``SOLUTION_MASTER_SPECIES`` of ``phreeqc.dat``. PHREEQC also
applies a ~0.3% correction because one litre of solution holds slightly less
than one kilogram of water; that correction is ignored here (the legacy inputs
were randomized directly in mol/kgw, so there is no such correction on that side
either, and 0.3% is far below the model's resolution).

Cross-check (2026-09-03): the professor's hand-converted mol/kgw templates for the
three schists agree with this table for every species except Si, where the
template divided by 28.08 (element Si) rather than 60.08 (SiO2). Asked by mail,
the chemistry professor confirmed the same day that the mg/L silica value is a
measured SiO2 concentration (labs measure SiO2, not elemental Si) and that
PHREEQC reads it as SiO2. Every factor in the table is therefore verified.

Notes on the two derived factors:

* ``SI``: `phreeqc.dat` lists ``Si  H4SiO4  0  SiO2  28.0843``. The formula
  field (``SiO2``, 60.08 g/mol) is what PHREEQC uses for mg/L input, so silica is
  reported as mg/L SiO2, not mg/L Si. Used 60.08.
* ``ALKALINITY``: the schist templates write ``Alkalinity {ALKALINITY} as SO4-2``.
  PHREEQC uses the gram formula weight of the ``as`` formula literally
  (96.06 g/mol for SO4-2) without dividing by charge; that is why the database
  default formula is the half-equivalent ``Ca0.5(CO3)0.5``. Used 96.06.

The ``verified`` flag records that a factor was checked against the professor's
templates or the professor's written answer; ``conversion_table()`` exposes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SUPPORTED_INPUT_UNITS: tuple[str, ...] = ("mol_kgw", "mg_L")
DEFAULT_INPUT_UNITS = "mol_kgw"


@dataclass(frozen=True)
class SpeciesGfw:
    """Gram formula weight used by PHREEQC when an input is given in mg/L."""

    gfw_g_per_mol: float
    verified: bool
    source: str


# Keys are the `{FEATURE}` names written in the professor's `*_Input.txt` files.
# Values come from `SOLUTION_MASTER_SPECIES` in the professor's `phreeqc.dat`.
SOLUTION_SPECIES_GFW: dict[str, SpeciesGfw] = {
    "NA": SpeciesGfw(22.9898, True, "phreeqc.dat: Na Na+ 0 Na 22.9898"),
    "K": SpeciesGfw(39.102, True, "phreeqc.dat: K K+ 0 K 39.102"),
    "CA": SpeciesGfw(40.08, True, "phreeqc.dat: Ca Ca+2 0 Ca 40.08; ratio checked on Quartzite outputs"),
    "MG": SpeciesGfw(24.312, True, "phreeqc.dat: Mg Mg+2 0 Mg 24.312"),
    "CL": SpeciesGfw(35.453, True, "phreeqc.dat: Cl Cl- 0 Cl 35.453"),
    "MN": SpeciesGfw(54.938, True, "phreeqc.dat: Mn Mn+2 0 Mn 54.938"),
    "FE(2)": SpeciesGfw(55.847, True, "phreeqc.dat: Fe Fe+2 0 Fe 55.847; ratio checked on Quartzite outputs"),
    "AL": SpeciesGfw(26.9815, True, "phreeqc.dat: Al Al+3 0 Al 26.9815"),
    "SI": SpeciesGfw(
        60.08,
        True,
        "phreeqc.dat: Si H4SiO4 0 SiO2 28.0843 -> mg/L is read as SiO2 (60.08). The professor's hand-converted template used 28.08 (element Si), but the professor confirmed by mail (2026-09-03) that the value is measured SiO2 and PHREEQC treats it as SiO2.",
    ),
    "ALKALINITY": SpeciesGfw(96.06, True, ".phr: 'Alkalinity {ALKALINITY} as SO4-2' -> gfw of SO4-2; matches the professor's converted template"),
    "S6": SpeciesGfw(96.06, True, "phreeqc.dat: S(6) SO4-2 0 SO4 (legacy rocks only, always mol/kgw)"),
}

# Features that are concentrations and therefore change with `input_units`.
# Everything else in the input file (temperature, volumes, gas mole amounts,
# mineral moles/areas) is unit-independent and passes through untouched.
SOLUTION_SPECIES_FEATURES: tuple[str, ...] = tuple(SOLUTION_SPECIES_GFW)


def mg_per_l_to_mol_per_kgw(value_mg_per_l: float, gfw_g_per_mol: float) -> float:
    """Convert one concentration from mg/L to mol/kgw using PHREEQC's rule."""
    if gfw_g_per_mol <= 0.0:
        raise ValueError(f"gfw must be positive, got {gfw_g_per_mol}")
    return (float(value_mg_per_l) / 1000.0) / gfw_g_per_mol


def convert_solution_units(
    values: Mapping[str, Any],
    *,
    input_units: str,
) -> dict[str, Any]:
    """Return a copy of ``values`` with solution scalars expressed in mol/kgw.

    ``values`` is the parsed ``{KEY} value`` mapping of one input file. Only the
    keys listed in ``SOLUTION_SPECIES_FEATURES`` are touched; unknown keys and
    non-concentration keys are returned unchanged. ``input_units`` declares the
    units the file was written in (``"mol_kgw"`` is a no-op).
    """
    if input_units not in SUPPORTED_INPUT_UNITS:
        raise ValueError(
            f"Unsupported input_units {input_units!r}; expected one of {SUPPORTED_INPUT_UNITS}"
        )
    converted = dict(values)
    if input_units == "mol_kgw":
        return converted
    for feature, species in SOLUTION_SPECIES_GFW.items():
        if feature not in converted:
            continue
        raw = converted[feature]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric to convert units, got {raw!r}") from exc
        converted[feature] = mg_per_l_to_mol_per_kgw(value, species.gfw_g_per_mol)
    return converted


def conversion_table() -> list[dict[str, Any]]:
    """Serializable description of the factors, for cache manifests and reports."""
    return [
        {
            "feature": feature,
            "gfw_g_per_mol": species.gfw_g_per_mol,
            "mg_L_to_mol_kgw_divisor": species.gfw_g_per_mol * 1000.0,
            "verified": species.verified,
            "source": species.source,
        }
        for feature, species in SOLUTION_SPECIES_GFW.items()
    ]
