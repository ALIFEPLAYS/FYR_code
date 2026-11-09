"""Central journal-paper scientific assumptions with pure calculation helpers."""

from __future__ import annotations


ECOINVENT_VERSION = "3.12"

# SUT inventory assumptions. The workbook remains the untouched source record.
THESIS_TUBING_TO_ASSEMBLY_MASS_RATIO = 2.0
TUBING_TO_ASSEMBLY_MASS_RATIO = 1.0
BIOREACTOR_DEFECT_RATE = 0.11
MIXING_BAG_DEFECT_RATE = 0.02
PACKAGING_UPLIFT_RATE = 0.0

# Upper-bound bundled supplier-stage fabrication proxy: Reiners et al. (2026),
# citing Ramasamy (2018). Covers supplier transport, cleanroom assembly and gamma
# sterilisation; it replaces granular fabrication factors and not virgin resin.
SUT_MANUFACTURING_GWP_ADDER_KGCO2E_PER_KG = 1.9

# Quality-adjusted virgin displacement per kg of recovered mechanical recyclate.
SUBSTITUTION_RATIO = 0.81


def consumable_defect_rate(
    flow_name: str,
    *,
    bioreactor_rate: float = BIOREACTOR_DEFECT_RATE,
    mixing_bag_rate: float = MIXING_BAG_DEFECT_RATE,
) -> float:
    """Return the evidence-backed defect uplift for a named SUT consumable.

    CultiBag classes are treated as bioreactor consumables and Flexel mixing
    bags as mixing bags. Storage bags, filters, tubing-only items and generic
    bags are deliberately not inferred into either class.
    """
    name = " ".join(str(flow_name).strip().lower().split())
    if name.startswith("cultibag "):
        return float(bioreactor_rate)
    if name.startswith("flexel bag for mixing "):
        return float(mixing_bag_rate)
    return 0.0


def adjusted_consumable_mass(nominal_mass: float, defect_rate: float) -> float:
    return float(nominal_mass) * (1.0 + float(defect_rate))


def avoided_virgin_mass(
    incoming_waste_mass: float,
    recovery_yield: float,
    substitution_ratio: float = SUBSTITUTION_RATIO,
) -> float:
    """Calculate avoided virgin mass without changing physical recovery yield."""
    return (
        float(incoming_waste_mass)
        * float(recovery_yield)
        * float(substitution_ratio)
    )


def sut_manufacturing_gwp(
    modelled_sut_mass: float,
    adder_kgco2e_per_kg: float = SUT_MANUFACTURING_GWP_ADDER_KGCO2E_PER_KG,
) -> float:
    return float(modelled_sut_mass) * float(adder_kgco2e_per_kg)
