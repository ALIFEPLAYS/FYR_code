"""ReCiPe 2016 midpoint-to-human-health endpoint bridge.

This module creates one reporting-only Brightway method whose characterisation
factors implement, for every inventory, the transparent relationship::

    human_health_daly = sum(midpoint_score_i * damage_factor_i)

The eight ``i`` pathways are the ReCiPe 2016 Hierarchist midpoint categories
that contribute to human health. The installed ReCiPe Endpoint (H) total is
used only to validate the combined characterisation factors; it is never added
to the reported result. Nothing in this module is imported by the MILP model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping


IMPACT_CATEGORY = "human_health_daly"
IMPACT_UNIT = "DALY"

RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD = (
    "IS thesis",
    "ReCiPe 2016 v1.03, midpoint-to-endpoint (H)",
    "human health",
    "total",
)

# The unmodified ecoinvent method is a validation comparator only. ecoinvent's
# four-part key uses "total: human health" before the "human health" indicator.
RECIPE_2016_ENDPOINT_HUMAN_HEALTH_METHOD = (
    "ecoinvent-3.12",
    "ReCiPe 2016 v1.03, endpoint (H)",
    "total: human health",
    "human health",
)

_MIDPOINT_PREFIX = (
    "ecoinvent-3.12",
    "ReCiPe 2016 v1.03, midpoint (H)",
)

# ReCiPe 2016 Hierarchist midpoint-to-human-health damage factors. Factor units
# are DALY per unit of the corresponding midpoint. The non-carcinogenic factor
# follows the active ecoinvent ReCiPe v1.03 implementation.
HUMAN_HEALTH_MIDPOINT_PATHWAYS = {
    "climate_change": {
        "method": _MIDPOINT_PREFIX
        + ("climate change", "global warming potential (GWP100)"),
        "midpoint_unit": "kg CO2-eq",
        "damage_factor": 9.28e-7,
    },
    "ozone_depletion": {
        "method": _MIDPOINT_PREFIX
        + ("ozone depletion", "ozone depletion potential (ODPinfinite)"),
        "midpoint_unit": "kg CFC-11-eq",
        "damage_factor": 5.31e-4,
    },
    "ionising_radiation": {
        "method": _MIDPOINT_PREFIX
        + ("ionising radiation", "ionising radiation potential (IRP)"),
        "midpoint_unit": "kBq Co-60-eq",
        "damage_factor": 8.50e-9,
    },
    "photochemical_ozone_formation_human_health": {
        "method": _MIDPOINT_PREFIX
        + (
            "photochemical oxidant formation: human health",
            "photochemical oxidant formation potential: humans (HOFP)",
        ),
        "midpoint_unit": "kg NOx-eq",
        "damage_factor": 9.10e-7,
    },
    "fine_particulate_matter_formation": {
        "method": _MIDPOINT_PREFIX
        + (
            "particulate matter formation",
            "particulate matter formation potential (PMFP)",
        ),
        "midpoint_unit": "kg PM2.5-eq",
        "damage_factor": 6.29e-4,
    },
    "human_carcinogenic_toxicity": {
        "method": _MIDPOINT_PREFIX
        + ("human toxicity: carcinogenic", "human toxicity potential (HTPc)"),
        "midpoint_unit": "kg 1,4-DCB-eq",
        "damage_factor": 3.32e-6,
    },
    "human_non_carcinogenic_toxicity": {
        "method": _MIDPOINT_PREFIX
        + (
            "human toxicity: non-carcinogenic",
            "human toxicity potential (HTPnc)",
        ),
        "midpoint_unit": "kg 1,4-DCB-eq",
        "damage_factor": 2.28e-7,
    },
    "water_consumption": {
        "method": _MIDPOINT_PREFIX
        + ("water use", "water consumption potential (WCP)"),
        "midpoint_unit": "m3 water consumed",
        "damage_factor": 2.22e-6,
    },
}

CLIMATE_CHANGE_HUMAN_HEALTH_DALY_PER_KG_CO2E = HUMAN_HEALTH_MIDPOINT_PATHWAYS[
    "climate_change"
]["damage_factor"]

# Rounded published factors cause a small difference from ecoinvent's directly
# implemented endpoint CFs. The bridge must remain within 0.5% on an L1
# comparison of the installed characterisation-factor vectors.
MAX_BRIDGE_CF_RELATIVE_L1_DIFFERENCE = 0.005


def _method_cf_map(bd, method: tuple[str, ...]) -> dict[object, float]:
    """Load deterministic CF values from a Brightway method."""
    return {
        row[0]: float(row[1])
        for row in bd.Method(method).load()
        if len(row) >= 2
    }


def build_midpoint_bridge_characterisation(
    midpoint_cf_maps: Mapping[str, Mapping[object, float]],
) -> dict[object, float]:
    """Return CFs for ``sum(midpoint_i * damage_factor_i)``."""
    combined: defaultdict[object, float] = defaultdict(float)
    for pathway, specification in HUMAN_HEALTH_MIDPOINT_PATHWAYS.items():
        if pathway not in midpoint_cf_maps:
            raise KeyError(f"Missing ReCiPe human-health midpoint CFs for {pathway}.")
        factor = float(specification["damage_factor"])
        for flow, midpoint_cf in midpoint_cf_maps[pathway].items():
            combined[flow] += float(midpoint_cf) * factor
    return dict(combined)


def _validate_against_direct_endpoint(
    bridge: Mapping[object, float],
    direct_endpoint: Mapping[object, float],
) -> float:
    """Validate bridge CFs against the direct ecoinvent endpoint CFs."""
    flows = set(bridge) | set(direct_endpoint)
    absolute_difference = sum(
        abs(float(bridge.get(flow, 0.0)) - float(direct_endpoint.get(flow, 0.0)))
        for flow in flows
    )
    direct_magnitude = sum(abs(float(direct_endpoint.get(flow, 0.0))) for flow in flows)
    relative_difference = absolute_difference / max(direct_magnitude, 1e-30)
    if relative_difference > MAX_BRIDGE_CF_RELATIVE_L1_DIFFERENCE:
        raise ValueError(
            "The ReCiPe midpoint-to-DALY bridge does not reconcile with the "
            f"installed direct Endpoint (H) method: {relative_difference:.3%}."
        )
    return relative_difference


def ensure_midpoint_bridge_method() -> tuple[str, ...]:
    """Create or update the reporting-only Brightway midpoint bridge method."""
    import bw2data as bd

    required_methods = {
        pathway: tuple(specification["method"])
        for pathway, specification in HUMAN_HEALTH_MIDPOINT_PATHWAYS.items()
    }
    missing = [method for method in required_methods.values() if method not in bd.methods]
    if RECIPE_2016_ENDPOINT_HUMAN_HEALTH_METHOD not in bd.methods:
        missing.append(RECIPE_2016_ENDPOINT_HUMAN_HEALTH_METHOD)
    if missing:
        raise ValueError(
            "The active ecoinvent 3.12 Brightway project is missing required "
            f"ReCiPe methods: {missing}"
        )

    midpoint_cf_maps = {
        pathway: _method_cf_map(bd, method)
        for pathway, method in required_methods.items()
    }
    bridge = build_midpoint_bridge_characterisation(midpoint_cf_maps)
    direct_endpoint = _method_cf_map(
        bd,
        RECIPE_2016_ENDPOINT_HUMAN_HEALTH_METHOD,
    )
    relative_difference = _validate_against_direct_endpoint(bridge, direct_endpoint)

    method = bd.Method(RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD)
    if RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD not in bd.methods:
        method.register(
            unit=IMPACT_UNIT,
            description=(
                "Reporting-only ReCiPe 2016 (H) human-health endpoint constructed "
                "as eight midpoint scores times their human-health damage factors. "
                "The ecoinvent direct endpoint is validation only."
            ),
        )
    try:
        existing = _method_cf_map(bd, RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD)
    except bd.errors.MissingIntermediateData:
        # Registration creates metadata; a new method has no CF data until written.
        existing = {}
    if existing != bridge:
        method.write([(flow, factor) for flow, factor in bridge.items()])
    metadata = {
        "midpoint_pathways": list(HUMAN_HEALTH_MIDPOINT_PATHWAYS),
        "equation": "human_health_daly = sum(midpoint_i * damage_factor_i)",
        "direct_endpoint_validation_method": list(
            RECIPE_2016_ENDPOINT_HUMAN_HEALTH_METHOD
        ),
        "bridge_cf_relative_l1_difference": relative_difference,
    }
    if existing != bridge and any(bd.methods[RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD].get(k) != v for k, v in metadata.items()):
        bd.methods[RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD].update(metadata)
        bd.methods.flush()
    return RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD


def add_human_health_method(methods: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Return an LCIA method mapping with the midpoint-derived DALY method."""
    result = dict(methods)
    result[IMPACT_CATEGORY] = ensure_midpoint_bridge_method()
    return result


def sut_manufacturing_human_health_daly_per_kg(settings: Mapping) -> float:
    """Climate-only DALY for the manual 1.9 kg CO2e/kg SUT burden."""
    plastics = settings["plastics"]
    human_health = settings["human_health_daly"]
    gwp_adder = float(plastics["sut_manufacturing_gwp_adder_kgco2e_per_kg"])
    damage_factor = float(
        human_health["climate_change_damage_factor_daly_per_kg_co2e"]
    )
    return gwp_adder * damage_factor
