"""Reporting-only ReCiPe 2016 (H) ecosystem-quality endpoint.

All inventory-based scores use the unmodified ecoinvent 3.12 total method.
The only foreground quantity without an LCI is the existing SUT fabrication
GWP proxy; its climate-only endpoint conversion is identified separately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping


IMPACT_CATEGORY = "ecosystem_quality"
IMPACT_UNIT = "species.yr"
ENDPOINT_METHOD = (
    "ecoinvent-3.12",
    "ReCiPe 2016 v1.03, endpoint (H)",
    "total: ecosystem quality",
    "ecosystem quality",
)
CLIMATE_MIDPOINT_METHOD = (
    "ecoinvent-3.12",
    "ReCiPe 2016 v1.03, midpoint (H)",
    "climate change",
    "global warming potential (GWP100)",
)


def audit_endpoint_method(bd) -> dict[str, object]:
    """Require the installed total to equal its twelve unmodified submethods."""
    if ENDPOINT_METHOD not in bd.methods or CLIMATE_MIDPOINT_METHOD not in bd.methods:
        raise ValueError("The installed ReCiPe 2016 (H) ecosystem methods are missing.")
    parts = [
        method for method in bd.methods
        if method[:2] == ENDPOINT_METHOD[:2]
        and method[2] == "ecosystem quality"
    ]
    if len(parts) != 12:
        raise ValueError(f"Expected 12 ecosystem pathways; found {len(parts)}.")
    total = dict(bd.Method(ENDPOINT_METHOD).load())
    summed = defaultdict(float)
    for method in parts:
        if bd.methods[method].get("unit") != IMPACT_UNIT:
            raise ValueError(f"Unexpected unit for {method}.")
        for flow, factor in bd.Method(method).load():
            summed[flow] += float(factor)
    flows = set(total) | set(summed)
    relative_l1_error = sum(abs(total.get(f, 0) - summed.get(f, 0)) for f in flows) / max(
        sum(abs(value) for value in total.values()), 1e-30
    )
    if relative_l1_error > 1e-10:
        raise ValueError(f"Ecosystem total does not reconcile to pathways: {relative_l1_error}.")
    if bd.methods[ENDPOINT_METHOD].get("unit") != IMPACT_UNIT:
        raise ValueError("Ecosystem endpoint unit is not species.yr.")
    return {"pathways": tuple(parts), "relative_l1_error": relative_l1_error}


def add_ecosystem_method(methods: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    import bw2data as bd

    audit_endpoint_method(bd)
    result = dict(methods)
    result[IMPACT_CATEGORY] = ENDPOINT_METHOD
    return result


def sut_manufacturing_climate_ecosystem_per_kg(settings: Mapping) -> float:
    """Convert the existing GWP-only SUT proxy through the installed CO2 CF.

    This is a climate-only approximation for an existing foreground proxy, not
    an invented biosphere exchange or a full manufacturing LCI.
    """
    import bw2data as bd

    audit_endpoint_method(bd)
    midpoint = dict(bd.Method(CLIMATE_MIDPOINT_METHOD).load())
    endpoint = dict(bd.Method(ENDPOINT_METHOD).load())
    co2_flows = [
        flow for flow in midpoint
        if (activity := bd.get_activity(flow)).get("name") == "Carbon dioxide, fossil"
        and activity.get("categories") == ("air",)
    ]
    if len(co2_flows) != 1 or abs(midpoint[co2_flows[0]] - 1.0) > 1e-12:
        raise ValueError("Cannot identify the unique ReCiPe fossil CO2 climate CF.")
    climate_factor = float(endpoint[co2_flows[0]])
    return (
        float(settings["plastics"]["sut_manufacturing_gwp_adder_kgco2e_per_kg"])
        * climate_factor
    )
