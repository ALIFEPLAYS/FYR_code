"""Canonical LCIA methods and foreground AWARE 2.0 regionalisation helpers."""

from __future__ import annotations

from typing import Any, Mapping


CARBON_FOOTPRINT_METHOD = (
    "ecoinvent-3.12",
    "IMPACT World+ v2.1, footprint version",
    "climate change",
    "carbon footprint",
)

# This method is intentionally unchanged by the IMPACT World+ climate update.
FOSSIL_RESOURCE_METHOD = (
    "ecoinvent-3.12",
    "ReCiPe 2016 v1.03, midpoint (H)",
    "energy resources: non-renewable, fossil",
    "fossil fuel potential (FFP)",
)

WATER_SCARCITY_METHOD = (
    "ecoinvent-3.12",
    "IMPACT World+ v2.1, footprint version",
    "water use",
    "water scarcity footprint",
)

CORE_LCIA_METHODS = {
    "gwp": CARBON_FOOTPRINT_METHOD,
    "fossil_resource_use": FOSSIL_RESOURCE_METHOD,
    "water_scarcity": WATER_SCARCITY_METHOD,
}

CARBON_FOOTPRINT_METHOD_LABEL = (
    "IMPACT World+ v2.1 Carbon footprint / Climate change short term (GWP100; IPCC AR6)"
)
WATER_SCARCITY_METHOD_LABEL = "IMPACT World+ v2.1 Water scarcity footprint (AWARE 2.0)"

# Annual, non-agricultural AWARE 2.0 basin factors for the named sites. Values
# were resolved from the site coordinates against the official basin polygons.
FOREGROUND_WATER_SITE_FACTORS = {
    "Oxford": {
        "factor_m3_world_eq_per_m3": 4.05,
        "basin_id": 24053,
        "latitude": 51.7520,
        "longitude": -1.2577,
    },
    "Cambridge": {
        "factor_m3_world_eq_per_m3": 4.37,
        "basin_id": 23128,
        "latitude": 52.1951,
        "longitude": 0.1313,
    },
    "Slough": {
        "factor_m3_world_eq_per_m3": 4.05,
        "basin_id": 24053,
        "latitude": 51.5105,
        "longitude": -0.5950,
    },
}

# Used only for site-agnostic per-batch scenario tables. Site-resolved MILP
# reporting uses the basin factors above.
UK_NON_AGRICULTURAL_COUNTRY_FACTOR = 2.29

AWARE_2_SOURCE = "https://doi.org/10.5281/zenodo.15133241"
IMPACT_WORLD_PLUS_21_SOURCE = "https://doi.org/10.5281/zenodo.14041258"


def core_lcia_methods() -> dict[str, tuple[str, ...]]:
    """Return a copy so callers cannot mutate the canonical method mapping."""
    return dict(CORE_LCIA_METHODS)


def _method_cf_values(method_data: list[Any]) -> list[float]:
    values: list[float] = []
    for row in method_data:
        if len(row) < 2:
            continue
        factor = row[1]
        if isinstance(factor, (int, float)):
            values.append(float(factor))
        elif isinstance(factor, Mapping) and "amount" in factor:
            values.append(float(factor["amount"]))
    return values


def audit_water_method(bd: Any, method: tuple[str, ...] = WATER_SCARCITY_METHOD) -> dict[str, Any]:
    """Inspect whether the installed AWARE method is spatial or world-generic."""
    if method not in bd.methods:
        raise ValueError(f"Configured water-scarcity method is unavailable: {method}")
    bw_method = bd.Method(method)
    metadata = dict(bw_method.metadata)
    factors = _method_cf_values(bw_method.load())
    unique_factors = sorted({round(value, 12) for value in factors})
    geocollections = list(metadata.get("geocollections", []))
    generic_world_fallback = geocollections == ["world"] or len(unique_factors) == 1
    return {
        "method": method,
        "unit": metadata.get("unit"),
        "geocollections": geocollections,
        "characterisation_factor_count": len(factors),
        "unique_characterisation_factors": unique_factors,
        "generic_world_fallback": bool(generic_world_fallback),
    }


def _is_tap_water_activity(activity: Any) -> bool:
    text = " ".join(
        str(activity.get(field, "")).lower()
        for field in ("name", "reference product")
    )
    return "tap water" in text


def _foreground_water_provider_ids(activity: Any) -> set[int]:
    """Trace the top-level tap-water product chain, excluding unrelated upstream water."""
    provider_ids: set[int] = set()
    visited: set[int] = set()
    stack = [activity]
    while stack:
        current = stack.pop()
        current_id = int(current.id)
        if current_id in visited:
            continue
        visited.add(current_id)
        if _is_tap_water_activity(current):
            provider_ids.add(current_id)
        for exchange in current.technosphere():
            provider = exchange.input
            if _is_tap_water_activity(provider):
                stack.append(provider)
    return provider_ids


def regionalise_foreground_utility_water(
    *,
    lca: Any,
    utility_water_activity: Any,
    site_factors: Mapping[str, float],
    country_factor: float = UK_NON_AGRICULTURAL_COUNTRY_FACTOR,
) -> dict[str, Any]:
    """Replace only the traced foreground tap-water CF with AWARE 2.0 site CFs.

    The native Brightway score is split by activity column. Direct water flows in
    the top-level tap-water product chain are removed at the installed world CF and
    re-added at the site factor. All other supply-chain columns remain untouched.
    """
    import bw2data as bd

    audit = audit_water_method(bd, tuple(lca.method))
    unique = audit["unique_characterisation_factors"]
    if not audit["generic_world_fallback"]:
        raise ValueError(
            "The installed AWARE method is already spatially resolved; the explicit "
            "world-to-site correction must be reviewed before use."
        )
    if len(unique) != 1 or unique[0] <= 0:
        raise ValueError(
            "Expected one positive world-generic AWARE factor in the reduced "
            f"ecoinvent method; found {unique}."
        )
    native_world_cf = float(unique[0])
    provider_ids = _foreground_water_provider_ids(utility_water_activity)
    activity_dict = lca.dicts.activity
    provider_columns = [activity_dict[activity_id] for activity_id in provider_ids if activity_id in activity_dict]
    native_foreground_score = float(
        lca.characterized_inventory[:, provider_columns].sum()
    ) if provider_columns else 0.0
    correction_scope = "traced_foreground_water_columns"
    if native_foreground_score <= 0:
        # The reduced ecoinvent footprint method can place the elementary-water
        # exchanges below the tap-water provider columns.  In that case the
        # explicit foreground utility-water service is recharacterised as one
        # aggregate; water in every other route/activity remains untouched.
        native_foreground_score = float(lca.score)
        correction_scope = "explicit_foreground_utility_water_service"
    foreground_consumption_m3 = native_foreground_score / native_world_cf
    background_score = float(lca.score) - native_foreground_score
    corrected = {
        site: background_score + foreground_consumption_m3 * float(factor)
        for site, factor in site_factors.items()
    }
    corrected["UK_country_fallback"] = (
        background_score + foreground_consumption_m3 * float(country_factor)
    )
    return {
        "audit": audit,
        "native_total_score": float(lca.score),
        "native_world_cf": native_world_cf,
        "native_foreground_score": native_foreground_score,
        "foreground_consumption_m3": foreground_consumption_m3,
        "background_score": background_score,
        "corrected_scores": corrected,
        "provider_activity_ids": sorted(provider_ids),
        "correction_scope": correction_scope,
    }
