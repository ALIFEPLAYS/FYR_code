from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .scientific_parameters import (
    ECOINVENT_VERSION,
    SUT_MANUFACTURING_GWP_ADDER_KGCO2E_PER_KG,
)
from .human_health_daly import (
    CLIMATE_CHANGE_HUMAN_HEALTH_DALY_PER_KG_CO2E,
    HUMAN_HEALTH_MIDPOINT_PATHWAYS,
    RECIPE_2016_ENDPOINT_HUMAN_HEALTH_METHOD,
    RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD,
)
from .ecosystem_quality import ENDPOINT_METHOD as ECOSYSTEM_ENDPOINT_METHOD
from .lcia_methods import (
    CARBON_FOOTPRINT_METHOD,
    FOSSIL_RESOURCE_METHOD,
    FOREGROUND_WATER_SITE_FACTORS,
    UK_NON_AGRICULTURAL_COUNTRY_FACTOR,
    WATER_SCARCITY_METHOD,
)


def validate_scientific_settings(settings: dict[str, Any]) -> None:
    """Reject stale thesis database settings and hidden packaging uplift."""
    brightway = settings.get("brightway", {})
    expected = {
        "ecoinvent_version": ECOINVENT_VERSION,
        "system_model": "cutoff",
        "project": "bw_ecoinvent_3_12_cutoff",
        "background_database": "ecoinvent-3.12-cutoff",
        "biosphere_database": "ecoinvent-3.12-biosphere",
    }
    mismatches = {
        key: (brightway.get(key), value)
        for key, value in expected.items()
        if brightway.get(key) != value
    }
    if mismatches:
        detail = ", ".join(
            f"{key}={actual!r} (expected {required!r})"
            for key, (actual, required) in mismatches.items()
        )
        raise ValueError(
            "The journal-paper model requires ecoinvent 3.12 cutoff and will not "
            f"fall back to the thesis database: {detail}"
        )

    plastics = settings.get("plastics", {})
    packaging_uplift = float(plastics.get("packaging_uplift_rate", 0.0))
    if packaging_uplift != 0.0:
        raise ValueError(
            "plastics.packaging_uplift_rate must be 0.0 because packaging is already "
            "represented in the source SUT masses."
        )

    adder = float(plastics.get("sut_manufacturing_gwp_adder_kgco2e_per_kg", -1.0))
    if adder != SUT_MANUFACTURING_GWP_ADDER_KGCO2E_PER_KG:
        raise ValueError(
            "plastics.sut_manufacturing_gwp_adder_kgco2e_per_kg must be 1.9."
        )
    if plastics.get("sut_manufacturing_gwp_adder_impact_scope") != "gwp_only":
        raise ValueError("The SUT component-manufacturing proxy must remain GWP-only.")
    if plastics.get("sut_manufacturing_gwp_adder_reporting_stage") != "sut_component_manufacturing":
        raise ValueError(
            "The SUT component-manufacturing proxy must retain its separate reporting stage."
        )
    if plastics.get("sut_manufacturing_gwp_adder_value_qualifier") != "upper_bound_up_to_1.9":
        raise ValueError("The 1.9 SUT fabrication proxy must be identified as an upper-bound value.")
    source_refs = plastics.get("sut_manufacturing_gwp_adder_source_references", [])
    if "https://doi.org/10.1002/bit.70297" not in source_refs or not any(
        "10048939" in str(reference) for reference in source_refs
    ):
        raise ValueError("The 1.9 SUT fabrication proxy must cite Reiners (2026) and Ramasamy (2018).")
    boundary = str(plastics.get("sut_manufacturing_gwp_adder_boundary", ""))
    for required_term in ("supplier-side transport", "cleanroom assembly", "gamma sterilisation"):
        if required_term not in boundary:
            raise ValueError(f"The 1.9 SUT proxy boundary must document {required_term}.")
    if (
        plastics.get("mechanical_recycling_substitution_credit_scope")
        != "virgin_polymer_production_only_excludes_sut_component_manufacturing"
    ):
        raise ValueError(
            "Mechanical-recycling substitution may credit virgin-polymer production "
            "only, not pharmaceutical-grade SUT component manufacture."
        )

    lcia = settings.get("lcia", {})
    configured_methods = lcia.get("methods", {})
    expected_methods = {
        "gwp": CARBON_FOOTPRINT_METHOD,
        "fossil_resource_use": FOSSIL_RESOURCE_METHOD,
        "water_scarcity": WATER_SCARCITY_METHOD,
    }
    for name, expected_method in expected_methods.items():
        if tuple(configured_methods.get(name, ())) != expected_method:
            raise ValueError(f"Unexpected LCIA method configured for {name}.")

    water = lcia.get("water_scarcity_regionalisation", {})
    if water.get("foreground_scope") != "known_site_tap_water_only":
        raise ValueError("AWARE regionalisation must be limited to known foreground tap water.")
    if water.get("background_scope") != "retain_native_brightway_geographies":
        raise ValueError("Background water consumption must retain native Brightway geographies.")
    if water.get("generic_scenario_factor_basis") != "UK_annual_nonagricultural_country":
        raise ValueError("Site-agnostic scenarios must use the documented UK non-agricultural factor.")
    if float(water.get("generic_scenario_factor_m3_world_eq_per_m3", -1.0)) != (
        UK_NON_AGRICULTURAL_COUNTRY_FACTOR
    ):
        raise ValueError("Unexpected UK country AWARE 2.0 fallback factor.")
    configured_sites = water.get("sites", {})
    for site, expected in FOREGROUND_WATER_SITE_FACTORS.items():
        configured = configured_sites.get(site, {})
        for field in ("factor_m3_world_eq_per_m3", "basin_id", "latitude", "longitude"):
            if float(configured.get(field, float("nan"))) != float(expected[field]):
                raise ValueError(f"Unexpected AWARE 2.0 {field} for {site}.")

    ecosystem = settings.get("ecosystem_quality", {})
    if ecosystem.get("internal_name") != "ecosystem_quality" or ecosystem.get("reporting_only") is not True:
        raise ValueError("Ecosystem quality must remain a reporting-only KPI.")
    if tuple(ecosystem.get("method", ())) != ECOSYSTEM_ENDPOINT_METHOD or ecosystem.get("unit") != "species.yr":
        raise ValueError("Use the installed ReCiPe 2016 (H) total ecosystem-quality endpoint.")
    if ecosystem.get("manual_sut_proxy_scope") != "climate_change_only_from_existing_gwp_proxy":
        raise ValueError("The SUT proxy has no full LCI; only its existing GWP may feed climate ecosystem damage.")

    human_health = settings.get("human_health_daly", {})
    if human_health.get("internal_name") != "human_health_daly":
        raise ValueError("The human-health endpoint KPI must use internal name human_health_daly.")
    if human_health.get("reporting_only") is not True:
        raise ValueError("human_health_daly must remain a reporting-only KPI.")
    if tuple(human_health.get("method", ())) != RECIPE_2016_MIDPOINT_BRIDGE_HUMAN_HEALTH_METHOD:
        raise ValueError(
            "human_health_daly must use the ReCiPe 2016 (H) midpoint-to-endpoint bridge."
        )
    if tuple(human_health.get("direct_endpoint_validation_method", ())) != (
        RECIPE_2016_ENDPOINT_HUMAN_HEALTH_METHOD
    ):
        raise ValueError("The direct ReCiPe Endpoint (H) comparator is configured incorrectly.")
    expected_equation = "human_health_daly = sum(midpoint_score_i * human_health_damage_factor_i)"
    if human_health.get("equation") != expected_equation:
        raise ValueError("The human-health DALY midpoint-to-endpoint equation is missing.")
    expected_pathways = list(HUMAN_HEALTH_MIDPOINT_PATHWAYS)
    if human_health.get("contributing_midpoint_pathways") != expected_pathways:
        raise ValueError("The human-health DALY pathway list is incomplete or out of order.")
    configured_factors = human_health.get("midpoint_to_endpoint_damage_factors", {})
    configured_units = human_health.get("midpoint_units", {})
    for pathway, specification in HUMAN_HEALTH_MIDPOINT_PATHWAYS.items():
        if float(configured_factors.get(pathway, -1.0)) != float(
            specification["damage_factor"]
        ):
            raise ValueError(f"Unexpected midpoint-to-endpoint damage factor for {pathway}.")
        if configured_units.get(pathway) != specification["midpoint_unit"]:
            raise ValueError(f"Unexpected ReCiPe midpoint unit for {pathway}.")
    if human_health.get("direct_endpoint_role") != "validation_only_not_added_to_reported_daly":
        raise ValueError("The direct endpoint may validate DALY but must not be added to it.")
    damage_factor = float(
        human_health.get("climate_change_damage_factor_daly_per_kg_co2e", -1.0)
    )
    if damage_factor != CLIMATE_CHANGE_HUMAN_HEALTH_DALY_PER_KG_CO2E:
        raise ValueError("Unexpected ReCiPe (H) climate-to-human-health damage factor.")
    if human_health.get("manual_sut_proxy_scope") != "climate_change_human_health_only":
        raise ValueError("The manual 1.9 kg CO2e SUT proxy may add climate-health damage only.")


def load_settings(project_root: Path, settings_path: Path | None = None) -> dict[str, Any]:
    path = Path(settings_path) if settings_path else Path(project_root) / "parameters" / "lca_parameters.json"
    with path.open("r", encoding="utf-8") as handle:
        settings = json.load(handle)
    validate_scientific_settings(settings)
    return settings


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(settings: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = copy.deepcopy(settings)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use dotted.path=value: {override}")
        path, raw_value = override.split("=", 1)
        target = result
        keys = path.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = _parse_value(raw_value)
    validate_scientific_settings(result)
    return result


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()
