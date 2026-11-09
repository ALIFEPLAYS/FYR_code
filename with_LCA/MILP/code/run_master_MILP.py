from __future__ import annotations

"""
Deterministic industrial-symbiosis MILP: policy-cost core build.

The central analysis solves:
  1. the constructed business-as-usual benchmark; and
  2. the single cost-plus-lifecycle-carbon-value optimisation.

Pareto, uncertainty, and scenario analyses have separate entry points.
"""

import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack

# -----------------------------------------------------------------------------
# Project paths and solver controls
# -----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
INPUT_DIR = PROJECT_ROOT / 'inputs'
PARAMETER_DIR = PROJECT_ROOT.parent / 'parameters'
RESULT_ROOT = PROJECT_ROOT / 'results'

GAMS_RESOURCE_DIR = Path('/Library/Frameworks/GAMS.framework/Resources')
if GAMS_RESOURCE_DIR.exists():
    os.environ['PATH'] = str(GAMS_RESOURCE_DIR) + os.pathsep + os.environ.get('PATH', '')

GAMS_SOLVER = os.environ.get('IS_GAMS_SOLVER', 'cplex').strip().lower()
GAMS_SOLVER_IO = os.environ.get('IS_GAMS_SOLVER_IO', 'shell').strip().lower()
GAMS_SOLVER_NAME = f'gams:{GAMS_SOLVER}'
GAMS_TEE = True
GAMS_KEEPFILES = False
GAMS_PREFLIGHT = True

INPUT_FILES = {
    'supply_chain': Path(os.environ.get(
        'IS_SUPPLY_CHAIN_FILE',
        PROJECT_ROOT.parent / 'mAb_SupplyChain_model/mab_supply_chain_superpro_v2_results.csv',
    )),
    'eligibility': INPUT_DIR / 'polymer_route_eligibility_20260722.xlsx',
    'lca': Path(os.environ.get(
        'IS_LCA_ROUTE_FACTOR_FILE',
        PROJECT_ROOT.parent / 'LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx',
    )),
    'superpro_master': PROJECT_ROOT.parent / 'SuperPro_Reports_Data_Extraction/data/processed/SP_master_dataset.xlsx',
    'params': PARAMETER_DIR / 'MILP_central_params.json',
}
RESULT_DIR = RESULT_ROOT / 'ecosystem_master_20261015'
GAMS_LOG_DIR = RESULT_DIR / 'gams_logs'
GAMS_WORK_DIR = RESULT_DIR / 'gams_work'
GAMS_SOLVE_REPORT = RESULT_DIR / 'gams_solve_report.csv'
GAMS_PREFLIGHT_LOG = GAMS_LOG_DIR / '0000_gams_preflight.log'
RESULTS_WORKBOOK = RESULT_DIR / 'industrial_symbiosis_MILP_results.xlsx'
_GAMS_SOLVE_COUNTER = 0
_GAMS_PREFLIGHT_DONE = False

MIP_REL_GAP = 1e-9
SOLVE_TIME_LIMIT_SECONDS: Optional[int] = None
TOL = 1e-7

# These names are retained because they are the model's reported metrics, not
# because a multi-objective sweep is performed.
OBJECTIVES = ['cost', 'gwp', 'circularity_loss', 'freshwater', 'residual']
OBJECTIVE_LABELS = {
    'cost': 'Annual network cost (GBP/year)',
    'gwp': 'Carbon footprint, IMPACT World+ v2.1 GWP100/IPCC AR6 (kg CO2e/year)',
    'circularity_loss': 'Circularity loss (t/year)',
    'freshwater': 'Mains freshwater withdrawal (m3/year)',
    'residual': 'Residual disposal (t/year)',
}

@dataclass
class LinearExpression:
    coeff: np.ndarray
    const: float = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        return float(self.const + np.dot(self.coeff, x))

@dataclass
class ModelData:
    sites: List[str]
    site_location: Dict[str, str]
    polymers: List[str]
    routes: List[str]
    route_lca_pathway: Dict[str, str]
    transport_modes: List[str]
    electricity_sources: List[str]
    batches: Dict[str, float]  # fixed BAU allocation from upstream supply-chain solution
    batch_high: float
    batch_low: float
    batch_delta: float
    total_batches: float
    site_open: Dict[str, int]
    product_kg_per_batch: Dict[str, float]
    electricity_kwh_per_batch: float
    wastewater_m3_per_batch: float
    utility_water_m3_per_batch: float
    fixed_mains_water_m3_per_batch: float
    electricity_kwh: Dict[str, float]  # BAU annual quantities
    wastewater_m3: Dict[str, float]  # BAU annual quantities
    utility_water_m3: Dict[str, float]  # BAU annual quantities
    fixed_mains_water_m3: Dict[str, float]  # BAU annual quantities
    water_recovery: Dict[str, float]
    wwt_capacity_m3: Dict[str, float]
    polymer_kg_per_batch: Dict[str, float]
    eligibility: Dict[Tuple[str, str], int]
    recovery_yield: Dict[Tuple[str, str], float]
    substitution_ratio: Dict[Tuple[str, str], float]
    residual_fraction: Dict[Tuple[str, str], float]
    route_cost_gbp_per_t: Dict[Tuple[str, str], float]
    autoclave_required: Dict[Tuple[str, str], int]
    autoclave_cost_gbp_per_t: Dict[str, float]
    recovered_value_gbp_per_t: Dict[Tuple[str, str], float]
    value_capture: Dict[Tuple[str, str], float]
    distance_km: Dict[Tuple[str, str], float]
    roundtrip_multiplier: Dict[Tuple[str, str, str], float]
    payload_t: Dict[str, float]
    transport_cost_gbp_per_km: Dict[str, float]
    transport_gwp_kg_per_km: Dict[str, float]
    electricity_cost_gbp_per_kwh: Dict[str, float]
    electricity_gwp_kg_per_kwh: Dict[str, float]
    electricity_available: Dict[Tuple[str, str], int]
    treatment_fixed_cost_gbp: Dict[str, float]
    treatment_opex_gbp_per_m3: Dict[str, float]
    fresh_water_cost_gbp_per_m3: Dict[str, float]
    discharge_cost_gbp_per_m3: Dict[str, float]
    treatment_gwp_kg_per_m3: Dict[str, float]
    fresh_water_gwp_kg_per_m3: Dict[str, float]
    plastic_gwp_kg_per_kg: Dict[Tuple[str, str], float]
    virgin_polymer_gwp_kg_per_kg: Dict[Tuple[str, str], float]
    sut_component_manufacturing_gwp_kg_per_kg: Dict[Tuple[str, str], float]
    avoided_virgin_material_credit_gwp_kg_per_kg: Dict[Tuple[str, str], float]
    plastic_human_health_daly_per_kg: Dict[Tuple[str, str], float]
    electricity_human_health_daly_per_kwh: Dict[str, float]
    treatment_human_health_daly_per_m3: Dict[str, float]
    fresh_water_human_health_daly_per_m3: Dict[str, float]
    fixed_non_electric_human_health_daly_per_batch: float
    biomass_human_health_daly_per_batch: float
    plastic_ecosystem_quality_per_kg: Dict[Tuple[str, str], float]
    electricity_ecosystem_quality_per_kwh: Dict[str, float]
    treatment_ecosystem_quality_per_m3: Dict[str, float]
    fresh_water_ecosystem_quality_per_m3: Dict[str, float]
    fixed_non_electric_ecosystem_quality_per_batch: float
    fixed_manufacturing_gwp_kg: float
    biomass_gwp_kg: float
    biomass_cost_gbp: float
    biomass_kg_per_batch: float
    biomass_routes: List[str]
    biomass_route_factors: Dict[str, Dict[str, float]]
    biomass_route_cost_gbp_per_kg: Dict[str, float]
    biomass_useful_heat_mj_per_kg: Dict[str, float]
    lifecycle_carbon_value_gbp_per_t: float
    total_product_kg: float
    route_capacities_t: Dict[str, float] = field(default_factory=dict)
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)

@dataclass
class LinearModel:
    data: ModelData
    var_names: List[str]
    var_keys: List[Tuple]
    var_types: List[str]
    index: Dict[Tuple, int]
    lower: np.ndarray
    upper: np.ndarray
    integrality: np.ndarray
    A: coo_matrix
    b_l: np.ndarray
    b_u: np.ndarray
    expressions: Dict[str, LinearExpression]

@dataclass
class SolveResult:
    case_id: str
    success: bool
    status: int
    message: str
    x: np.ndarray
    metrics: Dict[str, float]
    raw_objective: float
    primary_objective: str
    selection_method: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def _single_row(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:
    rows = df.loc[mask]
    require(len(rows) == 1, f'Expected exactly one row for {label}; found {len(rows)}.')
    return rows.iloc[0]

def load_model_data() -> ModelData:
    for name, path in INPUT_FILES.items():
        require(path.exists(), f'Missing required input file: {name}: {path}')

    supply = pd.read_csv(INPUT_FILES['supply_chain'])
    eligibility_raw = pd.read_excel(INPUT_FILES['eligibility'], sheet_name='polymer_route_eligibility')
    site_map_df = pd.read_excel(INPUT_FILES['eligibility'], sheet_name='site_mapping')
    route_factors = pd.read_excel(INPUT_FILES['lca'], sheet_name='route_factor_matrix')
    io_table = pd.read_excel(INPUT_FILES['lca'], sheet_name='input_output_table')
    sp = pd.read_excel(INPUT_FILES['superpro_master'], sheet_name='Superpro_SR_ICR_processed')
    params = json.loads(INPUT_FILES['params'].read_text())

    # Never consume a stale thesis-era route-factor workbook silently.
    required_lca_metadata = {
        'ecoinvent_version': '3.12',
        'ecoinvent_system_model': 'cutoff',
    }
    for column, expected in required_lca_metadata.items():
        require(column in route_factors.columns, f'LCA workbook is missing required metadata column {column!r}.')
        observed = set(route_factors[column].dropna().astype(str).unique())
        require(observed == {expected}, f'LCA workbook {column} must be {expected!r}; found {sorted(observed)}.')
    expected_gwp_method = (
        'ecoinvent-3.12 | IMPACT World+ v2.1, footprint version | '
        'climate change | carbon footprint'
    )
    require(
        'gwp_method' in route_factors.columns
        and set(route_factors['gwp_method'].dropna().astype(str).unique()) == {expected_gwp_method},
        'LCA workbook must use IMPACT World+ v2.1 Carbon footprint (IPCC AR6) for GWP.',
    )
    expected_water_method = (
        'ecoinvent-3.12 | IMPACT World+ v2.1, footprint version | '
        'water use | water scarcity footprint'
    )
    require(
        'water_scarcity_method' in route_factors.columns
        and set(route_factors['water_scarcity_method'].dropna().astype(str).unique())
        == {expected_water_method},
        'LCA workbook must retain IMPACT World+ v2.1 AWARE 2.0 water scarcity.',
    )
    expected_site_factors = {'Oxford': 4.05, 'Cambridge': 4.37, 'Slough': 4.05}
    for location, expected_factor in expected_site_factors.items():
        factor_column = f'aware20_site_cf__{location}'
        site_factor_column = f'net_water_scarcity_factor__{location}'
        require(factor_column in route_factors.columns, f'Missing AWARE 2.0 factor for {location}.')
        require(site_factor_column in route_factors.columns, f'Missing site-resolved water factor for {location}.')
        require(
            set(pd.to_numeric(route_factors[factor_column], errors='raise').unique())
            == {expected_factor},
            f'Unexpected AWARE 2.0 factor for {location}.',
        )
    require(
        'sut_manufacturing_gwp_adder_kgco2e_per_kg' in route_factors.columns,
        'LCA workbook predates the explicit SUT manufacturing GWP adder.',
    )
    require(
        set(pd.to_numeric(
            route_factors['sut_manufacturing_gwp_adder_kgco2e_per_kg'], errors='coerce'
        ).dropna().unique()) == {1.9},
        'LCA workbook must use the 1.9 kg CO2e/kg SUT manufacturing GWP adder.',
    )
    require(
        'sut_manufacturing_gwp_adder_value_qualifier' in route_factors.columns
        and set(route_factors['sut_manufacturing_gwp_adder_value_qualifier'].dropna().astype(str).unique())
        == {'upper_bound_up_to_1.9'},
        'The 1.9 SUT fabrication proxy must be identified as an upper-bound value.',
    )
    require(
        'sut_manufacturing_gwp_adder_source_references' in route_factors.columns
        and route_factors['sut_manufacturing_gwp_adder_source_references'].astype(str).str.contains(
            '10.1002/bit.70297', regex=False
        ).all()
        and route_factors['sut_manufacturing_gwp_adder_source_references'].astype(str).str.contains(
            '10048939', regex=False
        ).all(),
        'The 1.9 SUT proxy must retain the Reiners (2026) and Ramasamy (2018) source chain.',
    )
    require(
        'packaging_uplift_rate' in route_factors.columns
        and set(pd.to_numeric(route_factors['packaging_uplift_rate'], errors='coerce').dropna().unique()) == {0.0},
        'LCA workbook must document zero additional packaging uplift.',
    )
    expected_human_health_method = (
        'IS thesis | ReCiPe 2016 v1.03, midpoint-to-endpoint (H) | human health | total'
    )
    require(
        'human_health_daly_method' in route_factors.columns
        and set(route_factors['human_health_daly_method'].dropna().astype(str).unique())
        == {expected_human_health_method},
        'LCA workbook must contain the ReCiPe 2016 (H) midpoint-to-human-health endpoint bridge.',
    )
    require(
        'human_health_daly_equation' in route_factors.columns
        and set(route_factors['human_health_daly_equation'].dropna().astype(str).unique())
        == {'human_health_daly = sum(midpoint_score_i * human_health_damage_factor_i)'},
        'LCA workbook must retain the explicit midpoint-to-DALY equation.',
    )
    required_daly_midpoints = {
        'climate_change',
        'ozone_depletion',
        'ionising_radiation',
        'photochemical_ozone_formation_human_health',
        'fine_particulate_matter_formation',
        'human_carcinogenic_toxicity',
        'human_non_carcinogenic_toxicity',
        'water_consumption',
    }
    require(
        'human_health_daly_contributing_midpoints' in route_factors.columns
        and route_factors['human_health_daly_contributing_midpoints'].dropna().astype(str).map(
            lambda value: set(value.split(' | ')) == required_daly_midpoints
        ).all(),
        'LCA workbook must retain all eight ReCiPe human-health midpoint pathways.',
    )
    require(
        'human_health_daly_direct_endpoint_role' in route_factors.columns
        and set(route_factors['human_health_daly_direct_endpoint_role'].dropna().astype(str).unique())
        == {'validation_only_not_added_to_reported_daly'},
        'The direct ReCiPe endpoint must remain validation-only.',
    )
    require(
        'human_health_daly_reporting_only' in route_factors.columns
        and route_factors['human_health_daly_reporting_only'].fillna(False).astype(bool).all(),
        'human_health_daly must be marked reporting-only.',
    )
    expected_ecosystem_method = (
        'ecoinvent-3.12 | ReCiPe 2016 v1.03, endpoint (H) | '
        'total: ecosystem quality | ecosystem quality'
    )
    require(
        'ecosystem_quality_method' in route_factors.columns
        and set(route_factors['ecosystem_quality_method'].dropna().astype(str).unique())
        == {expected_ecosystem_method}
        and 'ecosystem_quality_unit' in route_factors.columns
        and set(route_factors['ecosystem_quality_unit'].dropna().astype(str).unique())
        == {'species.yr'}
        and 'ecosystem_quality_reporting_only' in route_factors.columns
        and route_factors['ecosystem_quality_reporting_only'].fillna(False).astype(bool).all(),
        'The LCA workbook must contain the unmodified reporting-only ReCiPe ecosystem endpoint.',
    )

    # Fixed upstream manufacturing solution.
    require(supply['manufacturing_site'].is_unique, 'Manufacturing sites are not unique in supply-chain results.')
    sites = supply['manufacturing_site'].astype(str).tolist()
    batches = dict(zip(sites, supply['mab_batches_per_year'].astype(float)))
    site_open = dict(zip(sites, supply['site_open'].round().astype(int)))
    require(all(abs(v - round(v)) < 1e-8 for v in batches.values()), 'Annual batch counts must be integral.')
    batches = {j: float(round(v)) for j, v in batches.items()}
    require(all(v in (0, 1) for v in site_open.values()), 'site_open must be binary.')

    # Preserve the upstream batch pattern but allow the optimisation to choose which active site
    # receives the unique low-volume allocation. The uploaded solution is 328, 328 and 205 batches.
    active_batch_values = [batches[j] for j in sites if site_open[j] == 1]
    unique_batch_values = sorted(set(active_batch_values))
    require(len(unique_batch_values) == 2,
            f'Expected exactly two active-site batch levels (high and low); found {unique_batch_values}.')
    batch_low, batch_high = float(unique_batch_values[0]), float(unique_batch_values[1])
    require(sum(abs(v - batch_low) <= 1e-9 for v in active_batch_values) == 1,
            'Exactly one active site must have the low batch allocation.')
    require(sum(abs(v - batch_high) <= 1e-9 for v in active_batch_values) == len(active_batch_values) - 1,
            'All remaining active sites must have the high batch allocation.')
    batch_delta = batch_high - batch_low
    require(batch_delta > 0, 'The high batch level must exceed the low batch level.')
    total_batches = float(sum(active_batch_values))

    require(site_map_df['manufacturing_site'].is_unique, 'Site mapping contains duplicate manufacturing_site rows.')
    site_location = dict(zip(site_map_df['manufacturing_site'].astype(str), site_map_df['site_location'].astype(str)))
    require(set(site_location) == set(sites), 'Site mapping does not exactly cover supply-chain sites.')

    # Saleable final mAb output is the Product/mAb stream, not total product solution mass or overall mAb balance.
    product_row = _single_row(
        sp,
        (sp['flow_name_superpro'].astype(str).str.fullmatch('Product', case=False, na=False))
        & (sp['component'].astype(str).str.fullmatch('mAb', case=False, na=False))
        & (sp['unit_per_batch'].astype(str).str.fullmatch('kg/batch', case=False, na=False)),
        'saleable Product/mAb output',
    )
    pi = float(product_row['amount_per_batch'])
    require(pi > 0, 'Saleable mAb output must be positive.')
    product_kg_per_batch = {j: pi for j in sites}

    # Foreground quantities from route-factor database.
    def rf_row(flow: str, pathway: Optional[str] = None, material: Optional[str] = None) -> pd.Series:
        mask = route_factors['optimisation_flow'].astype(str).eq(flow)
        if pathway is not None:
            mask &= route_factors['pathway'].astype(str).eq(pathway)
        if material is not None:
            mask &= route_factors['material_or_service'].astype(str).eq(material)
        return _single_row(route_factors, mask, f'route factor {flow}/{pathway}/{material}')

    electricity_grid_row = rf_row('manufacturing__electricity', 'electricity_grid_uk_bau')
    electricity_wind_row = rf_row('manufacturing__electricity', 'electricity_renewable_onshore_wind')
    ww_discharge_row = rf_row('plant_wastewater', 'treatment_and_discharge')
    fresh_water_row = rf_row('fresh_utility_water', 'fresh_utility_water_supply')
    biomass_row = rf_row('residual_biomass', 'ad_biomethane_digestate')
    biomass_kg_per_batch = float(biomass_row['represented_quantity_per_batch'])
    require(biomass_kg_per_batch > 0, 'Residual biomass per batch must be positive.')
    biomass_routes = ['ad_gas_supply', 'ad_external_steam']
    biomass_assumptions = json.loads((PROJECT_ROOT.parent / 'parameters'
                                     / 'electrcity_biomass_route_assumptions.json').read_text())
    heat_efficiency = float(biomass_assumptions['biomethane_to_useful_steam_efficiency'])
    require(0 < heat_efficiency <= 1, 'Invalid biomethane-to-steam efficiency.')
    biomass_io = pd.read_excel(INPUT_FILES['lca'], sheet_name='input_output_table')
    biomethane_rows = biomass_io.loc[
        biomass_io['source_module'].eq('biomass')
        & biomass_io['material_or_service'].eq('biomethane_to_gas_grid')
    ]
    require(len(biomethane_rows) == 1, 'Expected one biomethane output row.')
    biomethane_m3_per_batch = float(biomethane_rows.iloc[0]['quantity_per_batch'])
    biomethane_ncv = float(biomass_assumptions['biomethane_ncv_mj_per_m3'])
    heat_mj_per_kg = biomethane_m3_per_batch * biomethane_ncv * heat_efficiency / biomass_kg_per_batch
    stages = pd.read_excel(INPUT_FILES['lca'], sheet_name='lca_stage_breakdown')
    impact_columns = {
        'gwp': 'net_gwp_factor_per_optimisation_unit',
        'fossil_resource_use': 'net_fossil_resource_use_factor_per_optimisation_unit',
        'water_scarcity': 'net_water_scarcity_factor_per_optimisation_unit',
        'human_health_daly': 'net_human_health_daly_factor_per_optimisation_unit',
        'ecosystem_quality': 'net_ecosystem_quality_factor_per_optimisation_unit',
    }
    biomass_route_factors = {route: {} for route in biomass_routes}
    for impact, column in impact_columns.items():
        gas_stage = stages.loc[
            stages['source_module'].eq('biomass')
            & stages['life_cycle_stage'].eq('avoided_natural_gas')
            & stages['impact_category'].eq(impact)
        ]
        steam_stage = stages.loc[
            stages['optimisation_flow'].eq('manufacturing__steam')
            & stages['life_cycle_stage'].eq('manufacturing_utility_supply')
            & stages['impact_category'].eq(impact)
        ]
        require(len(gas_stage) == 1 and len(steam_stage) == 1,
                f'Missing unique gas/steam LCIA stage for {impact}.')
        gas_credit = float(gas_stage.iloc[0]['represented_impact_per_batch']) / biomass_kg_per_batch
        steam_factor = float(steam_stage.iloc[0]['impact_factor'])
        require(gas_credit < 0 and steam_factor > 0, f'Unexpected biomass LCIA signs for {impact}.')
        gas_route = float(biomass_row[column])
        biomass_route_factors['ad_gas_supply'][impact] = gas_route
        biomass_route_factors['ad_external_steam'][impact] = gas_route - gas_credit - heat_mj_per_kg * steam_factor
    biomass_route_cost_gbp_per_kg = {route: 0.0 for route in biomass_routes}
    biomass_useful_heat_mj_per_kg = {'ad_gas_supply': 0.0, 'ad_external_steam': heat_mj_per_kg}
    wfi_row = rf_row('manufacturing__wfi', 'baseline_supply')
    chilled_row = rf_row('manufacturing__chilled_water', 'baseline_supply')
    cooling_row = rf_row('manufacturing__cooling_water', 'baseline_supply')
    steam_row = rf_row('manufacturing__steam', 'baseline_supply')

    electricity_per_batch = float(electricity_grid_row['represented_quantity_per_batch'])
    require(abs(electricity_per_batch - float(electricity_wind_row['represented_quantity_per_batch'])) < 1e-9,
            'Grid and wind electricity rows do not use the same per-batch demand.')
    wastewater_per_batch = float(ww_discharge_row['represented_quantity_per_batch'])
    utility_water_per_batch = float(fresh_water_row['represented_quantity_per_batch'])
    wfi_kg_per_batch = float(wfi_row['represented_quantity_per_batch'])
    fixed_mains_per_batch = wfi_kg_per_batch / 1000.0  # water mass-volume identity used only for withdrawal accounting

    # Current workbook explicitly marks the full wastewater volume as recoverable internal utility water.
    recover_row = _single_row(
        io_table,
        (io_table['source_module'].astype(str).eq('wastewater_lca'))
        & (io_table['io_table_role'].astype(str).eq('recoverable_internal_utility_water'))
        & (io_table['pathway'].astype(str).eq('treatment_for_utility_reuse')),
        'recoverable internal utility water',
    )
    recoverable_per_batch = float(recover_row['optimisation_quantity_per_batch'])
    eta_w = recoverable_per_batch / wastewater_per_batch
    require(0 <= eta_w <= 1 + 1e-9, f'Water recovery fraction outside [0,1]: {eta_w}')
    eta_w = min(1.0, eta_w)
    require(utility_water_per_batch + 1e-9 >= eta_w * wastewater_per_batch,
            'Recoverable water exceeds eligible utility-water demand; current formulation would require curtailment not represented.')

    # These annual dictionaries retain the original upstream allocation for the fixed BAU benchmark.
    # Optimised cases derive annual quantities from the endogenous B[j] variables.
    electricity_kwh = {j: batches[j] * electricity_per_batch for j in sites}
    wastewater_m3 = {j: batches[j] * wastewater_per_batch for j in sites}
    utility_water_m3 = {j: batches[j] * utility_water_per_batch for j in sites}
    fixed_mains_water_m3 = {j: batches[j] * fixed_mains_per_batch for j in sites}
    water_recovery = {j: eta_w for j in sites}

    wwt = params['onsite_wastewater_treatment_facility']
    wwt_capacity = {j: float(wwt['design_capacity_m3_per_day']) * 365.0 for j in sites}
    for j in sites:
        require(wastewater_m3[j] <= wwt_capacity[j] * site_open[j] + 1e-8,
                f'Wastewater treatment capacity infeasible at {j}: demand={wastewater_m3[j]}, capacity={wwt_capacity[j]}')

    # Polymer inventory and route LCIA rows.
    plastic_rows = route_factors.loc[route_factors['optimisation_flow'].astype(str).str.startswith('sut_plastic__')].copy()
    required_sut_production_columns = {
        'gwp__production',
        'gwp__virgin_polymer_production',
        'gwp__sut_component_manufacturing',
        'gwp__avoided_virgin_material_credit',
    }
    missing_sut_production_columns = required_sut_production_columns - set(plastic_rows.columns)
    require(
        not missing_sut_production_columns,
        'LCA workbook must report virgin-polymer and SUT component-manufacturing '
        f'GWP separately; missing {sorted(missing_sut_production_columns)}.',
    )
    component_manufacturing_gwp = pd.to_numeric(
        plastic_rows['gwp__sut_component_manufacturing'], errors='raise'
    )
    virgin_polymer_gwp = pd.to_numeric(
        plastic_rows['gwp__virgin_polymer_production'], errors='raise'
    )
    aggregate_production_gwp = pd.to_numeric(
        plastic_rows['gwp__production'], errors='raise'
    )
    require(
        np.allclose(component_manufacturing_gwp, 1.9, atol=1e-12),
        'Every SUT route row must report the 1.9 kg CO2e/kg component-manufacturing stage separately.',
    )
    require(
        np.allclose(
            aggregate_production_gwp,
            virgin_polymer_gwp + component_manufacturing_gwp,
            atol=1e-10,
        ),
        'Aggregated SUT production GWP does not reconcile to virgin polymer plus component manufacture.',
    )
    polymers = sorted(plastic_rows['material_or_service'].astype(str).unique().tolist())
    per_batch_by_polymer = plastic_rows.groupby('material_or_service')['represented_quantity_per_batch'].nunique()
    require((per_batch_by_polymer == 1).all(), 'Polymer per-batch quantity differs across route rows.')
    polymer_kg_per_batch = plastic_rows.groupby('material_or_service')['represented_quantity_per_batch'].first().astype(float).to_dict()
    require(sum(polymer_kg_per_batch.values()) > 0, 'Modelled SUT polymer inventory must be positive.')

    # Base-case route availability only. STRAP remains disabled and no scenario file is applied.
    base = eligibility_raw.loc[
        eligibility_raw['eligible'].fillna(False).astype(bool)
        & eligibility_raw['base_enabled'].fillna(False).astype(bool)
    ].copy()
    base = base.loc[base['lca_pathway'] != 'chemical_recycling_strap_hypothetical'].copy()

    # Expand the mechanical-recycling pathway into two route nodes because the supplied distances use two facilities.
    route_lca_pathway = {
        'landfill_all': 'landfill_all',
        'incineration_efw_heat_electricity': 'incineration_efw_heat_electricity',
        'mechanical_recycling_polyolefins': 'mechanical_recycling_main_residual_incineration',
        'mechanical_recycling_tpe_tubing': 'mechanical_recycling_main_residual_incineration',
    }
    routes = list(route_lca_pathway)
    eligibility: Dict[Tuple[str, str], int] = {(p, r): 0 for p in polymers for r in routes}
    autoclave_required: Dict[Tuple[str, str], int] = {(p, r): 0 for p in polymers for r in routes}
    output_role: Dict[Tuple[str, str], str] = {(p, r): 'not_applicable' for p in polymers for r in routes}
    for row in base.itertuples(index=False):
        p = str(row.polymer)
        pathway = str(row.lca_pathway)
        if pathway == 'mechanical_recycling_main_residual_incineration':
            route = 'mechanical_recycling_tpe_tubing' if p == 'TPE' else 'mechanical_recycling_polyolefins'
        else:
            route = pathway
        eligibility[p, route] = 1
        requires_autoclave = int(bool(row.requires_autoclave))
        # Validated decontamination logic for the deterministic formulation:
        # landfill and mechanical recycling require autoclaving; EfW does not.
        if route == 'incineration_efw_heat_electricity':
            require(
                requires_autoclave == 0,
                f'Autoclaving must not be assigned to the EfW route {p}/{route}.',
            )
        elif route == 'landfill_all' or route.startswith('mechanical_recycling'):
            require(
                requires_autoclave == 1,
                f'Autoclaving is required for landfill and mechanical recycling: {p}/{route}.',
            )
        autoclave_required[p, route] = requires_autoclave
        output_role[p, route] = str(row.output_role)

    for p in polymers:
        require(sum(eligibility[p, r] for r in routes) >= 1, f'No base-case route available for polymer {p}.')

    # Physical recovery yield and quality-adjusted virgin substitution are explicit,
    # independent route parameters. Revenue uses recovery yield; the 0.81 factor is
    # already embedded once in the LCA avoided-virgin credit only.
    recovery_yield: Dict[Tuple[str, str], float] = {}
    substitution_ratio: Dict[Tuple[str, str], float] = {}
    residual_fraction: Dict[Tuple[str, str], float] = {}
    plastic_gwp: Dict[Tuple[str, str], float] = {}
    virgin_polymer_gwp: Dict[Tuple[str, str], float] = {}
    sut_component_manufacturing_gwp: Dict[Tuple[str, str], float] = {}
    avoided_virgin_material_credit_gwp: Dict[Tuple[str, str], float] = {}
    plastic_human_health_daly: Dict[Tuple[str, str], float] = {}
    plastic_ecosystem_quality: Dict[Tuple[str, str], float] = {}
    for p in polymers:
        for r in routes:
            pathway = route_lca_pathway[r]
            row = _single_row(
                plastic_rows,
                (plastic_rows['material_or_service'].astype(str).eq(p))
                & (plastic_rows['pathway'].astype(str).eq(pathway)),
                f'plastic route factor {p}/{pathway}',
            )
            plastic_gwp[p, r] = float(row['net_gwp_factor_per_optimisation_unit'])
            virgin_polymer_gwp[p, r] = float(row['gwp__virgin_polymer_production'])
            sut_component_manufacturing_gwp[p, r] = float(row['gwp__sut_component_manufacturing'])
            avoided_virgin_material_credit_gwp[p, r] = float(row['gwp__avoided_virgin_material_credit'])
            plastic_human_health_daly[p, r] = float(
                row['net_human_health_daly_factor_per_optimisation_unit']
            )
            plastic_ecosystem_quality[p, r] = float(
                row['net_ecosystem_quality_factor_per_optimisation_unit']
            )

            # The route-factor workbook already embeds the autoclave LCIA
            # burden in its net route factor for landfill and recycling. In the
            # current exported matrix, the explanatory stage columns aggregate
            # autoclaving with route treatment for some pathways; gwp__supply is
            # therefore not a reliable standalone autoclave field. Validate the
            # physical route membership from the explicit life-cycle-stage label,
            # but optimise with the supplied net factor to avoid double counting.
            if eligibility[p, r]:
                stages = {
                    token.strip()
                    for token in str(row.get('life_cycle_stages', '')).split('|')
                    if token.strip()
                }
                has_embedded_autoclave = 'autoclave_cleaning' in stages
                require(
                    has_embedded_autoclave == bool(autoclave_required[p, r]),
                    f'Autoclave eligibility and embedded LCIA stage disagree for {p}/{r}: '
                    f'requires_autoclave={autoclave_required[p, r]}, stages={sorted(stages)}.',
                )
                require(
                    math.isfinite(plastic_gwp[p, r]),
                    f'Non-finite net plastic GWP factor for {p}/{r}.',
                )

            if eligibility[p, r] and output_role[p, r] == 'recovered':
                require(
                    'recycling_recovery_fraction' in row.index
                    and 'virgin_material_substitution_ratio' in row.index,
                    f'Explicit recovery/substitution parameters are missing for {p}/{r}.',
                )
                eta = float(row['recycling_recovery_fraction'])
                substitution = float(row['virgin_material_substitution_ratio'])
                require(0 <= eta <= 1, f'Recovery yield outside [0,1] for {p}/{r}: {eta}')
                require(
                    math.isclose(substitution, 0.81, abs_tol=1e-12),
                    f'Mechanical-recycling substitution ratio must be 0.81 for {p}/{r}; found {substitution}.',
                )
                # Fossil-resource production contains no 1.9 GWP-only adder, so it
                # provides a clean check that the environmental credit is y * 0.81.
                production = float(row['fossil_resource_use__production'])
                credit = float(row['fossil_resource_use__credit'])
                if production > 0 and credit < 0:
                    require(
                        math.isclose(-credit / production, eta * substitution, rel_tol=1e-9, abs_tol=1e-12),
                        f'Avoided-virgin credit does not equal recovery yield × substitution ratio for {p}/{r}.',
                    )
                gwp_virgin_production = float(row['gwp__virgin_polymer_production'])
                gwp_avoided_credit = float(row['gwp__avoided_virgin_material_credit'])
                require(
                    math.isclose(
                        gwp_avoided_credit,
                        -gwp_virgin_production * eta * substitution,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    ),
                    f'GWP avoided-material credit for {p}/{r} must replace virgin-polymer '
                    'production only, excluding SUT component manufacture.',
                )
            else:
                eta = 0.0
                substitution = 0.0
            recovery_yield[p, r] = eta
            substitution_ratio[p, r] = substitution
            residual_fraction[p, r] = 1.0 - eta

    # Monetary basis conversion to nominal GBP 2026.
    mult = params['currency_and_price_year_conversions']['multipliers_to_nominal_GBP_2026']
    m2025 = float(mult['GBP_2025_nominal'])
    m2020real = float(mult['GBP_2020_real'])
    landfill_cost = 26.0 * m2025 + 130.75  # explicit special rule in parameter file
    efw_cost = 121.0 * m2025
    route_cost: Dict[Tuple[str, str], float] = {}
    recovered_value: Dict[Tuple[str, str], float] = {}
    value_capture: Dict[Tuple[str, str], float] = {}
    volk = params['recycling_costs']['volk_regranulation_cost_conversion']
    original_eur2019 = float(volk['MECH_RECYCLING_COST_ORIGINAL_EUR2019_PER_T'])
    inflation_multiplier = float(volk['inflation_multiplier_2019_to_2026_august'])
    fx_gbp_per_eur = float(volk['fx_rate_gbp_per_eur'])
    calculated_volk_gbp2026 = original_eur2019 * inflation_multiplier * fx_gbp_per_eur
    configured_volk_gbp2026 = float(volk['MECH_RECYCLING_COST_GBP2026_PER_T'])
    require(
        math.isclose(calculated_volk_gbp2026, configured_volk_gbp2026, abs_tol=0.005),
        'Documented Volk inflation/FX conversion does not reconcile to the configured model cost.',
    )
    for p in polymers:
        for r in routes:
            if r == 'landfill_all':
                c = landfill_cost
            elif r == 'incineration_efw_heat_electricity':
                c = efw_cost
            elif r.startswith('mechanical_recycling'):
                if eligibility[p, r]:
                    entry = params['recycling_costs']['plastic_reprocessing_costs'].get(p)
                    require(entry is not None, f'Missing recycling cost for eligible polymer {p}.')
                    require(
                        entry.get('price_basis') == 'nominal_GBP_2026',
                        f'Mechanical-recycling cost for {p} must be stored on a nominal_GBP_2026 basis.',
                    )
                    require(
                        entry.get('requires_additional_currency_or_price_year_conversion') is False,
                        f'Mechanical-recycling cost for {p} is already converted and must not be converted again.',
                    )
                    c = float(entry['value'])
                    require(
                        math.isclose(c, configured_volk_gbp2026, abs_tol=1e-9),
                        f'Mechanical-recycling cost for {p} is not the documented Volk value.',
                    )
                else:
                    c = 0.0
            else:
                raise KeyError(r)
            route_cost[p, r] = c

            if output_role[p, r] == 'recovered':
                if p in ('LDPE', 'LLDPE'):
                    v = float(params['recycled_ldpe_output_value'][p])
                elif p == 'PP':
                    v = float(params['recycled_pp_output_value'][p])
                elif p == 'TPE':
                    v = float(params['recycled_tpe_output_value'][p])
                else:
                    v = 0.0
                sigma = 1.0  # output_role='recovered' is the supplied saleability/value-capture indicator
            else:
                v = 0.0
                sigma = 0.0
            recovered_value[p, r] = v
            value_capture[p, r] = sigma

    grid_cost = float(params['industrial_grid_electricity_cost']['total_cost_gbp_per_kwh'])
    autoclave_kwh_per_kg = float(params['autoclave_pre_treatment']['electricity_kwh_per_kg_input']['central'])
    autoclave_cost = {j: autoclave_kwh_per_kg * 1000.0 * grid_cost for j in sites}
    reference_ac = float(params['autoclave_pre_treatment']['cost_calculation_rule']['central_reference_cost_gbp_per_tonne_input'])
    require(abs(next(iter(autoclave_cost.values())) - reference_ac) < 1e-9, 'Dynamic autoclave cost does not match reference check.')

    # Site-route distances, preserving the two distinct recycling sinks.
    distance: Dict[Tuple[str, str], float] = {}
    for j in sites:
        loc = site_location[j]
        ddata = params['waste_route_distances'][loc]
        distance[j, 'landfill_all'] = float(ddata['landfill_all']['distance_km'])
        distance[j, 'incineration_efw_heat_electricity'] = float(ddata['incineration_efw_heat_electricity']['distance_km'])
        mech = ddata['mechanical_recycling_main_residual_incineration']
        distance[j, 'mechanical_recycling_polyolefins'] = float(mech['polyolefins']['distance_km'])
        distance[j, 'mechanical_recycling_tpe_tubing'] = float(mech['tpe_tubing']['distance_km'])

    transport_modes = list(params['road_transport_modes'])
    payload = {l: float(params['road_transport_modes'][l]['maximum_payload_tonnes']) for l in transport_modes}

    # The per-vehicle-km transport-service rates are model-ready, all-in coefficients.
    # In particular, the battery-electric HGV rate already includes charging electricity;
    # therefore no electricity-price term is added elsewhere in the objective.
    transport_cost = {
        l: float(params['road_transport_modes'][l]['cost_gbp_per_vehicle_route_km'])
        for l in transport_modes
    }
    bev_params = params['road_transport_modes']['battery_electric_articulated_hgv']
    require(
        not bool(bev_params.get('additional_electricity_cost_in_objective', False)),
        'Battery-electric HGV charging electricity must not be added separately to the objective.',
    )
    require(
        bool(bev_params.get('cost_includes_charging_electricity', True)),
        'Battery-electric HGV all-in transport rate must include charging electricity.',
    )

    # Transport GWP is calculated once. Diesel uses its supplied WTW coefficient.
    # BEV uses socket electricity demand multiplied by the method-consistent UK-grid
    # life-cycle electricity coefficient. The 0.236 kgCO2e/km field is validation-only
    # and is never added to this dynamic result.
    grid_gwp = float(electricity_grid_row['net_gwp_factor_per_optimisation_unit'])
    bev_energy_kwh_per_km = float(bev_params['energy_consumption_kwh_per_vehicle_km'])
    transport_gwp = {
        'diesel_articulated_hgv': float(params['road_transport_modes']['diesel_articulated_hgv']['gwp_kgco2e_per_vehicle_km']),
        'battery_electric_articulated_hgv': bev_energy_kwh_per_km * grid_gwp,
    }
    require(set(transport_gwp) == set(transport_modes), 'Transport GWP coefficients do not cover all transport modes.')
    roundtrip = {(j, r, l): 2.0 for j in sites for r in routes for l in transport_modes}

    electricity_sources = ['grid', 'wind']
    renewable_cost_params = params['industrial_renewable_electricity_cost']
    renewable_central_cost = renewable_cost_params.get('central_value', renewable_cost_params.get('value'))
    require(renewable_central_cost is not None, 'Renewable electricity cost requires central_value or value.')
    electricity_cost = {
        'grid': grid_cost,
        'wind': float(renewable_central_cost),
    }
    electricity_gwp = {
        'grid': grid_gwp,
        'wind': float(electricity_wind_row['net_gwp_factor_per_optimisation_unit']),
    }
    electricity_human_health_daly = {
        'grid': float(electricity_grid_row['net_human_health_daly_factor_per_optimisation_unit']),
        'wind': float(electricity_wind_row['net_human_health_daly_factor_per_optimisation_unit']),
    }
    electricity_ecosystem_quality = {
        'grid': float(electricity_grid_row['net_ecosystem_quality_factor_per_optimisation_unit']),
        'wind': float(electricity_wind_row['net_ecosystem_quality_factor_per_optimisation_unit']),
    }
    electricity_available = {(j, e): 1 for j in sites for e in electricity_sources}

    treatment_fixed = {j: float(wwt['annualised_capex_gbp_per_site_year']) for j in sites}
    treatment_opex = {j: float(wwt['treatment_opex_gbp_per_m3']) for j in sites}
    fresh_water_cost = {j: float(params['avoided_mains_utility_water_cost'][site_location[j]]) for j in sites}
    discharge_cost = {j: float(params['avoided_treated_effluent_discharge_cost'][site_location[j]]) for j in sites}
    treatment_gwp = {j: float(ww_discharge_row['net_gwp_factor_per_optimisation_unit']) for j in sites}
    fresh_water_gwp = {j: float(fresh_water_row['net_gwp_factor_per_optimisation_unit']) for j in sites}
    treatment_human_health_daly = {
        j: float(ww_discharge_row['net_human_health_daly_factor_per_optimisation_unit'])
        for j in sites
    }
    fresh_water_human_health_daly = {
        j: float(fresh_water_row['net_human_health_daly_factor_per_optimisation_unit'])
        for j in sites
    }
    treatment_ecosystem_quality = {
        j: float(ww_discharge_row['net_ecosystem_quality_factor_per_optimisation_unit'])
        for j in sites
    }
    fresh_water_ecosystem_quality = {
        j: float(fresh_water_row['net_ecosystem_quality_factor_per_optimisation_unit'])
        for j in sites
    }

    # Reconcile the fixed WFI proxy with F=F_FIX+f so WFI is not double-counted as generic utility water.
    fixed_per_batch_gwp = (
        float(wfi_row['represented_quantity_per_batch']) * float(wfi_row['net_gwp_factor_per_optimisation_unit'])
        + float(chilled_row['represented_quantity_per_batch']) * float(chilled_row['net_gwp_factor_per_optimisation_unit'])
        + float(cooling_row['represented_quantity_per_batch']) * float(cooling_row['net_gwp_factor_per_optimisation_unit'])
        + float(steam_row['represented_quantity_per_batch']) * float(steam_row['net_gwp_factor_per_optimisation_unit'])
        - fixed_mains_per_batch * float(fresh_water_row['net_gwp_factor_per_optimisation_unit'])
    )
    fixed_manufacturing_gwp = total_batches * fixed_per_batch_gwp
    biomass_gwp = total_batches * float(biomass_row['represented_quantity_per_batch']) * float(biomass_row['net_gwp_factor_per_optimisation_unit'])
    fixed_non_electric_human_health_daly_per_batch = (
        float(wfi_row['represented_quantity_per_batch'])
        * float(wfi_row['net_human_health_daly_factor_per_optimisation_unit'])
        + float(chilled_row['represented_quantity_per_batch'])
        * float(chilled_row['net_human_health_daly_factor_per_optimisation_unit'])
        + float(cooling_row['represented_quantity_per_batch'])
        * float(cooling_row['net_human_health_daly_factor_per_optimisation_unit'])
        + float(steam_row['represented_quantity_per_batch'])
        * float(steam_row['net_human_health_daly_factor_per_optimisation_unit'])
        - fixed_mains_per_batch
        * float(fresh_water_row['net_human_health_daly_factor_per_optimisation_unit'])
    )
    biomass_human_health_daly_per_batch = (
        float(biomass_row['represented_quantity_per_batch'])
        * float(biomass_row['net_human_health_daly_factor_per_optimisation_unit'])
    )
    fixed_non_electric_ecosystem_quality_per_batch = (
        float(wfi_row['represented_quantity_per_batch'])
        * float(wfi_row['net_ecosystem_quality_factor_per_optimisation_unit'])
        + float(chilled_row['represented_quantity_per_batch'])
        * float(chilled_row['net_ecosystem_quality_factor_per_optimisation_unit'])
        + float(cooling_row['represented_quantity_per_batch'])
        * float(cooling_row['net_ecosystem_quality_factor_per_optimisation_unit'])
        + float(steam_row['represented_quantity_per_batch'])
        * float(steam_row['net_ecosystem_quality_factor_per_optimisation_unit'])
        - fixed_mains_per_batch
        * float(fresh_water_row['net_ecosystem_quality_factor_per_optimisation_unit'])
    )
    biomass_cost = sum(site_open.values()) * float(params['external_biomass_ad_collection']['annual_collection_cost_gbp_per_active_site_year'])
    carbon_value = float(params['lifecycle_carbon_value']['central_value']) * m2020real
    total_product_kg = sum(batches[j] * product_kg_per_batch[j] for j in sites)

    audit_rows = [
        ('sites', len(sites), 'count', 'supply-chain CSV'),
        ('annual_batches_total', total_batches, 'batches/year', 'supply-chain CSV; fixed total'),
        ('batch_level_high', batch_high, 'batches/year/site', 'supply-chain CSV; retained allocation level'),
        ('batch_level_low', batch_low, 'batches/year/site', 'supply-chain CSV; location selected by optimiser'),
        ('batch_level_difference', batch_delta, 'batches/year/site', 'derived high minus low'),
        ('saleable_mab_per_batch', pi, 'kg/batch', 'SP master Product/mAb row'),
        ('saleable_mab_total', total_product_kg, 'kg/year', 'derived'),
        ('sut_plastic_total_per_batch', sum(polymer_kg_per_batch.values()), 'kg/batch', 'LCA route-factor workbook'),
        ('ecoinvent_version', 3.12, 'version', 'LCA route-factor workbook; required journal-paper database'),
        ('gwp_method', expected_gwp_method, 'method', 'IMPACT World+ v2.1 short-term carbon footprint; IPCC AR6'),
        ('water_scarcity_method', expected_water_method, 'method', 'IMPACT World+ v2.1 AWARE 2.0'),
        ('water_scarcity_regionalisation', 'foreground tap-water columns only', 'scope', 'site AWARE 2.0 basin factor; background retains native Brightway geography'),
        ('sut_manufacturing_gwp_adder', 1.9, 'kgCO2e/kg SUT plastic', 'LCA route-factor workbook'),
        ('sut_manufacturing_gwp_adder_source', 'Reiners et al. (2026), citing Ramasamy (2018)', 'reference', 'LCA route-factor workbook'),
        ('human_health_daly_method', expected_human_health_method, 'method', 'LCA route-factor workbook; reporting only'),
        ('human_health_daly_equation', 'human_health_daly = sum(midpoint_score_i * human_health_damage_factor_i)', 'equation', 'LCA route-factor workbook; reporting only'),
        ('human_health_daly_midpoints', ' | '.join(sorted(required_daly_midpoints)), 'pathways', 'LCA route-factor workbook; reporting only'),
        ('human_health_daly_direct_endpoint_role', 'validation_only_not_added_to_reported_daly', 'role', 'LCA route-factor workbook; reporting only'),
        ('ecosystem_quality_method', expected_ecosystem_method, 'method', 'LCA route-factor workbook; reporting only'),
        ('ecosystem_quality_unit', 'species.yr', 'unit', 'ecoinvent ReCiPe 2016 Endpoint (H)'),
        ('mechanical_recycling_cost_original', original_eur2019, 'EUR2019/t input', 'Volk et al. (2021); Plasticker 2019 basis'),
        ('mechanical_recycling_inflation_multiplier', inflation_multiplier, '2019 to August 2026', 'Eurostat German all-items HICP'),
        ('mechanical_recycling_fx', fx_gbp_per_eur, 'GBP/EUR', f"ECB reference rate {volk['fx_date']}"),
        ('mechanical_recycling_cost_2026', configured_volk_gbp2026, 'GBP2026/t input', 'Volk value after German HICP escalation and ECB FX conversion'),
        ('electricity_per_batch', electricity_per_batch, 'kWh/batch', 'LCA route-factor workbook'),
        ('wastewater_per_batch', wastewater_per_batch, 'm3/batch', 'LCA route-factor workbook'),
        ('eligible_utility_water_per_batch', utility_water_per_batch, 'm3/batch', 'LCA route-factor workbook'),
        ('fixed_wfi_mains_per_batch', fixed_mains_per_batch, 'm3/batch', 'WFI kg converted at 1000 kg/m3'),
        ('water_recovery_fraction', eta_w, 'fraction', 'LCA input-output table'),
        ('carbon_value_2026', carbon_value, 'GBP_2026/tCO2e', 'MILP params conversion rule'),
        ('landfill_cost_2026', landfill_cost, 'GBP_2026/t', '26*1.022 + 130.75 special rule'),
        ('efw_cost_2026', efw_cost, 'GBP_2026/t', '121*1.022'),
        ('diesel_transport_cost_model_coefficient', transport_cost['diesel_articulated_hgv'], 'GBP/vehicle-km', 'Direct all-in service rate; no price-year multiplier'),
        ('bev_transport_cost_model_coefficient', transport_cost['battery_electric_articulated_hgv'], 'GBP/vehicle-km', 'Direct all-in service rate including charging electricity; no additional charging cost'),
        ('bev_transport_energy_for_gwp_only', bev_energy_kwh_per_km, 'kWh/vehicle-km', 'Socket demand including charging losses; environmental calculation only'),
        ('bev_transport_gwp_dynamic', transport_gwp['battery_electric_articulated_hgv'], 'kgCO2e/vehicle-km', '1.8 kWh/km * method-consistent UK-grid life-cycle electricity factor; 0.236 reference not added'),
        ('bev_reference_grid_generation_gwp_validation_only', float(bev_params.get('reference_uk_grid_gwp_kgco2e_per_vehicle_km', 0.236)), 'kgCO2e/vehicle-km', 'Generation-only validation field; excluded from objective and LCIA total'),
    ]
    for location, factor in expected_site_factors.items():
        audit_rows.append((f'aware20_site_cf::{location}', factor, 'm3 world-eq/m3', 'AWARE 2.0 annual non-agricultural basin factor'))
    audit_rows.extend([
        ('biomass_kg_per_batch', biomass_kg_per_batch, 'kg/batch', 'active LCA route-factor workbook'),
        ('biomethane_m3_per_batch', biomethane_m3_per_batch, 'm3/batch', 'active LCA input-output table'),
        ('biomethane_ncv', biomethane_ncv, 'MJ/m3', 'electrcity_biomass_route_assumptions.json'),
        ('biomethane_to_useful_steam_efficiency', heat_efficiency, 'fraction', 'electrcity_biomass_route_assumptions.json; illustrative'),
        ('external_useful_steam', heat_mj_per_kg * biomass_kg_per_batch, 'MJ/batch', 'uncapped biomethane energy times efficiency'),
    ])
    for route in biomass_routes:
        for impact, factor in biomass_route_factors[route].items():
            audit_rows.append((f'biomass_factor::{route}::{impact}', factor,
                               f'{impact}/kg biomass', 'active scored LCA stages; gas supply replaced by external steam for heat route'))
    for p in polymers:
        for r in routes:
            if eligibility[p, r] and recovery_yield[p, r] > 0:
                audit_rows.append((f'recovery_yield::{p}::{r}', recovery_yield[p, r], 'fraction', 'explicit physical recovery parameter from LCA workbook'))
                audit_rows.append((f'substitution_ratio::{p}::{r}', substitution_ratio[p, r], 'kg virgin/kg recovered recyclate', 'explicit 0.81 quality/downcycling parameter'))
    audit = pd.DataFrame(audit_rows, columns=['parameter', 'value', 'unit', 'source_or_derivation'])

    return ModelData(
        sites=sites,
        site_location=site_location,
        polymers=polymers,
        routes=routes,
        route_lca_pathway=route_lca_pathway,
        transport_modes=transport_modes,
        electricity_sources=electricity_sources,
        batches=batches,
        batch_high=batch_high,
        batch_low=batch_low,
        batch_delta=batch_delta,
        total_batches=total_batches,
        site_open=site_open,
        product_kg_per_batch=product_kg_per_batch,
        electricity_kwh_per_batch=electricity_per_batch,
        wastewater_m3_per_batch=wastewater_per_batch,
        utility_water_m3_per_batch=utility_water_per_batch,
        fixed_mains_water_m3_per_batch=fixed_mains_per_batch,
        electricity_kwh=electricity_kwh,
        wastewater_m3=wastewater_m3,
        utility_water_m3=utility_water_m3,
        fixed_mains_water_m3=fixed_mains_water_m3,
        water_recovery=water_recovery,
        wwt_capacity_m3=wwt_capacity,
        polymer_kg_per_batch=polymer_kg_per_batch,
        eligibility=eligibility,
        recovery_yield=recovery_yield,
        substitution_ratio=substitution_ratio,
        residual_fraction=residual_fraction,
        route_cost_gbp_per_t=route_cost,
        autoclave_required=autoclave_required,
        autoclave_cost_gbp_per_t=autoclave_cost,
        recovered_value_gbp_per_t=recovered_value,
        value_capture=value_capture,
        distance_km=distance,
        roundtrip_multiplier=roundtrip,
        payload_t=payload,
        transport_cost_gbp_per_km=transport_cost,
        transport_gwp_kg_per_km=transport_gwp,
        electricity_cost_gbp_per_kwh=electricity_cost,
        electricity_gwp_kg_per_kwh=electricity_gwp,
        electricity_available=electricity_available,
        treatment_fixed_cost_gbp=treatment_fixed,
        treatment_opex_gbp_per_m3=treatment_opex,
        fresh_water_cost_gbp_per_m3=fresh_water_cost,
        discharge_cost_gbp_per_m3=discharge_cost,
        treatment_gwp_kg_per_m3=treatment_gwp,
        fresh_water_gwp_kg_per_m3=fresh_water_gwp,
        plastic_gwp_kg_per_kg=plastic_gwp,
        virgin_polymer_gwp_kg_per_kg=virgin_polymer_gwp,
        sut_component_manufacturing_gwp_kg_per_kg=sut_component_manufacturing_gwp,
        avoided_virgin_material_credit_gwp_kg_per_kg=avoided_virgin_material_credit_gwp,
        plastic_human_health_daly_per_kg=plastic_human_health_daly,
        electricity_human_health_daly_per_kwh=electricity_human_health_daly,
        treatment_human_health_daly_per_m3=treatment_human_health_daly,
        fresh_water_human_health_daly_per_m3=fresh_water_human_health_daly,
        fixed_non_electric_human_health_daly_per_batch=fixed_non_electric_human_health_daly_per_batch,
        biomass_human_health_daly_per_batch=biomass_human_health_daly_per_batch,
        plastic_ecosystem_quality_per_kg=plastic_ecosystem_quality,
        electricity_ecosystem_quality_per_kwh=electricity_ecosystem_quality,
        treatment_ecosystem_quality_per_m3=treatment_ecosystem_quality,
        fresh_water_ecosystem_quality_per_m3=fresh_water_ecosystem_quality,
        fixed_non_electric_ecosystem_quality_per_batch=fixed_non_electric_ecosystem_quality_per_batch,
        fixed_manufacturing_gwp_kg=fixed_manufacturing_gwp,
        biomass_gwp_kg=biomass_gwp,
        biomass_cost_gbp=biomass_cost,
        biomass_kg_per_batch=biomass_kg_per_batch,
        biomass_routes=biomass_routes,
        biomass_route_factors=biomass_route_factors,
        biomass_route_cost_gbp_per_kg=biomass_route_cost_gbp_per_kg,
        biomass_useful_heat_mj_per_kg=biomass_useful_heat_mj_per_kg,
        lifecycle_carbon_value_gbp_per_t=carbon_value,
        total_product_kg=total_product_kg,
        route_capacities_t={},
        audit=audit,
    )

def build_linear_model(data: ModelData) -> LinearModel:
    var_names: List[str] = []
    var_keys: List[Tuple] = []
    var_types: List[str] = []
    lower: List[float] = []
    upper: List[float] = []
    integrality: List[int] = []
    index: Dict[Tuple, int] = {}

    def add_var(key: Tuple, name: str, vartype: str = 'C', lb: float = 0.0, ub: float = np.inf) -> int:
        idx = len(var_names)
        index[key] = idx
        var_keys.append(key)
        var_names.append(name)
        var_types.append(vartype)
        lower.append(lb)
        upper.append(ub)
        integrality.append(1 if vartype == 'I' else 0)
        return idx

    # Endogenous batch allocation. B[j] is the annual site batch count and low_batch[j]
    # identifies the single active site receiving the low upstream allocation.
    for j in data.sites:
        add_var(('B', j), f'B[{j}]', vartype='I', lb=0.0, ub=data.batch_high * data.site_open[j])
        add_var(('low_batch', j), f'low_batch[{j}]', vartype='I', lb=0.0, ub=float(data.site_open[j]))
        for route in data.biomass_routes:
            add_var(('bio_y', j, route), f'bio_y[{j},{route}]', vartype='I', ub=float(data.site_open[j]))
            add_var(('bio_q', j, route), f'bio_q[{j},{route}]',
                    ub=data.batch_high * data.biomass_kg_per_batch * data.site_open[j])

    for j in data.sites:
        for p in data.polymers:
            for r in data.routes:
                if data.eligibility[p, r]:
                    for l in data.transport_modes:
                        add_var(('x', j, p, r, l), f'x[{j},{p},{r},{l}]')
    # A route cannot require more journeys than moving the site's entire annual
    # plastic inventory by that mode. This is a mathematically valid finite bound
    # (not a modelling assumption) and prevents GAMS replacing unbounded integer
    # upper bounds with 1E100.
    maximum_site_plastic_t = (
        data.batch_high * sum(data.polymer_kg_per_batch.values()) / 1000.0
    )
    for j in data.sites:
        for r in data.routes:
            if any(data.eligibility[p, r] for p in data.polymers):
                for l in data.transport_modes:
                    maximum_journeys = math.ceil(maximum_site_plastic_t / data.payload_t[l])
                    add_var(
                        ('n', j, r, l),
                        f'n[{j},{r},{l}]',
                        vartype='I',
                        ub=float(maximum_journeys),
                    )
    for j in data.sites:
        for e in data.electricity_sources:
            ub = data.batch_high * data.electricity_kwh_per_batch if data.electricity_available[j, e] else 0.0
            add_var(('E', j, e), f'E[{j},{e}]', ub=ub)
    for j in data.sites:
        for v in ['qDIR', 'qREU', 'u', 'f', 'd']:
            add_var((v, j), f'{v}[{j}]')

    nvar = len(var_names)
    rows: List[Dict[int, float]] = []
    bl: List[float] = []
    bu: List[float] = []

    def add_constraint(coeffs: Mapping[int, float], lb: float, ub: float) -> None:
        rows.append(dict(coeffs)); bl.append(lb); bu.append(ub)

    # Exactly one active site receives the low batch allocation.
    add_constraint({index['low_batch', j]: 1.0 for j in data.sites}, 1.0, 1.0)
    for j in data.sites:
        # B_j = B_high*y_j - (B_high-B_low)*low_batch_j.
        add_constraint(
            {index['B', j]: 1.0, index['low_batch', j]: data.batch_delta},
            data.batch_high * data.site_open[j],
            data.batch_high * data.site_open[j],
        )
    # Explicit total-batch conservation check (redundant but transparent and protective).
    add_constraint({index['B', j]: 1.0 for j in data.sites}, data.total_batches, data.total_batches)
    # One exclusive AD destination for each active site's entire biomass output.
    # The flow is tied to endogenous site batches, so route LCIA changes with allocation.
    for j in data.sites:
        add_constraint({index['bio_y', j, r]: 1.0 for r in data.biomass_routes},
                       data.site_open[j], data.site_open[j])
        flow = {index['bio_q', j, r]: 1.0 for r in data.biomass_routes}
        flow[index['B', j]] = -data.biomass_kg_per_batch
        add_constraint(flow, 0.0, 0.0)
        for route in data.biomass_routes:
            add_constraint({index['bio_q', j, route]: 1.0,
                            index['bio_y', j, route]: -data.batch_high * data.biomass_kg_per_batch},
                           -np.inf, 0.0)

    # Full polymer routing, with site generation driven by endogenous annual batches.
    for j in data.sites:
        for p in data.polymers:
            coeff = {index['B', j]: -data.polymer_kg_per_batch[p] / 1000.0}
            for r in data.routes:
                if data.eligibility[p, r]:
                    for l in data.transport_modes:
                        coeff[index['x', j, p, r, l]] = 1.0
            add_constraint(coeff, 0.0, 0.0)

    # Vehicle payload. BAU and all optimised cases therefore include transport explicitly.
    for j in data.sites:
        for r in data.routes:
            if any(data.eligibility[p, r] for p in data.polymers):
                for l in data.transport_modes:
                    coeff = {index['n', j, r, l]: -data.payload_t[l]}
                    for p in data.polymers:
                        if data.eligibility[p, r]:
                            coeff[index['x', j, p, r, l]] = 1.0
                    add_constraint(coeff, -np.inf, 0.0)

    # Optional route capacities.
    for r, cap in data.route_capacities_t.items():
        coeff = {}
        for j in data.sites:
            for p in data.polymers:
                if data.eligibility[p, r]:
                    for l in data.transport_modes:
                        coeff[index['x', j, p, r, l]] = 1.0
        add_constraint(coeff, -np.inf, cap)

    # Electricity demand is proportional to the selected site batch count.
    for j in data.sites:
        coeff = {index['E', j, e]: 1.0 for e in data.electricity_sources}
        coeff[index['B', j]] = -data.electricity_kwh_per_batch
        add_constraint(coeff, 0.0, 0.0)

    # Wastewater and water balances are proportional to the selected site batch count.
    for j in data.sites:
        add_constraint(
            {index['qDIR', j]: 1.0, index['qREU', j]: 1.0, index['B', j]: -data.wastewater_m3_per_batch},
            0.0, 0.0,
        )
        add_constraint({index['u', j]: 1.0, index['qREU', j]: -data.water_recovery[j]}, 0.0, 0.0)
        add_constraint(
            {index['f', j]: 1.0, index['u', j]: 1.0, index['B', j]: -data.utility_water_m3_per_batch},
            0.0, 0.0,
        )
        add_constraint(
            {index['d', j]: 1.0, index['u', j]: 1.0, index['B', j]: -data.wastewater_m3_per_batch},
            0.0, 0.0,
        )
        add_constraint(
            {index['B', j]: data.wastewater_m3_per_batch},
            -np.inf,
            data.wwt_capacity_m3[j] * data.site_open[j],
        )

    # Sparse constraint matrix.
    rr, cc, vv = [], [], []
    for i, row in enumerate(rows):
        for k, v in row.items():
            rr.append(i); cc.append(k); vv.append(v)
    A = coo_matrix((vv, (rr, cc)), shape=(len(rows), nvar))

    # Linear expressions.
    def zeros() -> np.ndarray:
        return np.zeros(nvar, dtype=float)

    c_cost = zeros()
    cost_const = data.biomass_cost_gbp
    for j in data.sites:
        cost_const += data.treatment_fixed_cost_gbp[j] * data.site_open[j]
        c_cost[index['B', j]] += data.treatment_opex_gbp_per_m3[j] * data.wastewater_m3_per_batch
    for key, idx in index.items():
        if key[0] == 'x':
            _, j, p, r, l = key
            c_cost[idx] = (
                data.route_cost_gbp_per_t[p, r]
                + data.autoclave_required[p, r] * data.autoclave_cost_gbp_per_t[j]
                - data.value_capture[p, r] * data.recovered_value_gbp_per_t[p, r] * data.recovery_yield[p, r]
            )
        elif key[0] == 'n':
            _, j, r, l = key
            c_cost[idx] = data.roundtrip_multiplier[j, r, l] * data.distance_km[j, r] * data.transport_cost_gbp_per_km[l]
        elif key[0] == 'E':
            _, j, e = key
            c_cost[idx] = data.electricity_cost_gbp_per_kwh[e]
        elif key[0] == 'f':
            _, j = key
            c_cost[idx] = data.fresh_water_cost_gbp_per_m3[j]
        elif key[0] == 'd':
            _, j = key
            c_cost[idx] = data.discharge_cost_gbp_per_m3[j]
        elif key[0] == 'bio_q':
            _, j, route = key
            c_cost[idx] = data.biomass_route_cost_gbp_per_kg[route]

    c_gwp = zeros()
    gwp_const = data.fixed_manufacturing_gwp_kg
    for j in data.sites:
        c_gwp[index['B', j]] += (
            data.treatment_gwp_kg_per_m3[j] * data.wastewater_m3_per_batch
            + data.fresh_water_gwp_kg_per_m3[j] * data.fixed_mains_water_m3_per_batch
        )
    for key, idx in index.items():
        if key[0] == 'x':
            _, j, p, r, l = key
            c_gwp[idx] = 1000.0 * data.plastic_gwp_kg_per_kg[p, r]
        elif key[0] == 'n':
            _, j, r, l = key
            c_gwp[idx] = data.roundtrip_multiplier[j, r, l] * data.distance_km[j, r] * data.transport_gwp_kg_per_km[l]
        elif key[0] == 'E':
            _, j, e = key
            c_gwp[idx] = data.electricity_gwp_kg_per_kwh[e]
        elif key[0] == 'f':
            _, j = key
            c_gwp[idx] = data.fresh_water_gwp_kg_per_m3[j]
        elif key[0] == 'bio_q':
            _, j, route = key
            c_gwp[idx] = data.biomass_route_factors[route]['gwp']

    c_fw = zeros()
    for j in data.sites:
        c_fw[index['B', j]] = data.fixed_mains_water_m3_per_batch
        c_fw[index['f', j]] = 1.0

    c_rec = zeros()
    for key, idx in index.items():
        if key[0] == 'x':
            _, j, p, r, l = key
            c_rec[idx] = data.recovery_yield[p, r]
    total_plastic_t = data.total_batches * sum(data.polymer_kg_per_batch.values()) / 1000.0
    c_circ = -c_rec
    circ_const = total_plastic_t

    c_res = zeros()
    for key, idx in index.items():
        if key[0] == 'x':
            _, j, p, r, l = key
            c_res[idx] = data.residual_fraction[p, r]

    expressions = {
        'cost': LinearExpression(c_cost, cost_const),
        'gwp': LinearExpression(c_gwp, gwp_const),
        'circularity_loss': LinearExpression(c_circ, circ_const),
        'freshwater': LinearExpression(c_fw, 0.0),
        'residual': LinearExpression(c_res, 0.0),
        'recovered': LinearExpression(c_rec, 0.0),
    }
    policy_coeff = c_cost + data.lifecycle_carbon_value_gbp_per_t / 1000.0 * c_gwp
    policy_const = cost_const + data.lifecycle_carbon_value_gbp_per_t / 1000.0 * gwp_const
    expressions['policy_cost'] = LinearExpression(policy_coeff, policy_const)

    return LinearModel(
        data=data,
        var_names=var_names,
        var_keys=var_keys,
        var_types=var_types,
        index=index,
        lower=np.array(lower),
        upper=np.array(upper),
        integrality=np.array(integrality),
        A=A,
        b_l=np.array(bl),
        b_u=np.array(bu),
        expressions=expressions,
    )

def _assemble_problem(model: LinearModel, primary: str) -> Tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    """Return the unchanged core MILP with one selected scalar objective."""
    require(primary in model.expressions, f'Unknown objective: {primary}')
    return (
        model.expressions[primary].coeff.copy(),
        model.A.tocsr(),
        model.b_l.copy(),
        model.b_u.copy(),
    )

def _safe_solve_token(value: str) -> str:
    """Return a filesystem-safe case token for GAMS logs and work files."""
    cleaned = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in str(value))
    return cleaned.strip('_')[:120] or 'solve'

def _selected_solver_probe_model(number_of_variables: int = 1) -> str:
    """Small MIP used for the explicit GAMS/solver preflight check."""
    n = max(1, int(number_of_variables))
    return f"""
option lp={GAMS_SOLVER};
option mip={GAMS_SOLVER};
option limrow=0;
option limcol=0;
option solprint=off;
set I /1*{n}/;
binary variable y(I);
variable objective_value;
equation objective_definition;
objective_definition.. objective_value =e= sum(I, y(I));
model selected_solver_test /all/;
solve selected_solver_test using mip minimizing objective_value;
abort$((selected_solver_test.solvestat <> 1))
    'Selected GAMS MIP solver did not terminate normally', selected_solver_test.solvestat;
abort$((selected_solver_test.modelstat <> 1) and (selected_solver_test.modelstat <> 8))
    'Selected GAMS MIP solver did not return a globally optimal integer solution', selected_solver_test.modelstat;
"""

def _run_gams_preflight(gams_executable: str) -> None:
    """Run a visible, persisted GAMS/CPLEX check before Pyomo's opaque probe."""
    global _GAMS_PREFLIGHT_DONE
    if _GAMS_PREFLIGHT_DONE or not GAMS_PREFLIGHT:
        return

    preflight_dir = GAMS_LOG_DIR / 'preflight_work'
    if preflight_dir.exists():
        shutil.rmtree(preflight_dir)
    preflight_dir.mkdir(parents=True, exist_ok=True)
    model_file = preflight_dir / 'selected_solver_test.gms'
    listing_file = preflight_dir / 'selected_solver_test.lst'
    model_file.write_text(_selected_solver_probe_model(), encoding='utf-8')

    command = [
        str(gams_executable),
        model_file.name,
        f'curdir={preflight_dir}',
        'lo=3',
        f'o={listing_file}',
    ]
    completed = subprocess.run(
        command,
        cwd=preflight_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ''
    GAMS_PREFLIGHT_LOG.write_text(
        'COMMAND\n' + ' '.join(command) + '\n\nOUTPUT\n' + output,
        encoding='utf-8',
    )
    print('\n========== GAMS PREFLIGHT ==========')
    print(output.rstrip())
    print(f'Persisted preflight log: {GAMS_PREFLIGHT_LOG}')
    print('====================================\n')

    if completed.returncode != 0:
        tail = '\n'.join(output.splitlines()[-60:])
        raise RuntimeError(
            'The GAMS executable was found, but the selected GAMS MIP solver '
            f'{GAMS_SOLVER!r} failed the explicit preflight test '
            f'(return code {completed.returncode}).\n'
            f'Full diagnostic log: {GAMS_PREFLIGHT_LOG}\n\nLast GAMS lines:\n{tail}'
        )
    _GAMS_PREFLIGHT_DONE = True
    if not GAMS_KEEPFILES:
        shutil.rmtree(preflight_dir, ignore_errors=True)

def _patch_pyomo_gams_availability_probe(solver: Any) -> None:
    """
    Make Pyomo's internal availability test use the selected licensed solver.

    Pyomo's legacy GAMS interface otherwise solves its probe with the GAMS
    default LP solver, which can fail even when the requested MIP solver is
    installed and licensed. This changes only the availability probe, not the
    optimisation model.
    """
    original = getattr(solver, '_simple_model', None)
    if original is None:
        return

    def selected_solver_simple_model(n: int) -> str:
        return (
            f'option lp={GAMS_SOLVER};\n'
            f'option mip={GAMS_SOLVER};\n'
            + original(n)
        )

    solver._simple_model = selected_solver_simple_model

def _result_value(container: Any, attribute: str) -> Any:
    try:
        value = getattr(container, attribute)
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        return str(value) if value is not None else None

def _append_gams_solve_report(row: Mapping[str, Any]) -> None:
    columns = [
        'solve_number', 'case_id', 'solver', 'solver_io', 'solver_status',
        'termination_condition', 'elapsed_seconds_python', 'solver_time_seconds',
        'user_time_seconds', 'system_time_seconds', 'wallclock_time_seconds',
        'lower_bound', 'upper_bound', 'number_of_variables',
        'number_of_integer_variables', 'number_of_constraints', 'log_file',
        'summary_file', 'work_directory', 'exception',
    ]
    write_header = not GAMS_SOLVE_REPORT.exists()
    with GAMS_SOLVE_REPORT.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column) for column in columns})

def _solve_pyomo_gams(
    model: LinearModel,
    case_id: str,
    primary: str,
    selection_method: str = '',
    metadata: Optional[Dict[str, Any]] = None,
) -> SolveResult:
    """Build the unchanged MILP in Pyomo and solve it through GAMS."""
    global _GAMS_SOLVE_COUNTER

    import pyomo.environ as pyo
    from pyomo.common import Executable
    from pyomo.opt import SolverFactory, TerminationCondition

    require(GAMS_SOLVER_IO in {'shell', 'direct', 'python'},
            f'Unsupported GAMS_SOLVER_IO={GAMS_SOLVER_IO!r}; use shell or direct.')

    gams_executable = shutil.which('gams')
    if gams_executable is None:
        raise RuntimeError(
            'GAMS is not visible to Python. Expected the executable under '
            f'{GAMS_RESOURCE_DIR}, or add the GAMS Resources directory to PATH.'
        )
    # Pin Pyomo's executable cache to the same absolute GAMS installation.
    try:
        Executable('gams').set_path(str(gams_executable))
    except Exception:
        pass

    _run_gams_preflight(gams_executable)

    c, A, b_l, b_u = _assemble_problem(model, primary)
    A = A.tocsr()
    m = pyo.ConcreteModel(name=f'industrial_symbiosis_{case_id}')
    m.I = pyo.RangeSet(0, len(model.var_names) - 1)

    def domain_rule(_m, i):
        return pyo.NonNegativeIntegers if model.integrality[i] else pyo.NonNegativeReals

    def bounds_rule(_m, i):
        lb = None if not np.isfinite(model.lower[i]) else float(model.lower[i])
        ub = None if not np.isfinite(model.upper[i]) else float(model.upper[i])
        return lb, ub

    m.z = pyo.Var(m.I, domain=domain_rule, bounds=bounds_rule)
    m.core = pyo.ConstraintList()
    for row in range(A.shape[0]):
        row_start, row_stop = A.indptr[row], A.indptr[row + 1]
        expr = sum(
            float(A.data[k]) * m.z[int(A.indices[k])]
            for k in range(row_start, row_stop)
        )
        lb, ub = float(b_l[row]), float(b_u[row])
        if np.isfinite(lb) and np.isfinite(ub) and abs(lb - ub) <= 1e-10:
            m.core.add(expr == lb)
        else:
            if np.isfinite(lb):
                m.core.add(expr >= lb)
            if np.isfinite(ub):
                m.core.add(expr <= ub)

    m.objective = pyo.Objective(
        expr=sum(float(c[i]) * m.z[i] for i in range(len(c))),
        sense=pyo.minimize,
    )

    # Use the explicit GAMS interface and select the subsolver in GAMS options.
    # This is easier to switch and avoids reliance on a SolverFactory alias.
    solver = SolverFactory('gams', solver_io=GAMS_SOLVER_IO)
    if solver is None:
        raise RuntimeError("SolverFactory('gams') returned None.")
    _patch_pyomo_gams_availability_probe(solver)

    gams_options = [
        f'option lp={GAMS_SOLVER};',
        f'option mip={GAMS_SOLVER};',
        f'option optcr={MIP_REL_GAP};',
        'option optca=0;',
    ]
    if SOLVE_TIME_LIMIT_SECONDS is not None:
        gams_options.append(f'option reslim={int(SOLVE_TIME_LIMIT_SECONDS)};')

    if _GAMS_SOLVE_COUNTER == 0 and GAMS_SOLVE_REPORT.exists():
        try:
            prior = pd.read_csv(GAMS_SOLVE_REPORT, usecols=['solve_number'])
            if not prior.empty:
                _GAMS_SOLVE_COUNTER = int(pd.to_numeric(prior['solve_number'], errors='coerce').max())
        except Exception:
            _GAMS_SOLVE_COUNTER = 0
    _GAMS_SOLVE_COUNTER += 1
    token = _safe_solve_token(case_id)
    prefix = f'{_GAMS_SOLVE_COUNTER:04d}_{token}_{GAMS_SOLVER}'
    log_file = GAMS_LOG_DIR / f'{prefix}.log'
    summary_file = GAMS_LOG_DIR / f'{prefix}_solve_report.txt'

    temporary_work_directory = None
    if GAMS_KEEPFILES:
        work_directory = GAMS_WORK_DIR / prefix
        work_directory.mkdir(parents=True, exist_ok=True)
    else:
        temporary_work_directory = tempfile.TemporaryDirectory(prefix=f'is_gams_{prefix}_')
        work_directory = Path(temporary_work_directory.name)

    started = time.perf_counter()
    results = None
    exception_text = ''
    try:
        print(f'\n========== SOLVING {case_id} WITH GAMS/{GAMS_SOLVER.upper()} ==========')
        print(f'GAMS interface: {GAMS_SOLVER_IO}')
        print(f'GAMS log file: {log_file}')
        results = solver.solve(
            m,
            tee=GAMS_TEE,
            logfile=str(log_file),
            keepfiles=GAMS_KEEPFILES,
            tmpdir=str(work_directory),
            report_timing=True,
            symbolic_solver_labels=True,
            add_options=gams_options,
        )
        print('\nPYOMO/GAMS SOLVER REPORT')
        print(results)
        summary_file.write_text(str(results), encoding='utf-8')
    except Exception as exc:
        exception_text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        summary_file.write_text(exception_text, encoding='utf-8')
        raise RuntimeError(
            f'GAMS/{GAMS_SOLVER} solve failed for {case_id}. '
            f'Preflight: {GAMS_PREFLIGHT_LOG}; solve report: {summary_file}; '
            f'GAMS log: {log_file}'
        ) from exc
    finally:
        elapsed = time.perf_counter() - started
        solver_record = getattr(results, 'solver', None) if results is not None else None
        problem_record = getattr(results, 'problem', None) if results is not None else None
        _append_gams_solve_report({
            'solve_number': _GAMS_SOLVE_COUNTER,
            'case_id': case_id,
            'solver': GAMS_SOLVER,
            'solver_io': GAMS_SOLVER_IO,
            'solver_status': str(getattr(solver_record, 'status', '')) if solver_record is not None else '',
            'termination_condition': str(getattr(solver_record, 'termination_condition', '')) if solver_record is not None else '',
            'elapsed_seconds_python': elapsed,
            'solver_time_seconds': _result_value(solver_record, 'time') if solver_record is not None else None,
            'user_time_seconds': _result_value(solver_record, 'user_time') if solver_record is not None else None,
            'system_time_seconds': _result_value(solver_record, 'system_time') if solver_record is not None else None,
            'wallclock_time_seconds': _result_value(solver_record, 'wallclock_time') if solver_record is not None else None,
            'lower_bound': _result_value(problem_record, 'lower_bound') if problem_record is not None else None,
            'upper_bound': _result_value(problem_record, 'upper_bound') if problem_record is not None else None,
            'number_of_variables': _result_value(problem_record, 'number_of_variables') if problem_record is not None else None,
            'number_of_integer_variables': _result_value(problem_record, 'number_of_integer_variables') if problem_record is not None else None,
            'number_of_constraints': _result_value(problem_record, 'number_of_constraints') if problem_record is not None else None,
            'log_file': str(log_file),
            'summary_file': str(summary_file),
            'work_directory': str(work_directory) if GAMS_KEEPFILES else '',
            'exception': exception_text,
        })
        if temporary_work_directory is not None:
            temporary_work_directory.cleanup()

    term = results.solver.termination_condition
    success = term in {
        TerminationCondition.optimal,
        TerminationCondition.globallyOptimal,
    }
    if not success:
        return SolveResult(
            case_id=case_id,
            success=False,
            status=1,
            message=f'{results.solver.status}; {term}',
            x=np.full(len(model.var_names), np.nan),
            metrics={},
            raw_objective=np.nan,
            primary_objective=primary,
            selection_method=selection_method,
            metadata={
                **dict(metadata or {}),
                'solver_backend': f'pyomo_gams_{GAMS_SOLVER}',
                'gams_log_file': str(log_file),
                'gams_solve_report_file': str(summary_file),
            },
        )

    x = np.array([pyo.value(m.z[i]) for i in range(len(model.var_names))], dtype=float)
    metrics = {name: expression.evaluate(x) for name, expression in model.expressions.items()}
    meta = dict(metadata or {})
    meta['solver_backend'] = f'pyomo_gams_{GAMS_SOLVER}'
    meta['gams_solver_name'] = GAMS_SOLVER_NAME
    meta['gams_solver_io'] = GAMS_SOLVER_IO
    meta['gams_executable'] = gams_executable
    meta['gams_log_file'] = str(log_file)
    meta['gams_solve_report_file'] = str(summary_file)
    return SolveResult(
        case_id=case_id,
        success=True,
        status=0,
        message=f'{results.solver.status}; {term}',
        x=x,
        metrics=metrics,
        raw_objective=float(pyo.value(m.objective)),
        primary_objective=primary,
        selection_method=selection_method,
        metadata=meta,
    )

def solve_model(
    model: LinearModel,
    case_id: str,
    primary: str,
    selection_method: str = '',
    metadata: Optional[Dict[str, Any]] = None,
) -> SolveResult:
    """Solve the core MILP with one selected scalar objective."""
    def canonicalise_cost_tie(result: SolveResult) -> SolveResult:
        # The two AD routes have identical private costs. Report the lower-GWP
        # member of that exact cost-optimal tie instead of a solver-dependent
        # year-to-year switch with no economic meaning.
        if primary != 'cost' or not result.success:
            return result
        data = model.data
        route = min(data.biomass_routes, key=lambda r: data.biomass_route_factors[r]['gwp'])
        before = result.metrics['cost']
        for site in data.sites:
            for candidate in data.biomass_routes:
                result.x[model.index['bio_y', site, candidate]] = float(candidate == route)
                result.x[model.index['bio_q', site, candidate]] = (
                    result.x[model.index['B', site]] * data.biomass_kg_per_batch
                    if candidate == route else 0.0
                )
        result.metrics = {name: expr.evaluate(result.x) for name, expr in model.expressions.items()}
        require(abs(result.metrics['cost'] - before) <= 1e-6,
                'Biomass route cost tie-break changed private cost.')
        require(max(solution_validation(model, result).values()) <= 1e-5,
                'Biomass route cost tie-break violated the MILP.')
        result.selection_method += '; lower-GWP biomass route chosen among equal-cost AD optima'
        return result
    if os.environ.get('IS_MILP_BACKEND', 'gams').lower() == 'scipy':
        from scipy.optimize import Bounds, LinearConstraint, milp
        c, A, bl, bu = _assemble_problem(model, primary)
        raw = milp(c, integrality=model.integrality,
                   bounds=Bounds(model.lower, model.upper),
                   constraints=LinearConstraint(A, bl, bu),
                   options={'mip_rel_gap': MIP_REL_GAP})
        if not raw.success or raw.x is None:
            return SolveResult(case_id, False, int(raw.status), str(raw.message),
                               np.full(len(model.var_names), np.nan), {}, np.nan,
                               primary, selection_method,
                               {**dict(metadata or {}), 'solver_backend': 'scipy_highs_milp'})
        x = np.asarray(raw.x, dtype=float)
        metrics = {name: expression.evaluate(x) for name, expression in model.expressions.items()}
        return canonicalise_cost_tie(SolveResult(case_id, True, int(raw.status), f'optimal; {raw.message}',
                           x, metrics, float(raw.fun), primary, selection_method,
                           {**dict(metadata or {}), 'solver_backend': 'scipy_highs_milp',
                            'mip_gap': getattr(raw, 'mip_gap', None)}))
    return canonicalise_cost_tie(_solve_pyomo_gams(
        model=model,
        case_id=case_id,
        primary=primary,
        selection_method=selection_method,
        metadata=metadata,
    ))

def evaluate_bau(model: LinearModel) -> SolveResult:
    """Construct the fixed upstream BAU allocation with diesel transport included."""
    data = model.data
    x = np.zeros(len(model.var_names))
    low_sites = [j for j in data.sites if abs(data.batches[j] - data.batch_low) <= 1e-9]
    require(len(low_sites) == 1, 'BAU must contain exactly one low-batch site.')
    for j in data.sites:
        batches_j = data.batches[j]
        x[model.index['B', j]] = batches_j
        x[model.index['low_batch', j]] = 1.0 if j == low_sites[0] else 0.0
        x[model.index['bio_y', j, 'ad_gas_supply']] = float(data.site_open[j])
        x[model.index['bio_q', j, 'ad_gas_supply']] = batches_j * data.biomass_kg_per_batch
        for p in data.polymers:
            generation_t = batches_j * data.polymer_kg_per_batch[p] / 1000.0
            x[model.index['x', j, p, 'landfill_all', 'diesel_articulated_hgv']] = generation_t
        total_t = batches_j * sum(data.polymer_kg_per_batch.values()) / 1000.0
        x[model.index['n', j, 'landfill_all', 'diesel_articulated_hgv']] = math.ceil(
            total_t / data.payload_t['diesel_articulated_hgv'] - 1e-12
        )
        x[model.index['E', j, 'grid']] = batches_j * data.electricity_kwh_per_batch
        wastewater_j = batches_j * data.wastewater_m3_per_batch
        utility_j = batches_j * data.utility_water_m3_per_batch
        x[model.index['qDIR', j]] = wastewater_j
        x[model.index['qREU', j]] = 0.0
        x[model.index['u', j]] = 0.0
        x[model.index['f', j]] = utility_j
        x[model.index['d', j]] = wastewater_j
    metrics = {name: expr.evaluate(x) for name, expr in model.expressions.items()}
    return SolveResult(
        case_id='BAU', success=True, status=0, message='Constructed benchmark', x=x,
        metrics=metrics, raw_objective=metrics['cost'], primary_objective='benchmark',
        selection_method='fixed upstream batch allocation; all landfill; diesel transport; grid electricity; no reuse',
        metadata={'solver_backend': 'constructed_benchmark', 'low_batch_location_fixed': data.site_location[low_sites[0]]},
    )

def solution_validation(model: LinearModel, result: SolveResult) -> Dict[str, float]:
    ax = model.A.tocsr().dot(result.x)
    lower_violation = np.where(np.isfinite(model.b_l), np.maximum(model.b_l - ax, 0.0), 0.0)
    upper_violation = np.where(np.isfinite(model.b_u), np.maximum(ax - model.b_u, 0.0), 0.0)
    int_mask = model.integrality.astype(bool)
    int_violation = np.max(np.abs(result.x[int_mask] - np.round(result.x[int_mask]))) if int_mask.any() else 0.0
    bound_violation = max(
        float(np.max(np.maximum(model.lower - result.x, 0.0))),
        float(np.max(np.where(np.isfinite(model.upper), np.maximum(result.x - model.upper, 0.0), 0.0))),
    )
    return {
        'max_core_constraint_violation': float(max(np.max(lower_violation), np.max(upper_violation))),
        'max_integrality_violation': float(int_violation),
        'max_bound_violation': float(bound_violation),
    }

def solution_decisions(model: LinearModel, result: SolveResult) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = model.data
    batches_by_site = {j: float(result.x[model.index['B', j]]) for j in data.sites}
    low_flag = {j: float(result.x[model.index['low_batch', j]]) for j in data.sites}
    route_rows = []
    for j in data.sites:
        for p in data.polymers:
            generation = batches_by_site[j] * data.polymer_kg_per_batch[p] / 1000.0
            for r in data.routes:
                for l in data.transport_modes:
                    key = ('x', j, p, r, l)
                    if key in model.index:
                        mass = result.x[model.index[key]]
                        if mass > 1e-7:
                            route_rows.append({
                                'case_id': result.case_id, 'site': j, 'location': data.site_location[j],
                                'annual_batches': batches_by_site[j],
                                'is_low_batch_location': low_flag[j] > 0.5,
                                'polymer': p, 'route': r, 'lca_pathway': data.route_lca_pathway[r],
                                'transport_mode': l, 'tonnes_per_year': mass,
                                'share_of_site_polymer': mass / generation if generation > 0 else np.nan,
                                'recovered_tonnes': mass * data.recovery_yield[p, r],
                                'residual_tonnes': mass * data.residual_fraction[p, r],
                                'autoclave_required': bool(data.autoclave_required[p, r]),
                                'autoclaved_tonnes': mass * data.autoclave_required[p, r],
                                'virgin_polymer_production_gwp_kgco2e_per_year': (
                                    mass * 1000.0 * data.virgin_polymer_gwp_kg_per_kg[p, r]
                                ),
                                'sut_component_manufacturing_gwp_kgco2e_per_year': (
                                    mass * 1000.0 * data.sut_component_manufacturing_gwp_kg_per_kg[p, r]
                                ),
                                'avoided_virgin_material_credit_gwp_kgco2e_per_year': (
                                    mass * 1000.0 * data.avoided_virgin_material_credit_gwp_kg_per_kg[p, r]
                                ),
                                'net_plastic_route_gwp_kgco2e_per_year': (
                                    mass * 1000.0 * data.plastic_gwp_kg_per_kg[p, r]
                                ),
                            })
    routes_df = pd.DataFrame(route_rows)

    site_rows = []
    for j in data.sites:
        batches_j = batches_by_site[j]
        electricity_demand_j = batches_j * data.electricity_kwh_per_batch
        utility_demand_j = batches_j * data.utility_water_m3_per_batch
        fixed_mains_j = batches_j * data.fixed_mains_water_m3_per_batch
        grid = result.x[model.index['E', j, 'grid']]
        wind = result.x[model.index['E', j, 'wind']]
        qreu = result.x[model.index['qREU', j]]
        u = result.x[model.index['u', j]]
        f = result.x[model.index['f', j]]
        d = result.x[model.index['d', j]]
        journeys = {}
        for l in data.transport_modes:
            journeys[l] = sum(result.x[model.index['n', j, r, l]] for r in data.routes if ('n', j, r, l) in model.index)
        site_rows.append({
            'case_id': result.case_id, 'site': j, 'location': data.site_location[j],
            'annual_batches': batches_j,
            'is_low_batch_location': low_flag[j] > 0.5,
            'grid_kwh': grid, 'wind_kwh': wind,
            'wind_share': wind / electricity_demand_j if electricity_demand_j > TOL else np.nan,
            'wastewater_to_reuse_m3': qreu, 'water_reused_m3': u,
            'reuse_share_of_eligible_demand': u / utility_demand_j if utility_demand_j > TOL else np.nan,
            'fresh_utility_water_m3': f,
            'fixed_wfi_mains_water_m3': fixed_mains_j,
            'net_mains_freshwater_withdrawal_m3': fixed_mains_j + f,
            'effluent_discharged_m3': d,
            'diesel_journeys': journeys.get('diesel_articulated_hgv', 0.0),
            'bev_journeys': journeys.get('battery_electric_articulated_hgv', 0.0),
            'biomass_route': next(r for r in data.biomass_routes
                                  if result.x[model.index['bio_y', j, r]] > 0.5),
            'biomass_kg_per_year': batches_j * data.biomass_kg_per_batch,
            'biomass_useful_heat_mj_per_year': sum(
                result.x[model.index['bio_q', j, r]] * data.biomass_useful_heat_mj_per_kg[r]
                for r in data.biomass_routes),
        })
    sites_df = pd.DataFrame(site_rows)

    if routes_df.empty:
        route_mix = pd.DataFrame()
    else:
        route_mix = (routes_df.groupby(['case_id', 'route', 'transport_mode'], as_index=False)
                     .agg(tonnes_per_year=('tonnes_per_year', 'sum'),
                          recovered_tonnes=('recovered_tonnes', 'sum'),
                          residual_tonnes=('residual_tonnes', 'sum')))
        total = route_mix.groupby('case_id')['tonnes_per_year'].transform('sum')
        route_mix['share_of_total_plastic'] = route_mix['tonnes_per_year'] / total
        journey_lookup = {}
        vehicle_km_lookup = {}
        transport_cost_lookup = {}
        transport_gwp_lookup = {}
        for r in data.routes:
            for l in data.transport_modes:
                journeys = 0.0
                vehicle_km = 0.0
                for j in data.sites:
                    key = ('n', j, r, l)
                    if key in model.index:
                        n = result.x[model.index[key]]
                        journeys += n
                        vehicle_km += n * data.roundtrip_multiplier[j, r, l] * data.distance_km[j, r]
                journey_lookup[(r, l)] = journeys
                vehicle_km_lookup[(r, l)] = vehicle_km
                transport_cost_lookup[(r, l)] = vehicle_km * data.transport_cost_gbp_per_km[l]
                transport_gwp_lookup[(r, l)] = vehicle_km * data.transport_gwp_kg_per_km[l]
        keys = list(zip(route_mix['route'], route_mix['transport_mode']))
        route_mix['journeys_per_year'] = [journey_lookup[k] for k in keys]
        route_mix['vehicle_km_per_year'] = [vehicle_km_lookup[k] for k in keys]
        route_mix['transport_cost_gbp_per_year'] = [transport_cost_lookup[k] for k in keys]
        route_mix['transport_gwp_kgco2e_per_year'] = [transport_gwp_lookup[k] for k in keys]
    return routes_df, sites_df, route_mix

def compact_decision_text(model: LinearModel, result: SolveResult) -> str:
    routes, sites, mix = solution_decisions(model, result)
    route_text = '; '.join(
        f"{row.route}: {row.tonnes_per_year:.1f} t ({100*row.share_of_total_plastic:.0f}%, {row.transport_mode.replace('_articulated_hgv','')})"
        for row in mix.sort_values('tonnes_per_year', ascending=False).itertuples(index=False)
    )
    wind_share = sites['wind_kwh'].sum() / (sites['grid_kwh'].sum() + sites['wind_kwh'].sum())
    eligible_demand = (sites['water_reused_m3'] + sites['fresh_utility_water_m3']).sum()
    reuse_share = sites['water_reused_m3'].sum() / eligible_demand if eligible_demand > TOL else np.nan
    low_row = sites.loc[sites['is_low_batch_location']].iloc[0]
    return (
        f"Low-batch node: {low_row.location} ({low_row.annual_batches:.0f} batches); "
        f"electricity: {100*wind_share:.0f}% wind; eligible water reused: {100*reuse_share:.0f}%; "
        f"biomass: {', '.join(sorted(sites.biomass_route.unique()))}; plastics: {route_text}"
    )

CASE_NAMES = {
    'BAU': 'Base case',
    'OPT_COST_PLUS_CARBON_VALUE': 'Optimal - cost plus carbon value',
}


def case_name(case_id: str) -> str:
    return CASE_NAMES.get(case_id, case_id.replace('_', ' ').title())

def component_breakdown(model: LinearModel, result: SolveResult) -> pd.DataFrame:
    """Exact objective decomposition with electricity separated from industrial symbiosis."""
    data = model.data
    x = result.x
    batches = {j: x[model.index['B', j]] for j in data.sites}

    electricity_cost = sum(
        x[model.index['E', j, e]] * data.electricity_cost_gbp_per_kwh[e]
        for j in data.sites for e in data.electricity_sources
    )
    electricity_gwp = sum(
        x[model.index['E', j, e]] * data.electricity_gwp_kg_per_kwh[e]
        for j in data.sites for e in data.electricity_sources
    )

    plastics_cost = 0.0
    plastics_gwp = 0.0
    virgin_polymer_gwp = 0.0
    sut_component_manufacturing_gwp = 0.0
    avoided_virgin_material_credit_gwp = 0.0
    transport_cost = 0.0
    transport_gwp = 0.0
    for key, idx in model.index.items():
        if key[0] == 'x':
            _, j, p, r, l = key
            mass = x[idx]
            plastics_cost += mass * (
                data.route_cost_gbp_per_t[p, r]
                + data.autoclave_required[p, r] * data.autoclave_cost_gbp_per_t[j]
                - data.value_capture[p, r] * data.recovered_value_gbp_per_t[p, r] * data.recovery_yield[p, r]
            )
            plastics_gwp += mass * 1000.0 * data.plastic_gwp_kg_per_kg[p, r]
            virgin_polymer_gwp += mass * 1000.0 * data.virgin_polymer_gwp_kg_per_kg[p, r]
            sut_component_manufacturing_gwp += (
                mass * 1000.0 * data.sut_component_manufacturing_gwp_kg_per_kg[p, r]
            )
            avoided_virgin_material_credit_gwp += (
                mass * 1000.0 * data.avoided_virgin_material_credit_gwp_kg_per_kg[p, r]
            )
        elif key[0] == 'n':
            _, j, r, l = key
            journeys = x[idx]
            route_km = data.roundtrip_multiplier[j, r, l] * data.distance_km[j, r]
            transport_cost += journeys * route_km * data.transport_cost_gbp_per_km[l]
            transport_gwp += journeys * route_km * data.transport_gwp_kg_per_km[l]

    water_capex = sum(data.treatment_fixed_cost_gbp[j] * data.site_open[j] for j in data.sites)
    water_opex = 0.0
    water_gwp = 0.0
    freshwater = 0.0
    for j in data.sites:
        b = batches[j]
        f = x[model.index['f', j]]
        d = x[model.index['d', j]]
        wastewater = b * data.wastewater_m3_per_batch
        fixed_mains = b * data.fixed_mains_water_m3_per_batch
        water_opex += (
            data.treatment_opex_gbp_per_m3[j] * wastewater
            + data.fresh_water_cost_gbp_per_m3[j] * f
            + data.discharge_cost_gbp_per_m3[j] * d
        )
        water_gwp += (
            data.treatment_gwp_kg_per_m3[j] * wastewater
            + data.fresh_water_gwp_kg_per_m3[j] * (fixed_mains + f)
        )
        freshwater += fixed_mains + f

    rows = [
        ('Manufacturing electricity - reported separately', 'Electricity', electricity_cost, 0.0, electricity_gwp, 0.0),
        ('SUT plastics life cycle', 'Virgin polymer production', 0.0, 0.0, virgin_polymer_gwp, 0.0),
        ('SUT plastics life cycle', 'Pharmaceutical-grade SUT component manufacturing', 0.0, 0.0, sut_component_manufacturing_gwp, 0.0),
        ('SUT plastics life cycle', 'Avoided virgin polymer production credit', 0.0, 0.0, avoided_virgin_material_credit_gwp, 0.0),
        (
            'Industrial symbiosis routing',
            'Plastic end-of-life, autoclave and energy-credit balance',
            plastics_cost,
            0.0,
            plastics_gwp
            - virgin_polymer_gwp
            - sut_component_manufacturing_gwp
            - avoided_virgin_material_credit_gwp,
            0.0,
        ),
        ('Industrial symbiosis routing', 'Water', water_capex + water_opex, water_capex, water_gwp, freshwater),
        ('Industrial symbiosis routing', 'Biomass',
         data.biomass_cost_gbp + sum(x[model.index['bio_q', j, r]] * data.biomass_route_cost_gbp_per_kg[r]
                                     for j in data.sites for r in data.biomass_routes),
         0.0,
         sum(x[model.index['bio_q', j, r]] * data.biomass_route_factors[r]['gwp']
             for j in data.sites for r in data.biomass_routes), 0.0),
        ('Industrial symbiosis routing', 'Transport', transport_cost, 0.0, transport_gwp, 0.0),
        ('Fixed manufacturing background', 'Fixed non-electric manufacturing', 0.0, 0.0, data.fixed_manufacturing_gwp_kg, 0.0),
    ]
    out = pd.DataFrame(rows, columns=[
        'reporting_block', 'category', 'annual_cost_gbp', 'annualised_capex_gbp',
        'annual_gwp_kgco2e', 'net_freshwater_withdrawal_m3'
    ])
    out.insert(0, 'case_name', case_name(result.case_id))
    out.insert(0, 'case_id', result.case_id)
    out['annual_opex_and_net_revenue_gbp'] = out['annual_cost_gbp'] - out['annualised_capex_gbp']

    require(abs(out['annual_cost_gbp'].sum() - result.metrics['cost']) <= 1e-4,
            f'Cost decomposition does not reconcile for {result.case_id}.')
    require(abs(out['annual_gwp_kgco2e'].sum() - result.metrics['gwp']) <= 1e-4,
            f'GWP decomposition does not reconcile for {result.case_id}.')
    require(abs(out['net_freshwater_withdrawal_m3'].sum() - result.metrics['freshwater']) <= 1e-6,
            f'Freshwater decomposition does not reconcile for {result.case_id}.')
    return out

ROUTE_LABELS = {
    'landfill_all': 'Landfill',
    'incineration_efw_heat_electricity': 'Energy from Waste (EfW)',
    'mechanical_recycling_polyolefins': 'Polyolefin recycling',
    'mechanical_recycling_tpe_tubing': 'TPE recycling',
}
TRANSPORT_LABELS = {
    'diesel_articulated_hgv': 'Diesel HGV',
    'battery_electric_articulated_hgv': 'Electric HGV',
}

def route_choice_text(mix: pd.DataFrame) -> str:
    if mix.empty:
        return ''
    grouped = mix.groupby('route', as_index=False).agg(
        tonnes_per_year=('tonnes_per_year', 'sum'),
        share_of_total_plastic=('share_of_total_plastic', 'sum'),
    ).sort_values('tonnes_per_year', ascending=False)
    return '; '.join(
        f"{ROUTE_LABELS.get(r.route, r.route)}: {r.tonnes_per_year:.1f} t/y "
        f"({100*r.share_of_total_plastic:.1f}%)"
        for r in grouped.itertuples(index=False)
    )

def transport_choice_text(mix: pd.DataFrame) -> str:
    if mix.empty:
        return ''
    mode_rows = mix.groupby('transport_mode', as_index=False).agg(
        journeys_per_year=('journeys_per_year', 'sum'),
        vehicle_km_per_year=('vehicle_km_per_year', 'sum'),
    )
    return '; '.join(
        f"{TRANSPORT_LABELS.get(r.transport_mode, r.transport_mode)}: "
        f"{r.journeys_per_year:.0f} journeys/y, {r.vehicle_km_per_year:,.0f} km/y"
        for r in mode_rows.itertuples(index=False)
    )

def prepare_run_outputs() -> None:
    """Keep prior result tables and archive solver diagnostics for a new run."""
    global _GAMS_SOLVE_COUNTER, _GAMS_PREFLIGHT_DONE
    _GAMS_SOLVE_COUNTER = 0
    _GAMS_PREFLIGHT_DONE = False
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    previous = [path for path in (GAMS_LOG_DIR, GAMS_WORK_DIR, GAMS_SOLVE_REPORT) if path.exists()]
    if previous:
        archive_dir = RESULT_DIR / 'previous_solver_runs' / str(time.time_ns())
        archive_dir.mkdir(parents=True, exist_ok=False)
        for path in previous:
            path.rename(archive_dir / path.name)
    GAMS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if GAMS_KEEPFILES:
        GAMS_WORK_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_human_health_daly(model: LinearModel, result: SolveResult) -> float:
    """Evaluate reporting-only ReCiPe human-health damage for a solved case.

    DALY is intentionally absent from ``build_linear_model``. Transport is not
    included because the supplied transport parameters expose GWP, not a full LCI.
    """
    require(result.success, f'Cannot calculate DALY for unsuccessful case {result.case_id}.')
    data = model.data
    x = result.x
    total = data.total_batches * data.fixed_non_electric_human_health_daly_per_batch
    total += sum(x[model.index['bio_q', j, r]] * data.biomass_route_factors[r]['human_health_daly']
                 for j in data.sites for r in data.biomass_routes)
    for j in data.sites:
        for source in data.electricity_sources:
            total += x[model.index['E', j, source]] * data.electricity_human_health_daly_per_kwh[source]
    for key, idx in model.index.items():
        if key[0] == 'x':
            _, _site, polymer, route, _mode = key
            total += x[idx] * 1000.0 * data.plastic_human_health_daly_per_kg[polymer, route]
    for j in data.sites:
        batches_j = x[model.index['B', j]]
        fixed_mains_j = batches_j * data.fixed_mains_water_m3_per_batch
        fresh_j = x[model.index['f', j]]
        total += (
            batches_j * data.wastewater_m3_per_batch
            * data.treatment_human_health_daly_per_m3[j]
        )
        total += (
            (fixed_mains_j + fresh_j)
            * data.fresh_water_human_health_daly_per_m3[j]
        )
    return float(total)


def evaluate_ecosystem_quality(model: LinearModel, result: SolveResult) -> float:
    """Evaluate species·year from solved quantities; never enters the MILP."""
    require(result.success, f'Cannot calculate ecosystem damage for {result.case_id}.')
    data, x = model.data, result.x
    total = data.total_batches * data.fixed_non_electric_ecosystem_quality_per_batch
    total += sum(
        x[model.index['bio_q', j, route]] * data.biomass_route_factors[route]['ecosystem_quality']
        for j in data.sites for route in data.biomass_routes
    )
    for j in data.sites:
        for source in data.electricity_sources:
            total += x[model.index['E', j, source]] * data.electricity_ecosystem_quality_per_kwh[source]
    for key, idx in model.index.items():
        if key[0] == 'x':
            _, _site, polymer, route, _mode = key
            total += x[idx] * 1000.0 * data.plastic_ecosystem_quality_per_kg[polymer, route]
    for j in data.sites:
        batches_j = x[model.index['B', j]]
        fixed_mains_j = batches_j * data.fixed_mains_water_m3_per_batch
        fresh_j = x[model.index['f', j]]
        total += batches_j * data.wastewater_m3_per_batch * data.treatment_ecosystem_quality_per_m3[j]
        total += (fixed_mains_j + fresh_j) * data.fresh_water_ecosystem_quality_per_m3[j]
    return float(total)


def ecosystem_quality_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[[
        'case_id', 'case_name', 'ecosystem_quality',
        'avoided_ecosystem_quality_vs_bau', 'ecosystem_quality_reduction_vs_bau_pct',
    ]].rename(columns={
        'case_name': 'Solution',
        'ecosystem_quality': 'species.yr/year',
        'avoided_ecosystem_quality_vs_bau': 'Avoided species.yr vs BAU',
        'ecosystem_quality_reduction_vs_bau_pct': 'Reduction vs BAU [%]',
    })


def human_health_daly_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the standalone BAU-referenced human-health result table."""
    return summary[[
        'case_id', 'case_name', 'human_health_daly',
        'avoided_human_health_daly_vs_bau',
        'human_health_daly_reduction_vs_bau_pct',
    ]].rename(columns={
        'case_name': 'Solution',
        'human_health_daly': 'DALY/yr',
        'avoided_human_health_daly_vs_bau': 'Avoided DALY vs BAU',
        'human_health_daly_reduction_vs_bau_pct': 'Reduction vs BAU [%]',
    })


def _case_summary_row(model: LinearModel, result: SolveResult, bau: SolveResult) -> Dict[str, Any]:
    routes, sites, mix = solution_decisions(model, result)
    low = sites.loc[sites['is_low_batch_location']].iloc[0]
    row: Dict[str, Any] = {
        'case_id': result.case_id,
        'case_name': case_name(result.case_id),
        'case_type': 'benchmark' if result.case_id == 'BAU' else 'optimised',
        'primary_objective': result.primary_objective,
        'selection_method': result.selection_method,
        'solver_success': result.success,
        'solver_status': result.status,
        'solver_message': result.message,
        'solver_backend': result.metadata.get(
            'solver_backend', 'constructed_benchmark' if result.case_id == 'BAU' else ''
        ),
        'low_batch_location': low['location'],
        'low_batch_count': float(low['annual_batches']),
        'decision_snapshot': compact_decision_text(model, result),
        **solution_validation(model, result),
    }
    for objective in OBJECTIVES + ['policy_cost', 'recovered']:
        row[objective] = float(result.metrics[objective])
    row['annual_network_cost_gbp'] = row['cost']
    row['gwp_tco2e_per_year'] = row['gwp'] / 1000.0
    row['cost_gbp_per_kg_mab'] = row['cost'] / model.data.total_product_kg
    row['gwp_kgco2e_per_kg_mab'] = row['gwp'] / model.data.total_product_kg
    human_health_daly = evaluate_human_health_daly(model, result)
    bau_human_health_daly = evaluate_human_health_daly(model, bau)
    row['human_health_daly'] = human_health_daly
    row['avoided_human_health_daly_vs_bau'] = bau_human_health_daly - human_health_daly
    row['human_health_daly_reduction_vs_bau_pct'] = (
        100.0 * (bau_human_health_daly - human_health_daly) / abs(bau_human_health_daly)
        if abs(bau_human_health_daly) > TOL else np.nan
    )
    ecosystem_quality = evaluate_ecosystem_quality(model, result)
    bau_ecosystem_quality = evaluate_ecosystem_quality(model, bau)
    row['ecosystem_quality'] = ecosystem_quality
    row['avoided_ecosystem_quality_vs_bau'] = bau_ecosystem_quality - ecosystem_quality
    row['ecosystem_quality_reduction_vs_bau_pct'] = (
        100.0 * (bau_ecosystem_quality - ecosystem_quality) / abs(bau_ecosystem_quality)
        if abs(bau_ecosystem_quality) > TOL else np.nan
    )
    row['virgin_polymer_production_gwp_kgco2e_per_year'] = float(
        routes['virgin_polymer_production_gwp_kgco2e_per_year'].sum()
    )
    row['sut_component_manufacturing_gwp_kgco2e_per_year'] = float(
        routes['sut_component_manufacturing_gwp_kgco2e_per_year'].sum()
    )
    row['avoided_virgin_material_credit_gwp_kgco2e_per_year'] = float(
        routes['avoided_virgin_material_credit_gwp_kgco2e_per_year'].sum()
    )
    for objective in OBJECTIVES + ['policy_cost']:
        base = float(bau.metrics[objective])
        delta = float(result.metrics[objective]) - base
        row[f'delta_vs_bau__{objective}'] = delta
        row[f'pct_vs_bau__{objective}'] = 100.0 * delta / abs(base) if abs(base) > TOL else np.nan
    return row


def write_policy_results_workbook(
    model: LinearModel,
    summary: pd.DataFrame,
    routes_detail: pd.DataFrame,
    sites_detail: pd.DataFrame,
    route_mix: pd.DataFrame,
    breakdown: pd.DataFrame,
) -> Path:
    """Write numeric and categorical results as plain workbook tables."""
    solver_status = (
        pd.read_csv(GAMS_SOLVE_REPORT)
        if GAMS_SOLVE_REPORT.exists()
        else pd.DataFrame(columns=['case_id', 'solver_status', 'termination_condition'])
    )
    RESULTS_WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
    sheets = [
        ('Case summary', summary),
        ('Polymer routing', routes_detail),
        ('Site decisions', sites_detail),
        ('Route mix', route_mix),
        ('Objective breakdown', breakdown),
        ('Solver status', solver_status),
        ('Input audit', model.data.audit),
        ('Human health DALY', human_health_daly_comparison_table(summary)),
        ('Ecosystem quality', ecosystem_quality_comparison_table(summary)),
    ]
    with pd.ExcelWriter(RESULTS_WORKBOOK, engine='openpyxl') as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)
    return RESULTS_WORKBOOK


def run_analysis() -> Dict[str, Any]:
    print(f'Project root: {PROJECT_ROOT}')
    print(f'GAMS executable: {shutil.which("gams")}')
    backend = os.environ.get('IS_MILP_BACKEND', 'gams').lower()
    print(f'Configured MILP solver: {"SciPy HiGHS" if backend == "scipy" else GAMS_SOLVER_NAME + " via " + GAMS_SOLVER_IO}')
    print('Analysis scope: BAU benchmark + one cost-plus-carbon-value optimisation only.')

    data = load_model_data()
    model = build_linear_model(data)
    prepare_run_outputs()

    bau = evaluate_bau(model)
    policy = solve_model(
        model,
        'OPT_COST_PLUS_CARBON_VALUE',
        'policy_cost',
        selection_method='minimum annual network cost plus central lifecycle carbon value',
    )
    require(policy.success, f'Cost-plus-carbon-value solve failed: {policy.message}')

    selected = [bau, policy]
    summary = pd.DataFrame([_case_summary_row(model, result, bau) for result in selected])

    all_routes: List[pd.DataFrame] = []
    all_sites: List[pd.DataFrame] = []
    all_mix: List[pd.DataFrame] = []
    all_breakdown: List[pd.DataFrame] = []
    for result in selected:
        routes, sites, mix = solution_decisions(model, result)
        routes['case_name'] = case_name(result.case_id)
        sites['case_name'] = case_name(result.case_id)
        mix['case_name'] = case_name(result.case_id)
        all_routes.append(routes)
        all_sites.append(sites)
        all_mix.append(mix)
        all_breakdown.append(component_breakdown(model, result))

    routes_detail = pd.concat(all_routes, ignore_index=True)
    sites_detail = pd.concat(all_sites, ignore_index=True)
    route_mix = pd.concat(all_mix, ignore_index=True)
    breakdown = pd.concat(all_breakdown, ignore_index=True)

    invalid_efw_autoclave = routes_detail.loc[
        routes_detail['route'].eq('incineration_efw_heat_electricity')
        & routes_detail['autoclave_required'].astype(bool)
    ]
    require(invalid_efw_autoclave.empty, 'EfW was incorrectly assigned autoclaving.')

    required_autoclave_routes = routes_detail['route'].eq('landfill_all') | routes_detail['route'].str.startswith('mechanical_recycling')
    missing_required_autoclave = routes_detail.loc[
        required_autoclave_routes & ~routes_detail['autoclave_required'].astype(bool)
    ]
    require(
        missing_required_autoclave.empty,
        'A selected landfill or mechanical-recycling route is missing required autoclaving.',
    )

    # Exact downstream contract used by routing_comparison.py.
    summary.to_csv(RESULT_DIR / 'case_summary.csv', index=False)
    routes_detail.to_csv(RESULT_DIR / 'selected_case_polymer_routing.csv', index=False)
    sites_detail.to_csv(RESULT_DIR / 'selected_case_site_decisions.csv', index=False)
    route_mix.to_csv(RESULT_DIR / 'selected_case_route_mix.csv', index=False)
    breakdown.to_csv(RESULT_DIR / 'selected_case_objective_breakdown.csv', index=False)
    human_health_daly_comparison_table(summary).to_csv(
        RESULT_DIR / 'human_health_daly_comparison.csv', index=False
    )
    ecosystem_quality_comparison_table(summary).to_csv(
        RESULT_DIR / 'ecosystem_quality_comparison.csv', index=False
    )

    workbook = write_policy_results_workbook(
        model, summary, routes_detail, sites_detail, route_mix, breakdown
    )

    display_columns = [
        'case_id', 'low_batch_location', 'cost', 'policy_cost', 'gwp',
        'human_health_daly', 'ecosystem_quality', 'freshwater', 'circularity_loss', 'residual', 'decision_snapshot',
    ]
    print('\nCORE RESULT')
    print(summary[display_columns].to_string(index=False, float_format=lambda value: f'{value:,.4f}'))
    print('\nSELECTED ROUTING')
    print(route_mix.to_string(index=False, float_format=lambda value: f'{value:,.4f}'))
    print(f'\nHeadline and drill-down workbook: {workbook}')
    if GAMS_SOLVE_REPORT.exists():
        print(f'GAMS solve reports: {GAMS_SOLVE_REPORT}')
    print(f'Results written to: {RESULT_DIR}')

    return {
        'data': data,
        'model': model,
        'bau': bau,
        'policy': policy,
        'summary': summary,
        'routes_detail': routes_detail,
        'sites_detail': sites_detail,
        'route_mix': route_mix,
        'objective_breakdown': breakdown,
        'results_workbook': workbook,
    }


if __name__ == '__main__':
    run_analysis()
