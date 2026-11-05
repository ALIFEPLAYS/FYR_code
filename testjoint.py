"""One annual supply-chain/industrial-symbiosis cost MILP, exported to GAMS.

Run: python MILP/code/testjoint.py
Export without solving: python MILP/code/testjoint.py --export-only
The supply-chain formulation below is retained; the IS block shares its
site-opening and batch decisions. Monetary terms use nominal GBP 2026.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyomo.environ as pyo

# File locations, solver choice and common price basis. Paths are relative to
# this script, so running it from another working directory is supported.
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
PARAMETER_DIR = REPO_ROOT / 'parameters'
RESULT_DIR = PROJECT_ROOT / 'results' / 'testjoint' / 'joint_cost'
GAMS_SOLVER = os.environ.get('IS_GAMS_SOLVER', 'cplex').strip().lower()
MIP_REL_GAP = 1e-9
INPUT_FILES = {
    'eligibility': PROJECT_ROOT / 'inputs/polymer_route_eligibility_20260722.xlsx',
    'lca': Path(os.environ.get('IS_LCA_ROUTE_FACTOR_FILE', str(
        REPO_ROOT / 'LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx'))),
    'params': PARAMETER_DIR / 'MILP_central_params.json',
    'itemised_costs': REPO_ROOT / 'SuperPro_Reports_Data_Extraction/data/processed/itemised_costs_20261026.csv',
}


def require(condition, message):
    """Stop with a clear input or validation error if a required condition is false."""
    if not condition:
        raise ValueError(message)


def single_row(frame, mask, label):
    """Select exactly one matching input row; reject missing or ambiguous records."""
    rows = frame.loc[mask]
    require(len(rows) == 1, f'Expected one row for {label}; found {len(rows)}.')
    return rows.iloc[0]


def load_joint_inputs():
    """Read physical quantities, eligibility and prices, without impact factors.

    There is no upstream solution CSV dependency and no imported site/batch
    pattern. The route workbook supplies physical per-batch inventories only.
    """
    # Input validation and site mapping: read source tables and match site IDs
    # to Oxford, Cambridge and Slough. Site openings are decided by the model.
    for label, path in INPUT_FILES.items():
        require(path.exists(), f'Missing {label} input: {path}')
    params = json.loads(INPUT_FILES['params'].read_text(encoding='utf-8-sig'))
    eligibility_raw = pd.read_excel(INPUT_FILES['eligibility'], sheet_name='polymer_route_eligibility')
    mapping = pd.read_excel(INPUT_FILES['eligibility'], sheet_name='site_mapping')
    factors = pd.read_excel(INPUT_FILES['lca'], sheet_name='route_factor_matrix')
    io = pd.read_excel(INPUT_FILES['lca'], sheet_name='input_output_table')
    require(mapping['manufacturing_site'].is_unique, 'Duplicate manufacturing site mapping.')
    site_location = dict(zip(mapping['manufacturing_site'].astype(str), mapping['site_location'].astype(str)))
    sites = ['j1', 'j2', 'j3']
    require(set(site_location) == set(sites), 'Site mapping must cover j1, j2 and j3 exactly.')

    def physical_row(flow, pathway):
        """Find the physical inventory row for one flow and treatment or supply pathway."""
        return single_row(factors,
            factors['optimisation_flow'].astype(str).eq(flow)
            & factors['pathway'].astype(str).eq(pathway), f'{flow}/{pathway}')

    # Per-batch utility inventories: production determines electricity demand,
    # wastewater generation, eligible utility-water demand and biomass quantity.
    electricity = float(physical_row('manufacturing__electricity', 'electricity_grid_uk_bau')['represented_quantity_per_batch'])
    wind_quantity = float(physical_row('manufacturing__electricity', 'electricity_renewable_onshore_wind')['represented_quantity_per_batch'])
    require(math.isclose(electricity, wind_quantity), 'Electricity inventories disagree between sources.')
    wastewater = float(physical_row('plant_wastewater', 'treatment_and_discharge')['represented_quantity_per_batch'])
    utility_water = float(physical_row('fresh_utility_water', 'fresh_utility_water_supply')['represented_quantity_per_batch'])
    biomass = float(physical_row('residual_biomass', 'ad_biomethane_digestate')['represented_quantity_per_batch'])
    recover = single_row(io,
        io['source_module'].astype(str).eq('wastewater_lca')
        & io['io_table_role'].astype(str).eq('recoverable_internal_utility_water')
        & io['pathway'].astype(str).eq('treatment_for_utility_reuse'), 'recoverable utility water')
    require(all(math.isfinite(q) and q > 0 for q in [electricity, wastewater, utility_water, biomass]),
            'Per-batch quantities must be positive and finite.')
    eta = float(recover['optimisation_quantity_per_batch']) / wastewater
    require(0 <= eta <= 1 + 1e-9, 'Water recovery fraction must be within [0,1].')
    eta = min(eta, 1.0)

    # Plastic inventory and routing: collect kg generated per batch, then expand
    # recycling into the separate polyolefin and TPE destination facilities.
    plastics = factors.loc[factors['optimisation_flow'].astype(str).str.startswith('sut_plastic__')]
    require(not plastics.empty, 'No physical polymer inventory found.')
    grouped = plastics.groupby('material_or_service')['represented_quantity_per_batch']
    require(grouped.nunique().eq(1).all(), 'Per-batch plastic inventory varies by route.')
    polymer_kg = grouped.first().astype(float).to_dict()
    require(all(math.isfinite(q) and q >= 0 for q in polymer_kg.values()) and sum(polymer_kg.values()) > 0,
            'Invalid per-batch plastic quantities.')
    pathways = {
        'landfill_all': 'landfill_all',
        'incineration_efw_heat_electricity': 'incineration_efw_heat_electricity',
        'mechanical_recycling_polyolefins': 'mechanical_recycling_main_residual_incineration',
        'mechanical_recycling_tpe_tubing': 'mechanical_recycling_main_residual_incineration',
    }
    base = eligibility_raw.loc[eligibility_raw['eligible'].fillna(False).astype(bool)
                               & eligibility_raw['base_enabled'].fillna(False).astype(bool)]
    eligible, autoclave, yields, residual = [], {}, {}, {}
    net_route_cost = {}
    conversion = params['currency_and_price_year_conversions']['multipliers_to_nominal_GBP_2026']
    usd_to_gbp = float(conversion['USD_2019_nominal'])
    m2025 = float(conversion['GBP_2025_nominal'])
    grid_cost = float(params['industrial_grid_electricity_cost']['total_cost_gbp_per_kwh'])
    ac_cost = float(params['autoclave_pre_treatment']['electricity_kwh_per_kg_input']['central']) * 1000 * grid_cost
    # Eligible-route economics: add treatment and required autoclaving, then
    # subtract saleable recovered-material revenue using the physical yield.
    for row in base.itertuples(index=False):
        p, pathway = str(row.polymer), str(row.lca_pathway)
        if pathway not in set(pathways.values()):
            continue
        require(p in polymer_kg, f'Eligible polymer {p} missing from physical inventory.')
        route = ('mechanical_recycling_tpe_tubing' if p == 'TPE' else 'mechanical_recycling_polyolefins') if pathway == 'mechanical_recycling_main_residual_incineration' else pathway
        require((p, route) not in eligible, f'Duplicate eligibility for {p}/{route}.')
        eligible.append((p, route))
        ac = int(bool(row.requires_autoclave))
        require(ac == int(route != 'incineration_efw_heat_electricity'), f'Unexpected pretreatment for {p}/{route}.')
        autoclave[p, route] = ac
        record = single_row(plastics,
            plastics['material_or_service'].astype(str).eq(p)
            & plastics['pathway'].astype(str).eq(pathway), f'physical recycling yield {p}/{route}')
        recovered = str(row.output_role) == 'recovered'
        recovery = float(record['recycling_recovery_fraction']) if recovered else 0.0
        require(0 <= recovery <= 1, f'Invalid recovery yield for {p}/{route}.')
        yields[p, route] = recovery
        residual[p, route] = 1.0 - recovery
        if route == 'landfill_all':
            gate = 26.0 * m2025 + 130.75
        elif route == 'incineration_efw_heat_electricity':
            gate = 121.0 * m2025
        else:
            entry = params['recycling_costs']['plastic_reprocessing_costs'][p]
            require(entry['price_basis'] == 'nominal_GBP_2026'
                    and entry['requires_additional_currency_or_price_year_conversion'] is False,
                    'Recycling prices must already be in nominal GBP 2026.')
            gate = float(entry['value'])
        value_key = {'LDPE': 'recycled_ldpe_output_value', 'LLDPE': 'recycled_ldpe_output_value',
                     'PP': 'recycled_pp_output_value', 'TPE': 'recycled_tpe_output_value'}.get(p)
        output_value = float(params[value_key][p]) if recovered and value_key else 0.0
        net_route_cost[p, route] = gate + ac * ac_cost - recovery * output_value
    require(all(any(pair[0] == p for pair in eligible) for p in polymer_kg),
            'Every polymer must have at least one eligible route.')

    # Location-dependent costs: transport distances and local water tariffs
    # affect the relative cost of manufacturing at each site.
    modes = list(params['road_transport_modes'])
    payload = {l: float(params['road_transport_modes'][l]['maximum_payload_tonnes']) for l in modes}
    transport_cost = {l: float(params['road_transport_modes'][l]['cost_gbp_per_vehicle_route_km']) for l in modes}
    require(all(value > 0 for value in payload.values()), 'Vehicle payloads must be positive.')
    distance = {}
    for j in sites:
        entry = params['waste_route_distances'][site_location[j]]
        distance[j, 'landfill_all'] = float(entry['landfill_all']['distance_km'])
        distance[j, 'incineration_efw_heat_electricity'] = float(entry['incineration_efw_heat_electricity']['distance_km'])
        recycling = entry['mechanical_recycling_main_residual_incineration']
        distance[j, 'mechanical_recycling_polyolefins'] = float(recycling['polyolefins']['distance_km'])
        distance[j, 'mechanical_recycling_tpe_tubing'] = float(recycling['tpe_tubing']['distance_km'])
    renewable = params['industrial_renewable_electricity_cost']
    electricity_cost = {'grid': grid_cost, 'wind': float(renewable.get('central_value', renewable.get('value')))}
    wwt = params['onsite_wastewater_treatment_facility']

    # Replace the embedded manufacturing electricity bill with the optimised
    # electricity purchases below. Retain other manufacturing utility services.
    itemised = pd.read_csv(INPUT_FILES['itemised_costs'])
    power = itemised.loc[itemised['cost_group'].astype(str).eq('utility')
                         & itemised['item_name'].astype(str).eq('Std Power')]
    require(not power.empty, 'Missing embedded manufacturing electricity costs.')
    embedded_power = float(pd.to_numeric(power['cost_per_batch_usd'], errors='raise').sum())
    power_kwh = float(pd.to_numeric(power['amount_per_batch'], errors='raise').sum())
    require(math.isclose(power_kwh, electricity, rel_tol=1e-6),
            'Itemised manufacturing electricity quantity differs from IS demand; reconcile inputs before combining costs.')

    return SimpleNamespace(
        sites=sites, site_location=site_location, polymers=list(polymer_kg),
        routes=list(pathways), eligible=eligible, modes=modes,
        polymer_kg=polymer_kg, recovery_yield=yields, residual_fraction=residual,
        route_cost=net_route_cost, payload=payload, transport_cost=transport_cost,
        distance=distance, electricity=electricity, electricity_cost=electricity_cost,
        wastewater=wastewater, utility_water=utility_water, recovery=eta,
        biomass=biomass, biomass_routes=['ad_gas_supply', 'ad_external_steam'],
        biomass_collection=float(params['external_biomass_ad_collection']['annual_collection_cost_gbp_per_active_site_year']),
        treatment_fixed=float(wwt['annualised_capex_gbp_per_site_year']),
        treatment_variable=float(wwt['treatment_opex_gbp_per_m3']),
        treatment_capacity=float(wwt['design_capacity_m3_per_day']) * 365.0,
        fresh_cost={j: float(params['avoided_mains_utility_water_cost'][site_location[j]]) for j in sites},
        discharge_cost={j: float(params['avoided_treated_effluent_discharge_cost'][site_location[j]]) for j in sites},
        usd_to_gbp=usd_to_gbp, embedded_electricity_usd=embedded_power,
    )


def create_mab_supply_chain_model(scenario_set, scenario_probability, r_u, alpha_u, alpha_d, alpha_g, x_g, C_LABju,
                                  C_LABjd, C_LABjg, C_CAPju, C_CAPjd, C_CAPjg, C_FACju, C_FACjd, C_FACjg, C_VARju,
                                  C_VARjd, C_VARjg, sales_activation, total_demand_mab):
    # Network dimensions: one year, three candidate factories, one scale,
    # one product, four possible lines per stage, one hub and five demand zones.
    ## Universal sets

    """Build the supply-chain variables, cost definitions and constraints in 2019 USD; the joint builder later replaces its objective."""
    T = [1]  # Time periods
    J = ['j1', 'j2', 'j3']  # Manufacturing locations
    A = ['L']  # Scales; first mAb version retains only the 1000 L case
    I = ['mAb']  # Single mAb product
    U = [1, 2, 3, 4]  # Upstream lines
    D = [1, 2, 3, 4]  # Downstream lines
    G = [1, 2, 3, 4]  # Fill and finish lines
    K = ['k1']  # Storage facilities
    L = ['l1', 'l2', 'l3', 'l4', 'l5']  # Demand zones

    # Scenario set and probabilities are passed explicitly.
    S = list(scenario_set)
    phi = dict(scenario_probability)

    #### Substitute scenario-independent demand

    demand = total_demand_mab

    D_l = {}
    D_l[(1, "l1", "mAb")] = demand[(1)] / 5  # cardinality of 5 demand zones
    D_l[(1, "l2", "mAb")] = demand[(1)] / 5
    D_l[(1, "l3", "mAb")] = demand[(1)] / 5
    D_l[(1, "l4", "mAb")] = demand[(1)] / 5
    D_l[(1, 'l5', 'mAb')] = demand[(1)] / 5

    ### Distance parameters (km)
    # Generic case-study distances retained for the first mAb case.
    D_jk = {("j1", "k1"): 538, ("j2", "k1"): 799, ("j3", "k1"): 984}
    D_kl = {("k1", "l1"): 1640, ("k1", "l2"): 1878, ("k1", "l3"): 632, ("k1", "l4"): 905, ("k1", "l5"): 1140}

    product_unit = {"mAb": 1.0}  # kg mAb per demand unit
    T_MINu = 0  # days
    T_MAX = 329  # annual operating horizon from SuperPro
    Q_min = 0  # batches
    Q_max = 1e6
    r_d = 1  # days
    r_g = 1  # days
    H_ja = 329  # maximum annual utilisation days per line; aligned with the SuperPro throughput basis above.
    t_c = 0.05  # USD/kg-km; frozen transport assumption
    t_s = 0.4  # USD/kg handled; frozen storage assumption
    x_d = x_g  # kg final mAb-equivalent output per complete batch
    x_u = x_d  # kg final mAb-equivalent output per complete batch

    model = pyo.ConcreteModel()

    # ---------- BINARY VARIABLES --------------------------------------------------------------------------------------

    model.Y_ja = pyo.Var(T, J, A, within=pyo.Binary)  # scale selection
    model.Y_j = pyo.Var(T, J, within=pyo.Binary)  # manufacturing node activation
    model.Y_k = pyo.Var(T, K, within=pyo.Binary)  # storage node activation

    model.X_jk = pyo.Var(T, J, K, within=pyo.Binary)  # transport link J to K
    model.X_kl = pyo.Var(T, K, L, within=pyo.Binary)  # transport link K to L

    model.Z_u = pyo.Var(T, J, A, U, within=pyo.Binary)  # USP line activation
    model.Z_d = pyo.Var(T, J, A, D, within=pyo.Binary)  # DSP line activation
    model.Z_g = pyo.Var(T, J, A, G, within=pyo.Binary)  # FF line activation

    model.M_u = pyo.Var(T, J, A, U, I, within=pyo.Binary)  # USP product specific line
    model.M_d = pyo.Var(T, J, A, D, I, within=pyo.Binary)  # DSP product specific line
    model.M_g = pyo.Var(T, J, A, G, I, within=pyo.Binary)  # FF product specific line

    model.W_u = pyo.Var(T, J, A, U, I, S, within=pyo.Binary)  # product allocation USP
    model.W_d = pyo.Var(T, J, A, D, I, S, within=pyo.Binary)  # product allocation DSP
    model.W_g = pyo.Var(T, J, A, G, I, S, within=pyo.Binary)  # product allocation F

    # Production variables: integer batch counts, product output and processing
    # time at each stage. Interstage inventories are measured in batch equivalents.
    model.B_u = pyo.Var(T, J, A, U, I, S, within=pyo.NonNegativeIntegers)  # USP batches
    model.P_u = pyo.Var(T, J, A, U, I, S, within=pyo.NonNegativeReals)  # USP product output (kg mAb)
    model.T_u = pyo.Var(T, J, A, U, I, S, within=pyo.NonNegativeReals)  # USP time
    model.I_u = pyo.Var(T, J, A, I, S, within=pyo.NonNegativeReals)  # USP inventory (batches)

    model.B_d = pyo.Var(T, J, A, D, I, S, within=pyo.NonNegativeIntegers)  # DSP
    model.P_d = pyo.Var(T, J, A, D, I, S, within=pyo.NonNegativeReals)
    model.T_d = pyo.Var(T, J, A, D, I, S, within=pyo.NonNegativeReals)
    model.I_d = pyo.Var(T, J, A, I, S, within=pyo.NonNegativeReals)

    model.B_g = pyo.Var(T, J, A, G, I, S, within=pyo.NonNegativeIntegers)  # FF
    model.P_g = pyo.Var(T, J, A, G, I, S, within=pyo.NonNegativeReals)
    model.T_g = pyo.Var(T, J, A, G, I, S, within=pyo.NonNegativeReals)
    model.I_g = pyo.Var(T, J, A, I, S, within=pyo.NonNegativeReals)

    # Product distribution: flows go factory -> storage -> demand zone.
    # Reporting variables convert batch equivalents into kg of final product.
    model.Q_jk = pyo.Var(T, J, K, A, I, S, within=pyo.NonNegativeReals)  # Transport flows
    model.I_k = pyo.Var(T, K, A, I, S, within=pyo.NonNegativeReals)
    model.transport_jk = pyo.Var(T, J, K, I, S, within=pyo.NonNegativeReals)
    model.transport_kl = pyo.Var(T, K, L, I, S, within=pyo.NonNegativeReals)
    model.Q_kl = pyo.Var(T, K, L, A, I, S, within=pyo.NonNegativeReals)

    model.I_l = pyo.Var(T, L, A, I, S, within=pyo.NonNegativeReals)  # Demand-zone inventories
    model.S_l = pyo.Var(T, L, A, I, S, within=pyo.NonNegativeReals)
    model.Sales = pyo.Var(T, L, I, S, within=pyo.NonNegativeReals)
    model.Slack = pyo.Var(T, L, I, S, within=pyo.NonNegativeReals)
    model.expected_sales = pyo.Var(I, within=pyo.NonNegativeReals)

    # Cost-accounting variables define totals and breakdowns through equalities;
    # they do not represent additional physical flows. Values here remain USD.
    model.TC_CAPj = pyo.Var(T, S, I)
    model.TC_OPj = pyo.Var(T, S, I)
    model.TC_OPk = pyo.Var(T, S)
    model.TC_freight = pyo.Var(T, S)

    model.C_CAPj = pyo.Var(T, J, S, I, initialize=1, within=pyo.NonNegativeReals)
    model.C_OPj = pyo.Var(T, J, S, I, initialize=1, within=pyo.NonNegativeReals)
    model.C_VARj = pyo.Var(S, within=pyo.NonNegativeReals)
    model.C_FACj = pyo.Var(S, within=pyo.NonNegativeReals)
    model.C_LABj = pyo.Var(S, within=pyo.NonNegativeReals)
    model.C_OPk = pyo.Var(T, K, S, initialize=1, within=pyo.NonNegativeReals)
    model.C_freight_jk = pyo.Var(T, J, K, S, initialize=1, within=pyo.NonNegativeReals)
    model.C_freight_kl = pyo.Var(T, K, L, S, initialize=1, within=pyo.NonNegativeReals)
    model.total_demand = pyo.Var(T, S)
    model.total_lines = pyo.Var(T, S)

    # ------------ Additional costs for breakdown understanding -------------------------------------------

    model.cost = pyo.Var(S, I, within=pyo.NonNegativeReals)
    # model.missed_MAX = pyo.Param(default = 0)
    model.transport_cost = pyo.Var(S)
    model.storage_cost = pyo.Var(S)
    model.manufacturing_cost = pyo.Var(S, I)
    model.production_cost = pyo.Var(S, I)
    model.capital_risk = pyo.Var(S, I)

    model.demand = pyo.Var(T, L, I)

    # Objective function
    model.expected_cost = pyo.Var()

    def cost_rule(model, s, i):
        """Combine manufacturing operation, product freight, storage and site-activation charges."""
        return model.cost[s, i] == + sum(
            (model.TC_OPj[t, s, i] + model.TC_freight[t, s] + model.TC_OPk[t, s]) for t in T) + 100 * sum(
            (model.Y_j[1, j]) for j in J)

    model.COST = pyo.Constraint(S, I, rule=cost_rule)

    def capital_risk_rule(model, s, i):
        """Sum annualised capital charges; this legacy name does not represent statistical risk."""
        return model.capital_risk[s, i] == sum(model.TC_CAPj[t, s, i] for t in T)

    model.capital_riskkk = pyo.Constraint(S, I, rule=capital_risk_rule)

    def transport_rule(model, s):
        """Define the reported total freight cost across planning periods."""
        return model.transport_cost[s] == sum((model.TC_freight[t, s]) for t in T)

    model.transport_cost_cons = pyo.Constraint(S, rule=transport_rule)

    def manufacturing_rule(model, s, i):
        """Define the reported manufacturing operating cost across planning periods."""
        return model.manufacturing_cost[s, i] == sum((model.TC_OPj[t, s, i]) for t in T)

    model.manufacturing_cost_cons = pyo.Constraint(S, I, rule=manufacturing_rule)

    def storage_rule(model, s):
        """Define the reported storage-handling cost across planning periods."""
        return model.storage_cost[s] == sum((model.TC_OPk[t, s]) for t in T)

    model.storage_cost_cons = pyo.Constraint(S, rule=storage_rule)

    def total_capital_cost_rule(model, t, s, i):
        """Sum site capital charges to obtain the network capital cost."""
        return model.TC_CAPj[t, s, i] == sum((model.C_CAPj[t, j, s, i]) for j in J)

    model.total_capital_cost_constraint = pyo.Constraint(T, S, I, rule=total_capital_cost_rule)

    def capital_cost_rule(model, t, j, s, i):
        """Charge annualised capital for each product-enabled USP, DSP and FF line."""
        return model.C_CAPj[t, j, s, i] == sum(
            (C_CAPju[a, i, s] * model.M_u[t, j, a, u, i]) for a in A for u in U) + sum(
            (C_CAPjd[a, i, s] * model.M_d[t, j, a, d, i]) for a in A for d in D) + sum(
            (C_CAPjg[a, i, s] * model.M_g[t, j, a, g, i]) for a in A for g in G)

    model.capital_cost = pyo.Constraint(T, J, S, I, rule=capital_cost_rule)

    def total_operating_cost_rule(model, t, s, i):
        """Sum manufacturing operating costs over all sites."""
        return model.TC_OPj[t, s, i] == sum((model.C_OPj[t, j, s, i]) for j in J)

    model.total_operating_cost = pyo.Constraint(T, S, I, rule=total_operating_cost_rule)

    def total_varlab_cost_rule(model, s, i):
        """Report batch-dependent labour and variable costs separately from annual facility costs."""
        return model.production_cost[s, i] == sum(
            (C_LABju[a, i, s] * model.B_u[t, j, a, u, i, s] + C_VARju[a, i, s] * model.B_u[t, j, a, u, i, s]) for a in A
            for u in U for j in J for t in T) + sum(
            (C_LABjd[a, i, s] * model.B_d[t, j, a, d, i, s] + C_VARjd[a, i, s] * model.B_d[t, j, a, d, i, s]) for a in A
            for t in T for d in D for j in J) + + sum(
            (C_LABjg[a, i, s] * model.B_g[t, j, a, g, i, s] + C_VARjg[a, i, s] * model.B_g[t, j, a, g, i, s]) for a in A
            for g in G for j in J for t in T)

    model.total_varlab_cost = pyo.Constraint(S, I, rule=total_varlab_cost_rule)

    def operating_cost_rule(model, t, j, s, i):
        """Charge labour and variable costs per batch plus facility costs per product-enabled line."""
        return model.C_OPj[t, j, s, i] == (sum((C_LABju[a, i, s] * model.B_u[t, j, a, u, i, s] + C_VARju[a, i, s] *
                                                model.B_u[t, j, a, u, i, s] + C_FACju[a, i, s] * model.M_u[
                                                    t, j, a, u, i]) for a in A for u in U) + sum((C_LABjd[a, i, s] *
                                                                                                  model.B_d[
                                                                                                      t, j, a, d, i, s] +
                                                                                                  C_VARjd[a, i, s] *
                                                                                                  model.B_d[
                                                                                                      t, j, a, d, i, s] +
                                                                                                  C_FACjd[a, i, s] *
                                                                                                  model.M_d[
                                                                                                      t, j, a, d, i])
                                                                                                 for a in A for d in
                                                                                                 D) + sum((C_LABjg[
                                                                                                               a, i, s] *
                                                                                                           model.B_g[
                                                                                                               t, j, a, g, i, s] +
                                                                                                           C_VARjg[
                                                                                                               a, i, s] *
                                                                                                           model.B_g[
                                                                                                               t, j, a, g, i, s] +
                                                                                                           C_FACjg[
                                                                                                               a, i, s] *
                                                                                                           model.M_g[
                                                                                                               t, j, a, g, i])
                                                                                                          for a in A for
                                                                                                          g in G))

    model.operating_cost = pyo.Constraint(T, J, S, I, rule=operating_cost_rule)

    def total_operating_storage_cost_rule(model, t, s):
        """Sum handling costs across storage nodes."""
        return model.TC_OPk[t, s] == sum((model.C_OPk[t, k, s]) for k in K)

    model.total_operating_storage_cost = pyo.Constraint(T, S, rule=total_operating_storage_cost_rule)

    def operating_storage_cost_rule(model, t, k, s):
        """Charge storage handling on the mass shipped onward to demand zones."""
        return model.C_OPk[t, k, s] == sum(
            (t_s * model.Q_kl[t, k, l, a, i, s] * x_g[a, i, s]) for a in A for i in I for l in L)

    model.operating_storage_cost = pyo.Constraint(T, K, S, rule=operating_storage_cost_rule)

    def total_freight_cost_rule(model, t, s):
        """Combine factory-to-storage and storage-to-demand freight charges."""
        return model.TC_freight[t, s] == sum((model.C_freight_jk[t, j, k, s]) for j in J for k in K) + sum(
            (model.C_freight_kl[t, k, l, s]) for k in K for l in L)

    model.total_freight_cost = pyo.Constraint(T, S, rule=total_freight_cost_rule)

    def freight_cost_jk_rule(model, t, j, k, s):
        """Calculate factory-to-storage freight from distance and shipped product mass."""
        return model.C_freight_jk[t, j, k, s] == sum(
            (D_jk[j, k] * t_c * x_g[a, i, s] / product_unit[i] * model.Q_jk[t, j, k, a, i, s]) for a in A for i in I)

    model.freight_jk_cost = pyo.Constraint(T, J, K, S, rule=freight_cost_jk_rule)

    def freight_cost_kl_rule(model, t, k, l, s):
        """Calculate storage-to-demand freight from distance and shipped product mass."""
        return model.C_freight_kl[t, k, l, s] == sum(
            (D_kl[k, l] * t_c * x_g[a, i, s] / product_unit[i] * model.Q_kl[t, k, l, a, i, s]) for a in A for i in I)

    model.freight_kl_cost = pyo.Constraint(T, K, L, S, rule=freight_cost_kl_rule)

    #     #### Enough costs

    # Facility and line logic: connect site/scale activation to line use, order
    # equivalent lines, and ensure scenario allocations use enabled products.
    def manufacturing_existence_rule(model, t, j):
        """Allow one manufacturing scale at an open site and none at a closed site."""
        return sum((model.Y_ja[t, j, a]) for a in A) == model.Y_j[t, j]

    model.existence = pyo.Constraint(T, J, rule=manufacturing_existence_rule)

    def usp_scale_selection_rule(model, t, j, a, u, s):
        """Allow an active upstream line only at the selected site scale."""
        return model.Z_u[t, j, a, u] <= model.Y_ja[t, j, a]

    model.usp_scale_selection = pyo.Constraint(T, J, A, U, S, rule=usp_scale_selection_rule)

    def dsp_scale_selection_rule(model, t, j, a, d, s):
        """Allow an active downstream line only at the selected site scale."""
        return model.Z_d[t, j, a, d] <= model.Y_ja[t, j, a]

    model.dsp_scale_selection = pyo.Constraint(T, J, A, D, S, rule=dsp_scale_selection_rule)

    def ff_scale_selection_rule(model, t, j, a, g, s):
        """Allow an active fill-and-finish line only at the selected site scale."""
        return model.Z_g[t, j, a, g] <= model.Y_ja[t, j, a]

    model.ff_scale_selection = pyo.Constraint(T, J, A, G, S, rule=ff_scale_selection_rule)

    def degeneracy_usp_rule(model, t, j, a, u):
        """Activate upstream lines in index order to reduce equivalent solutions."""
        if u > 1:
            return model.Z_u[t, j, a, u] <= model.Z_u[t, j, a, u - 1]
        return pyo.Constraint.Skip

    model.degeneracy_usp = pyo.Constraint(T, J, A, U, rule=degeneracy_usp_rule)

    def degeneracy_dsp_rule(model, t, j, a, d):
        """Activate downstream lines in index order to reduce equivalent solutions."""
        if d > 1:
            return model.Z_d[t, j, a, d] <= model.Z_d[t, j, a, d - 1]
        return pyo.Constraint.Skip

    model.degeneracy_dsp = pyo.Constraint(T, J, A, D, rule=degeneracy_dsp_rule)

    def degeneracy_ff_rule(model, t, j, a, g):
        """Activate fill-and-finish lines in index order to reduce equivalent solutions."""
        if g > 1:
            return model.Z_g[t, j, a, g] <= model.Z_g[t, j, a, g - 1]
        return pyo.Constraint.Skip

    model.degeneracy_ff = pyo.Constraint(T, J, A, G, rule=degeneracy_ff_rule)

    def production_order_usp_rule(model, t, j, a, u, i, s):
        """Order upstream output from largest to smallest across numbered lines."""
        if u > 1:
            return model.P_u[t, j, a, u, i, s] <= model.P_u[t, j, a, u - 1, i, s]
        return pyo.Constraint.Skip

    model.production_order_usp = pyo.Constraint(T, J, A, U, I, S, rule=production_order_usp_rule)

    def production_order_dsp_rule(model, t, j, a, d, i, s):
        """Order downstream output from largest to smallest across numbered lines."""
        if d > 1:
            return model.P_d[t, j, a, d, i, s] <= model.P_d[t, j, a, d - 1, i, s]
        return pyo.Constraint.Skip

    model.production_order_dsp = pyo.Constraint(T, J, A, D, I, S, rule=production_order_dsp_rule)

    def production_order_ff_rule(model, t, j, a, g, i, s):
        """Order fill-and-finish output from largest to smallest across numbered lines."""
        if g > 1:
            return model.P_g[t, j, a, g, i, s] <= model.P_g[t, j, a, g - 1, i, s]
        return pyo.Constraint.Skip

    model.production_order_ff = pyo.Constraint(T, J, A, G, I, S, rule=production_order_ff_rule)

    def usp_product_allocation_rule(model, t, j, a, u, i):
        """Enable a product only on an active upstream line."""
        return model.M_u[t, j, a, u, i] <= model.Z_u[t, j, a, u]

    model.usp_product_allocation = pyo.Constraint(T, J, A, U, I, rule=usp_product_allocation_rule)

    def dsp_product_allocation_rule(model, t, j, a, d, i):
        """Enable a product only on an active downstream line."""
        return model.M_d[t, j, a, d, i] <= model.Z_d[t, j, a, d]

    model.dsp_product_allocation = pyo.Constraint(T, J, A, D, I, rule=dsp_product_allocation_rule)

    def ff_product_allocation_rule(model, t, j, a, g, i):
        """Enable a product only on an active fill-and-finish line."""
        return model.M_g[t, j, a, g, i] <= model.Z_g[t, j, a, g]

    model.ff_product_allocation = pyo.Constraint(T, J, A, G, I, rule=ff_product_allocation_rule)

    def usp_product_compatible_rule(model, t, j, a, u, i, s):
        """Assign scenario production only to a product-enabled upstream line."""
        return model.W_u[t, j, a, u, i, s] <= model.M_u[t, j, a, u, i]

    model.usp_product_compatible = pyo.Constraint(T, J, A, U, I, S, rule=usp_product_compatible_rule)

    def dsp_product_compatible_rule(model, t, j, a, d, i, s):
        """Assign scenario production only to a product-enabled downstream line."""
        return model.W_d[t, j, a, d, i, s] <= model.M_d[t, j, a, d, i]

    model.dsp_product_compatible = pyo.Constraint(T, J, A, D, I, S, rule=dsp_product_compatible_rule)

    def ff_product_compatible_rule(model, t, j, a, g, i, s):
        """Assign scenario production only to a product-enabled fill-and-finish line."""
        return model.W_g[t, j, a, g, i, s] <= model.M_g[t, j, a, g, i]

    model.ff_product_compatible = pyo.Constraint(T, J, A, G, I, S, rule=ff_product_compatible_rule)

    def network_link_jk_rule(model, t, j, k):
        """Permit an outbound factory-to-storage link only if the factory is open."""
        return model.X_jk[t, j, k] <= model.Y_j[t, j]

    model.network_link_jk = pyo.Constraint(T, J, K, rule=network_link_jk_rule)

    def network_link_kl_rule(model, t, k, l):
        """Permit a storage-to-demand link only if the storage node is open."""
        return model.X_kl[t, k, l] <= model.Y_k[t, k]

    model.network_link_kl = pyo.Constraint(T, K, L, rule=network_link_kl_rule)

    def sourcing_link_jk_rule(model, t, k):
        """Require an open storage node to have at least one incoming factory link."""
        return sum((model.X_jk[t, j, k]) for j in J) >= model.Y_k[t, k]

    model.sourcing_link_jk = pyo.Constraint(T, K, rule=sourcing_link_jk_rule)

    # Material conservation across USP -> DSP -> FF -> storage -> demand.
    # The zero-inventory limits below make stage batch totals match at each site.
    def inventory_ud_rule(model, t, j, a, i, s):
        """Balance USP batches against DSP batches and any carry-over USP inventory."""
        if t == 1:
            return model.I_u[t, j, a, i, s] == sum((model.B_u[t, j, a, u, i, s]) for u in U) - sum(
                (model.B_d[t, j, a, d, i, s]) for d in D)
        return model.I_u[t, j, a, i, s] == model.I_u[t - 1, j, a, i, s] + sum(
            (model.B_u[t, j, a, u, i, s]) for u in U) - sum((model.B_d[t, j, a, d, i, s]) for d in D)

    model.inventory_ud = pyo.Constraint(T, J, A, I, S, rule=inventory_ud_rule)

    def inventory_dg_rule(model, t, j, a, i, s):
        """Balance DSP batches against FF batches and any carry-over DSP inventory."""
        if t == 1:
            return model.I_d[t, j, a, i, s] == + sum((model.B_d[t, j, a, d, i, s]) for d in D) - sum(
                (model.B_g[t, j, a, g, i, s]) for g in G)
        return model.I_d[t, j, a, i, s] == model.I_d[t - 1, j, a, i, s] + sum(
            (model.B_d[t, j, a, d, i, s]) for d in D) - sum((model.B_g[t, j, a, g, i, s]) for g in G)

    model.inventory_dg = pyo.Constraint(T, J, A, I, S, rule=inventory_dg_rule)

    def inventory_j_rule(model, t, j, a, i, s):
        """Balance FF production against factory shipments in batch equivalents."""
        if t == 1:
            return model.I_g[t, j, a, i, s] == + sum((model.B_g[t, j, a, g, i, s]) for g in G) - sum(
                (model.Q_jk[t, j, k, a, i, s]) for k in K)
        return model.I_g[t, j, a, i, s] == model.I_g[t - 1, j, a, i, s] + sum(
            (model.B_g[t, j, a, g, i, s]) for g in G) - sum((model.Q_jk[t, j, k, a, i, s]) for k in K)

    model.inventory_j = pyo.Constraint(T, J, A, I, S, rule=inventory_j_rule)

    def inventory_k_rule(model, t, k, a, i, s):
        """Balance storage receipts against onward shipments in batch equivalents."""
        if t == 1:
            return model.I_k[t, k, a, i, s] == + sum((model.Q_jk[t, j, k, a, i, s]) for j in J) - sum(
                (model.Q_kl[t, k, l, a, i, s]) for l in L)
        return model.I_k[t, k, a, i, s] == model.I_k[t - 1, k, a, i, s] + sum(
            (model.Q_jk[t, j, k, a, i, s]) for j in J) - sum((model.Q_kl[t, k, l, a, i, s]) for l in L)

    model.inventory_k = pyo.Constraint(T, K, A, I, S, rule=inventory_k_rule)

    def inventory_l_rule(model, t, l, a, i, s):
        """Balance demand-zone receipts against deliveries in batch equivalents."""
        if t == 1:
            return model.I_l[t, l, a, i, s] == + sum((model.Q_kl[t, k, l, a, i, s]) for k in K) - model.S_l[
                t, l, a, i, s]
        return model.I_l[t, l, a, i, s] == model.I_l[t - 1, l, a, i, s] + sum(
            (model.Q_kl[t, k, l, a, i, s]) for k in K) - model.S_l[t, l, a, i, s]

    model.inventory_l = pyo.Constraint(T, L, A, I, S, rule=inventory_l_rule)

    def transport_flow_jk_rule(model, t, j, k, i, s):
        """Convert factory-to-storage batch-equivalent flows into product demand units."""
        return model.transport_jk[t, j, k, i, s] == sum(
            (model.Q_jk[t, j, k, a, i, s] * x_g[a, i, s] / product_unit[i]) for a in A)

    model.transport_flow_jk = pyo.Constraint(T, J, K, I, S, rule=transport_flow_jk_rule)

    def transport_flow_kl_rule(model, t, k, l, i, s):
        """Convert storage-to-demand batch-equivalent flows into product demand units."""
        return model.transport_kl[t, k, l, i, s] == sum(
            (model.Q_kl[t, k, l, a, i, s] * x_g[a, i, s] / product_unit[i]) for a in A)

    model.transport_flow_kl = pyo.Constraint(T, K, L, I, S, rule=transport_flow_kl_rule)

    def sales_product_rule(model, t, l, i, s):
        """Convert delivered batch equivalents into supplied demand units, kg in this case."""
        return model.Sales[t, l, i, s] == sum((model.S_l[t, l, a, i, s] * x_g[a, i, s] / product_unit[i]) for a in A)

    model.sales_product = pyo.Constraint(T, L, I, S, rule=sales_product_rule)

    # Process capacity: binary assignment enables time; duration and cycle
    # spacing determine how many whole batches each line can produce annually.
    def usp_time_allocation_max_rule(model, t, j, a, u, i, s):
        """Bound upstream time by annual capacity and force it to zero if unassigned."""
        return model.T_u[t, j, a, u, i, s] <= T_MAX * model.W_u[t, j, a, u, i, s]

    model.usp_time_allocation_max = pyo.Constraint(T, J, A, U, I, S, rule=usp_time_allocation_max_rule)

    def usp_time_allocation_min_rule(model, t, j, a, u, i, s):
        """Apply the configured minimum processing time to an assigned upstream line."""
        return T_MINu * model.W_u[t, j, a, u, i, s] <= model.T_u[t, j, a, u, i, s]

    model.usp_time_allocation_min = pyo.Constraint(T, J, A, U, I, S, rule=usp_time_allocation_min_rule)

    def max_utilisation_rule(model, t, j, a, u, s, i):
        """Apply the explicit annual USP time limit to each line and product."""
        return model.T_u[t, j, a, u, i, s] - H_ja <= 0

    model.max_utilisation_rule = pyo.Constraint(T, J, A, U, S, I, rule=max_utilisation_rule)

    def dsp_time_allocation_max_rule(model, t, j, a, d, i, s):
        """Bound downstream time by annual capacity and force it to zero if unassigned."""
        return model.T_d[t, j, a, d, i, s] <= T_MAX * model.W_d[t, j, a, d, i, s]

    model.dsp_time_allocation_max = pyo.Constraint(T, J, A, D, I, S, rule=dsp_time_allocation_max_rule)

    def dsp_time_allocation_min_rule(model, t, j, a, d, i, s):
        """Apply the configured minimum processing time to an assigned downstream line."""
        return T_MINu * model.W_d[t, j, a, d, i, s] <= model.T_d[t, j, a, d, i, s]

    model.dsp_time_allocation_min = pyo.Constraint(T, J, A, D, I, S, rule=dsp_time_allocation_min_rule)

    def ff_time_allocation_max_rule(model, t, j, a, g, i, s):
        """Bound fill-and-finish time by annual capacity and force it to zero if unassigned."""
        return model.T_g[t, j, a, g, i, s] <= T_MAX * model.W_g[t, j, a, g, i, s]

    model.ff_time_allocation_max = pyo.Constraint(T, J, A, G, I, S, rule=ff_time_allocation_max_rule)

    def ff_time_allocation_min_rule(model, t, j, a, g, i, s):
        """Apply the configured minimum processing time to an assigned fill-and-finish line."""
        return model.T_g[t, j, a, g, i, s] >= T_MINu * model.W_g[t, j, a, g, i, s]

    model.ff_time_allocation_min = pyo.Constraint(T, J, A, G, I, S, rule=ff_time_allocation_min_rule)

    def ff_batches_rule(model, t, j, a, g, i, s):
        """Link integer fill-and-finish batches to allocated time, first-batch duration and cycle spacing."""
        return model.B_g[t, j, a, g, i, s] == model.W_g[t, j, a, g, i, s] + 1 / r_g * (
                    model.T_g[t, j, a, g, i, s] - alpha_g[s, i] * model.W_g[
                t, j, a, g, i, s])  #### Need to recheck this expression

    model.ff_batches = pyo.Constraint(T, J, A, G, I, S, rule=ff_batches_rule)

    def usp_batches_rule(model, t, j, a, u, i, s):
        """Link integer upstream batches to allocated time, first-batch duration and cycle spacing."""
        return model.B_u[t, j, a, u, i, s] == model.W_u[t, j, a, u, i, s] + 1 / r_u[s, i] * (
                    model.T_u[t, j, a, u, i, s] - alpha_u[s, i] * model.W_u[t, j, a, u, i, s])

    model.usp_batches = pyo.Constraint(T, J, A, U, I, S, rule=usp_batches_rule)

    def dsp_batches_rule(model, t, j, a, d, i, s):
        """Link integer downstream batches to allocated time, first-batch duration and cycle spacing."""
        return model.B_d[t, j, a, d, i, s] == model.W_d[t, j, a, d, i, s] + 1 / r_d * (
                    model.T_d[t, j, a, d, i, s] - alpha_d[s, i] * model.W_d[t, j, a, d, i, s])

    model.dsp_batches = pyo.Constraint(T, J, A, D, I, S, rule=dsp_batches_rule)

    def usp_mass_produced_rule(model, t, j, a, u, i, s):
        """Convert upstream batch counts to final-product-equivalent output."""
        return model.P_u[t, j, a, u, i, s] == model.B_u[t, j, a, u, i, s] * x_u[a, i, s] / product_unit[i]

    model.usp_mass_produced = pyo.Constraint(T, J, A, U, I, S, rule=usp_mass_produced_rule)

    def dsp_mass_produced_rule(model, t, j, a, d, i, s):
        """Convert downstream batch counts to final-product-equivalent output."""
        return model.P_d[t, j, a, d, i, s] == model.B_d[t, j, a, d, i, s] * x_d[a, i, s] / product_unit[i]

    model.dsp_mass_produced = pyo.Constraint(T, J, A, D, I, S, rule=dsp_mass_produced_rule)

    def ff_mass_produced_rule(model, t, j, a, g, i, s):
        """Convert fill-and-finish batch counts to final-product-equivalent output."""
        return model.P_g[t, j, a, g, i, s] == model.B_g[t, j, a, g, i, s] * x_g[a, i, s] / product_unit[i]

    model.ff_mass_produced = pyo.Constraint(T, J, A, G, I, S, rule=ff_mass_produced_rule)

    # Link capacities: aggregate shipments are bounded by link activation.
    def min_transport_jk_rule(model, t, j, k, s):
        """Apply the minimum factory-to-storage shipment bound; it is zero in this case."""
        return Q_min * model.X_jk[t, j, k] <= sum((model.Q_jk[t, j, k, a, i, s]) for i in I for a in A)

    model.min_transport_jk = pyo.Constraint(T, J, K, S, rule=min_transport_jk_rule)

    def max_transport_jk_rule(model, t, j, k, s):
        """Permit factory-to-storage shipments only on an active link, up to its flow bound."""
        return sum((model.Q_jk[t, j, k, a, i, s]) for i in I for a in A) <= Q_max * model.X_jk[t, j, k]

    model.max_transport_jk = pyo.Constraint(T, J, K, S, rule=max_transport_jk_rule)

    def min_transport_kl_rule(model, t, k, l, s):
        """Apply the minimum storage-to-demand shipment bound; it is zero in this case."""
        return Q_min * model.X_kl[t, k, l] <= sum((model.Q_kl[t, k, l, a, i, s]) for i in I for a in A)

    model.min_transport_kl = pyo.Constraint(T, K, L, S, rule=min_transport_kl_rule)

    def max_transport_kl_rule(model, t, k, l, s):
        """Permit storage-to-demand shipments only on an active link, up to its flow bound."""
        return sum((model.Q_kl[t, k, l, a, i, s]) for i in I for a in A) <= Q_max * model.X_kl[t, k, l]

    model.max_transport_kl = pyo.Constraint(T, K, L, S, rule=max_transport_kl_rule)

    # End-of-period stock is zero at every node and production stage.
    def inventory_max_usp_rule(model, t, j, a, i, s):
        """Fix nonnegative USP ending inventory to zero through its upper bound."""
        return model.I_u[t, j, a, i, s] <= 0

    model.inventory_max_usp = pyo.Constraint(T, J, A, I, S, rule=inventory_max_usp_rule)

    def inventory_max_dsp_rule(model, t, j, a, i, s):
        """Fix nonnegative DSP ending inventory to zero through its upper bound."""
        return model.I_d[t, j, a, i, s] <= 0

    model.inventory_max_dsp = pyo.Constraint(T, J, A, I, S, rule=inventory_max_dsp_rule)

    def inventory_max_ff_rule(model, t, j, a, i, s):
        """Fix nonnegative FF ending inventory to zero through its upper bound."""
        return model.I_g[t, j, a, i, s] <= 0

    model.inventory_max_ff = pyo.Constraint(T, J, A, I, S, rule=inventory_max_ff_rule)

    def inventory_max_stor_rule(model, t, k, a, i, s):
        """Fix nonnegative storage ending inventory to zero through its upper bound."""
        return model.I_k[t, k, a, i, s] <= 0

    model.inventory_max_stor = pyo.Constraint(T, K, A, I, S, rule=inventory_max_stor_rule)

    def inventory_max_demand_rule(model, t, l, a, i, s):
        """Fix nonnegative demand-zone ending inventory to zero through its upper bound."""
        return model.I_l[t, l, a, i, s] <= 0

    model.inventory_max_demand = pyo.Constraint(T, L, A, I, S, rule=inventory_max_demand_rule)

    # Demand, scenario accounting and the original objective. The joint builder
    # retains these equalities but replaces the active objective with joint cost.
    def expected_cost_rule(model):
        """Probability-weight operating and annualised capital costs across scenarios."""
        return model.expected_cost == sum(
            (phi[s] * (model.cost[s, i] + model.capital_risk[s, i])) for i in I for s in S)

    model.expect_cost = pyo.Constraint(rule=expected_cost_rule)

    def missed_deliveries_rule(model, t, l, i, s):
        """Meet demand through sales, allowing shortage slack only when configured."""
        if sales_activation == True:
            return D_l[t, l, i] <= model.Sales[t, l, i, s]
        return D_l[t, l, i] <= model.Sales[t, l, i, s] + model.Slack[t, l, i, s]

    model.missed_deliveries = pyo.Constraint(T, L, I, S, rule=missed_deliveries_rule)

    def actual_demand(model, t, l, i):
        """Store the fixed demand parameter in a reporting variable."""
        return + model.demand[t, l, i] == D_l[t, l, i]

    model.actual = pyo.Constraint(T, L, I, rule=actual_demand)

    def objective_rule(model):
        """Define the original supply objective and optional shortage penalty; it is deactivated in the joint model."""
        if sales_activation == True:
            return model.expected_cost
        return model.expected_cost + sum((model.Slack[t, l, i, s] * 1e12) for t in T for l in L for i in I for s in S)

    model.objective1 = pyo.Objective(rule=objective_rule)

    def expected_sales_rule(model, i):
        """Probability-weight total delivered product across scenarios."""
        return model.expected_sales[i] == sum((phi[s] * model.Sales[t, l, i, s]) for s in S for t in T for l in L)

    model.expected_saleszzz = pyo.Constraint(I, rule=expected_sales_rule)

    return model


# %% Direct mAb parameter assignment and first deterministic case
#
# Deterministic supply-chain parameters retained from the original model.
# The joint IS block below supplies waste routing and utilities decisions.


def build_first_mab_input_data():
    """
    Create the deterministic mAb input dictionaries.

    The case represents one annual planning period for an illustrative
    multi-product commercial mAb network.

    Capital costs are annualised using a capital recovery factor so that all
    objective-function terms are expressed on an equivalent USD/year basis.
    """

    scenarios = ["S1"]
    scenario_probability = {"S1": 1.0}
    scales = ["L"]
    products = ["mAb"]

    # ------------------------------------------------------------------
    # Network demand assumption
    # ------------------------------------------------------------------
    # Aggregate portfolio demand rather than demand for one named mAb.
    #
    # At 4.65 kg per batch, 4,000 kg/year requires approximately 861
    # complete batches across the manufacturing network.
    total_demand_mab = {
        1: 4_000.0,
    }  # kg final mAb drug substance/year

    # ------------------------------------------------------------------
    # SuperPro manufacturing performance: 1,000 L SUT process
    # ------------------------------------------------------------------
    SUPERPRO_BATCH_OUTPUT_KG = 4.65
    SUPERPRO_RECIPE_CYCLE_TIME_DAYS = 2.50
    SUPERPRO_BATCHES_PER_YEAR = 121
    SUPERPRO_OPERATING_HORIZON_DAYS = 329

    # Section spans extracted from the SuperPro schedule.
    # Sections can overlap because operations are synchronised.
    SUPERPRO_USP_TIME_DAYS = 659.25 / 24.0
    SUPERPRO_DSP_TIME_DAYS = (691.58 - 656.25) / 24.0
    SUPERPRO_FILLING_TIME_DAYS = (694.83 - 691.58) / 24.0

    r_u = {}
    alpha_u = {}
    alpha_d = {}
    alpha_g = {}
    x_g = {}

    # ------------------------------------------------------------------
    # Annualisation assumptions
    # ------------------------------------------------------------------
    # The model contains one representative annual period. Consequently,
    # the full upfront SuperPro investment is converted to an equivalent
    # annual cost using the capital recovery factor:
    #
    #              r(1+r)^n
    # CRF = -------------------------
    #           (1+r)^n - 1
    #
    # A later multi-period formulation could instead charge investment in
    # the construction period and use discounted inter-period cash flows.
    CAPITAL_LIFETIME_YEARS = 15
    DISCOUNT_RATE = 0.08

    capital_recovery_factor = (
        DISCOUNT_RATE
        * (1.0 + DISCOUNT_RATE) ** CAPITAL_LIFETIME_YEARS
        / (
            (1.0 + DISCOUNT_RATE) ** CAPITAL_LIFETIME_YEARS
            - 1.0
        )
    )

    # ------------------------------------------------------------------
    # Cost dictionaries
    # ------------------------------------------------------------------
    # All monetary values retain the SuperPro 2019 USD basis.
    C_LABju, C_LABjd, C_LABjg = {}, {}, {}
    C_CAPju, C_CAPjd, C_CAPjg = {}, {}, {}
    C_FACju, C_FACjd, C_FACjg = {}, {}, {}
    C_VARju, C_VARjd, C_VARjg = {}, {}, {}

    # Labour costs per completed batch.
    SUPERPRO_USP_LABOUR_PER_BATCH = 38_226.31404958678
    SUPERPRO_DSP_LABOUR_PER_BATCH = 11_488.900826446281
    SUPERPRO_FILLING_LABOUR_PER_BATCH = 0.0

    # Variable costs per completed batch.
    # Waste treatment and disposal are excluded because waste-routing costs
    # will be represented separately in the industrial-symbiosis stage.
    SUPERPRO_USP_VARIABLE_PER_BATCH = 47_843.82644628099
    SUPERPRO_DSP_VARIABLE_PER_BATCH = 98_406.19834710743
    SUPERPRO_FILLING_VARIABLE_PER_BATCH = 0.0

    # Annual facility-dependent operating costs.
    SUPERPRO_USP_FACILITY_PER_YEAR = 16_951_300.00
    SUPERPRO_DSP_FACILITY_PER_YEAR = 7_855_864.00
    SUPERPRO_FILLING_FACILITY_PER_YEAR = 0.0

    # Total upfront capital investment from SuperPro.
    SUPERPRO_USP_TOTAL_CAPITAL = 123_832_934.00
    SUPERPRO_DSP_TOTAL_CAPITAL = 57_327_536.00
    SUPERPRO_FILLING_TOTAL_CAPITAL = 0.0

    # Equivalent annual capital costs used in the single-period objective.
    SUPERPRO_USP_ANNUALISED_CAPITAL = (
        SUPERPRO_USP_TOTAL_CAPITAL
        * capital_recovery_factor
    )
    SUPERPRO_DSP_ANNUALISED_CAPITAL = (
        SUPERPRO_DSP_TOTAL_CAPITAL
        * capital_recovery_factor
    )
    SUPERPRO_FILLING_ANNUALISED_CAPITAL = (
        SUPERPRO_FILLING_TOTAL_CAPITAL
        * capital_recovery_factor
    )

    for s in scenarios:
        for i in products:
            # USP cycle spacing controls the maximum number of overlapping
            # SuperPro recipes that can be initiated on one upstream line.
            r_u[s, i] = SUPERPRO_RECIPE_CYCLE_TIME_DAYS

            alpha_u[s, i] = SUPERPRO_USP_TIME_DAYS
            alpha_d[s, i] = SUPERPRO_DSP_TIME_DAYS
            alpha_g[s, i] = SUPERPRO_FILLING_TIME_DAYS

            for a in scales:
                x_g[a, i, s] = SUPERPRO_BATCH_OUTPUT_KG

                C_LABju[a, i, s] = (
                    SUPERPRO_USP_LABOUR_PER_BATCH
                )
                C_LABjd[a, i, s] = (
                    SUPERPRO_DSP_LABOUR_PER_BATCH
                )
                C_LABjg[a, i, s] = (
                    SUPERPRO_FILLING_LABOUR_PER_BATCH
                )

                C_VARju[a, i, s] = (
                    SUPERPRO_USP_VARIABLE_PER_BATCH
                )
                C_VARjd[a, i, s] = (
                    SUPERPRO_DSP_VARIABLE_PER_BATCH
                )
                C_VARjg[a, i, s] = (
                    SUPERPRO_FILLING_VARIABLE_PER_BATCH
                )

                C_FACju[a, i, s] = (
                    SUPERPRO_USP_FACILITY_PER_YEAR
                )
                C_FACjd[a, i, s] = (
                    SUPERPRO_DSP_FACILITY_PER_YEAR
                )
                C_FACjg[a, i, s] = (
                    SUPERPRO_FILLING_FACILITY_PER_YEAR
                )

                # Annualised capital replaces full upfront capital in the
                # objective because the model represents one annual period.
                C_CAPju[a, i, s] = (
                    SUPERPRO_USP_ANNUALISED_CAPITAL
                )
                C_CAPjd[a, i, s] = (
                    SUPERPRO_DSP_ANNUALISED_CAPITAL
                )
                C_CAPjg[a, i, s] = (
                    SUPERPRO_FILLING_ANNUALISED_CAPITAL
                )

    # Confirm that the SuperPro duration and cycle-time assumptions reproduce
    # its reported annual upstream throughput.
    calculated_batches_per_year = int(
        1
        + (
            SUPERPRO_OPERATING_HORIZON_DAYS
            - SUPERPRO_USP_TIME_DAYS
        )
        / SUPERPRO_RECIPE_CYCLE_TIME_DAYS
    )

    if not (calculated_batches_per_year == SUPERPRO_BATCHES_PER_YEAR):
        raise AssertionError(f'SuperPro throughput validation failed: calculated {calculated_batches_per_year} batches/year, expected {SUPERPRO_BATCHES_PER_YEAR}.')

    print("\nDeterministic mAb model assumptions:")
    print(
        f"  Aggregate annual demand: "
        f"{total_demand_mab[1]:,.0f} kg/year"
    )
    print(
        f"  Batch output: "
        f"{SUPERPRO_BATCH_OUTPUT_KG:.2f} kg/batch"
    )
    print(
        f"  SuperPro throughput: "
        f"{SUPERPRO_BATCHES_PER_YEAR} batches/USP line-year"
    )
    print(
        f"  Capital lifetime: "
        f"{CAPITAL_LIFETIME_YEARS} years"
    )
    print(
        f"  Discount rate: "
        f"{DISCOUNT_RATE:.1%}"
    )
    print(
        f"  Capital recovery factor: "
        f"{capital_recovery_factor:.5f}"
    )
    print(
        f"  Annualised USP capital per line: "
        f"${SUPERPRO_USP_ANNUALISED_CAPITAL:,.2f}/year"
    )
    print(
        f"  Annualised DSP capital per line: "
        f"${SUPERPRO_DSP_ANNUALISED_CAPITAL:,.2f}/year"
    )

    return {
        "scenarios": scenarios,
        "scenario_probability": scenario_probability,
        "r_u": r_u,
        "alpha_u": alpha_u,
        "alpha_d": alpha_d,
        "alpha_g": alpha_g,
        "x_g": x_g,
        "C_LABju": C_LABju,
        "C_LABjd": C_LABjd,
        "C_LABjg": C_LABjg,
        "C_CAPju": C_CAPju,
        "C_CAPjd": C_CAPjd,
        "C_CAPjg": C_CAPjg,
        "C_FACju": C_FACju,
        "C_FACjd": C_FACjd,
        "C_FACjg": C_FACjg,
        "C_VARju": C_VARju,
        "C_VARjd": C_VARjd,
        "C_VARjg": C_VARjg,
        "total_demand_mab": total_demand_mab,
        "capital_recovery_factor": capital_recovery_factor,
        "capital_lifetime_years": CAPITAL_LIFETIME_YEARS,
        "discount_rate": DISCOUNT_RATE,
    }


def extract_first_mab_results(model):
    """Extract a compact set of network and production results."""
    rows = []
    for j in ["j1", "j2", "j3"]:
        for a in ["L"]:
            site_open = pyo.value(model.Y_ja[1, j, a])
            usp_lines = sum(pyo.value(model.Z_u[1, j, a, u]) for u in [1, 2, 3, 4])
            dsp_lines = sum(pyo.value(model.Z_d[1, j, a, d]) for d in [1, 2, 3, 4])
            ff_lines = sum(pyo.value(model.Z_g[1, j, a, g]) for g in [1, 2, 3, 4])
            batches = sum(
                pyo.value(model.B_g[1, j, a, g, "mAb", "S1"])
                for g in [1, 2, 3, 4]
            )
            rows.append(
                {
                    "manufacturing_site": j,
                    "scale": a,
                    "site_open": site_open,
                    "usp_lines": usp_lines,
                    "dsp_lines": dsp_lines,
                    "fill_finish_lines": ff_lines,
                    "mab_batches_per_year": batches,
                }
            )
    return pd.DataFrame(rows)





def build_joint_model(data=None, supply_data=None):
    """Attach the IS block to the supply-chain model, with one objective."""
    data = load_joint_inputs() if data is None else data
    supply_data = build_first_mab_input_data() if supply_data is None else supply_data
    require(supply_data['scenarios'] == ['S1']
            and supply_data['scenario_probability'] == {'S1': 1.0},
            'This joint annual formulation supports the deterministic S1 case only.')
    kwargs = {key: value for key, value in supply_data.items()
              if key.startswith(('C_LAB', 'C_VAR', 'C_FAC', 'C_CAP'))
              or key in {'r_u', 'alpha_u', 'alpha_d', 'alpha_g', 'x_g', 'total_demand_mab'}}
    model = create_mab_supply_chain_model(
        scenario_set=supply_data['scenarios'],
        scenario_probability=supply_data['scenario_probability'],
        sales_activation=False, **kwargs)
    # Preserve the supply model's shortage allowance and penalty.
    model.objective1.deactivate()
    # Shared decisions: IS uses the same site-opening binaries as manufacturing.
    # site_batches sums FF lines; it is an expression, not a separate allocation.
    model.JOINT_SITES = pyo.Set(initialize=data.sites, ordered=True)
    model.site_batches = pyo.Expression(model.JOINT_SITES, rule=lambda m, j:
        sum(m.B_g[1, j, 'L', g, 'mAb', 'S1'] for g in [1, 2, 3, 4]))

    # Valid bounds come from all three production-stage capacities, not from
    # an old supply-chain solution's high/low batch allocation.
    limits = []
    for stage, spacing in [('u', supply_data['r_u']['S1', 'mAb']), ('d', 1.0), ('g', 1.0)]:
        alpha = supply_data[f'alpha_{stage}']['S1', 'mAb']
        per_line = math.floor(1 + (329.0 - alpha) / spacing + 1e-10)
        limits.append(4 * per_line)
        for variable in getattr(model, f'B_{stage}').values():
            variable.setub(per_line)
    max_batches = min(limits)
    require(max_batches > 0, 'No positive annual production capacity.')
    # IS decision block: eligible plastic flows, whole vehicle journeys,
    # electricity purchases, exclusive biomass routes and water-recovery flows.
    is_block = model.is_block = pyo.Block()
    is_block.PAIRS = pyo.Set(dimen=2, initialize=data.eligible)
    is_block.ROUTES = pyo.Set(initialize=[r for r in data.routes if any(pair[1] == r for pair in data.eligible)])
    is_block.MODES = pyo.Set(initialize=data.modes)
    is_block.SOURCES = pyo.Set(initialize=list(data.electricity_cost))
    is_block.BIO_ROUTES = pyo.Set(initialize=data.biomass_routes)
    is_block.x = pyo.Var(model.JOINT_SITES, is_block.PAIRS, is_block.MODES,
        domain=pyo.NonNegativeReals,
        bounds=lambda b, j, p, r, l: (0, max_batches * data.polymer_kg[p] / 1000.0))
    max_plastic = max_batches * sum(data.polymer_kg.values()) / 1000.0
    is_block.n = pyo.Var(model.JOINT_SITES, is_block.ROUTES, is_block.MODES,
        domain=pyo.NonNegativeIntegers,
        bounds=lambda b, j, r, l: (0, math.ceil(max_plastic / data.payload[l])))
    is_block.E = pyo.Var(model.JOINT_SITES, is_block.SOURCES,
        domain=pyo.NonNegativeReals, bounds=(0, max_batches * data.electricity))
    is_block.bio_y = pyo.Var(model.JOINT_SITES, is_block.BIO_ROUTES, domain=pyo.Binary)
    is_block.bio_q = pyo.Var(model.JOINT_SITES, is_block.BIO_ROUTES,
        domain=pyo.NonNegativeReals, bounds=(0, max_batches * data.biomass))
    is_block.qDIR = pyo.Var(model.JOINT_SITES, domain=pyo.NonNegativeReals)
    is_block.qREU = pyo.Var(model.JOINT_SITES, domain=pyo.NonNegativeReals)
    is_block.u = pyo.Var(model.JOINT_SITES, domain=pyo.NonNegativeReals)
    is_block.f = pyo.Var(model.JOINT_SITES, domain=pyo.NonNegativeReals)
    is_block.d = pyo.Var(model.JOINT_SITES, domain=pyo.NonNegativeReals)
    is_block.constraints = pyo.ConstraintList()
    # IS balances and capacities: each additional batch generates the fixed
    # per-batch inventories; closed sites have zero production and IS activity.
    for j in data.sites:
        B, y = model.site_batches[j], model.Y_j[1, j]
        is_block.constraints.add(B <= max_batches * y)
        is_block.constraints.add(sum(is_block.bio_y[j, r] for r in data.biomass_routes) == y)
        is_block.constraints.add(sum(is_block.bio_q[j, r] for r in data.biomass_routes) == data.biomass * B)
        for r in data.biomass_routes:
            is_block.constraints.add(is_block.bio_q[j, r] <= max_batches * data.biomass * is_block.bio_y[j, r])
        # Every polymer must be fully routed; partial vehicle loads are allowed.
        for p in data.polymers:
            is_block.constraints.add(sum(is_block.x[j, p, r, l]
                for pp, r in data.eligible if pp == p for l in data.modes) == data.polymer_kg[p] * B / 1000.0)
        for r in is_block.ROUTES:
            for l in data.modes:
                is_block.constraints.add(sum(is_block.x[j, p, r, l]
                    for p, rr in data.eligible if rr == r) <= data.payload[l] * is_block.n[j, r, l])
                is_block.constraints.add(is_block.n[j, r, l] <= math.ceil(max_plastic / data.payload[l]) * y)
        # Electricity must cover production demand. Water can be reused or
        # discharged, subject to recovery efficiency and treatment capacity.
        is_block.constraints.add(sum(is_block.E[j, e] for e in is_block.SOURCES) == data.electricity * B)
        is_block.constraints.add(is_block.qDIR[j] + is_block.qREU[j] == data.wastewater * B)
        is_block.constraints.add(is_block.u[j] == data.recovery * is_block.qREU[j])
        is_block.constraints.add(is_block.f[j] + is_block.u[j] == data.utility_water * B)
        is_block.constraints.add(is_block.d[j] + is_block.u[j] == data.wastewater * B)
        is_block.constraints.add(data.wastewater * B <= data.treatment_capacity * y)

    # IS annual private costs: plastic net treatment, transport, purchased
    # electricity, wastewater/fresh water/discharge and fixed biomass collection.
    model.is_plastics_cost = pyo.Expression(expr=sum(data.route_cost[p, r] * is_block.x[j, p, r, l]
        for j in data.sites for p, r in data.eligible for l in data.modes))
    model.is_transport_cost = pyo.Expression(expr=sum(2.0 * data.distance[j, r] * data.transport_cost[l] * is_block.n[j, r, l]
        for j in data.sites for r in is_block.ROUTES for l in data.modes))
    model.is_electricity_cost = pyo.Expression(expr=sum(data.electricity_cost[e] * is_block.E[j, e]
        for j in data.sites for e in is_block.SOURCES))
    model.is_water_cost = pyo.Expression(expr=sum(
        data.treatment_fixed * model.Y_j[1, j]
        + data.treatment_variable * data.wastewater * model.site_batches[j]
        + data.fresh_cost[j] * is_block.f[j] + data.discharge_cost[j] * is_block.d[j]
        for j in data.sites))
    model.is_biomass_cost = pyo.Expression(expr=sum(data.biomass_collection * model.Y_j[1, j] for j in data.sites))
    model.is_cost_gbp = pyo.Expression(expr=model.is_plastics_cost + model.is_transport_cost
        + model.is_electricity_cost + model.is_water_cost + model.is_biomass_cost)
    # Joint cost reconciliation: convert the supply total to GBP 2026, remove
    # its embedded electricity charge, then add the IS costs once. The separate
    # shortage penalty preserves the original supply model demand treatment.
    model.supply_original_cost_gbp = pyo.Expression(expr=data.usd_to_gbp * model.expected_cost)
    model.embedded_electricity_deduction_gbp = pyo.Expression(expr=
        data.usd_to_gbp * data.embedded_electricity_usd * sum(model.site_batches[j] for j in data.sites))
    model.supply_cost_gbp = pyo.Expression(expr=
        model.supply_original_cost_gbp - model.embedded_electricity_deduction_gbp)
    model.total_cost_gbp = pyo.Expression(expr=model.supply_cost_gbp + model.is_cost_gbp)
    model.shortage_penalty_gbp = pyo.Expression(expr=
        data.usd_to_gbp * 1e12 * sum(model.Slack[key] for key in model.Slack))
    model.joint_objective = pyo.Objective(expr=model.total_cost_gbp + model.shortage_penalty_gbp, sense=pyo.minimize)
    return model, data


def export_gams(model, output_dir=RESULT_DIR):
    """Write one self-contained GAMS input with one solve statement."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'joint_supply_chain_is.gms'
    # The writer translates all active linear equations into one self-contained
    # .gms input. Its symbol map links returned GAMS levels to Pyomo variables.
    filename, smap_id = model.write(str(path), format='gams', io_options={
        'solver': GAMS_SOLVER, 'mtype': 'mip', 'symbolic_solver_labels': True,
        'add_options': [f'option optcr={MIP_REL_GAP};', 'option optca=0;'],
        'put_results': 'joint_solution', 'put_results_format': 'dat',
    })
    symbol_map = model.solutions.symbol_map[smap_id]
    # A readable map allows inspecting exported symbols without Python.
    pd.DataFrame([{'gams_symbol': name, 'python_component': obj.name}
                  for name, obj in symbol_map.bySymbol.items()]).to_csv(output_dir / 'symbol_map.csv', index=False)
    return Path(filename), symbol_map


def validate_solution(model, tolerance=1e-5):
    """Check active constraints, variable bounds and integrality after loading GAMS results."""
    max_violation = 0.0
    for con in model.component_data_objects(pyo.Constraint, active=True):
        body = pyo.value(con.body)
        if con.has_lb():
            max_violation = max(max_violation, pyo.value(con.lower) - body)
        if con.has_ub():
            max_violation = max(max_violation, body - pyo.value(con.upper))
    for var in model.component_data_objects(pyo.Var, active=True):
        if var.value is None:  # unused reporting variables are absent from the solver
            continue
        if var.has_lb():
            max_violation = max(max_violation, pyo.value(var.lb) - var.value)
        if var.has_ub():
            max_violation = max(max_violation, var.value - pyo.value(var.ub))
        if var.is_integer():
            max_violation = max(max_violation, abs(var.value - round(var.value)))
    require(max_violation <= tolerance, f'Joint solution violates constraints/bounds by {max_violation}.')
    return max_violation


def solve_joint_model(model, output_dir=RESULT_DIR, gams_executable=None):
    """Export and solve one GAMS model, load variable levels and verify status, optimality gap and feasibility."""
    gams_path, symbol_map = export_gams(model, output_dir)
    executable = gams_executable or shutil.which('gams')
    require(executable is not None, 'GAMS is not on PATH. Add your GAMS installation to PATH or use --gams-executable.')
    output_dir = gams_path.parent
    # Relative GAMS filenames and cwd avoid Windows key=path quoting failures.
    for name in ['joint_solution.dat', 'joint_solutionstat.dat']:
        (output_dir / name).unlink(missing_ok=True)
    command = [str(executable), gams_path.name, 'curdir=.', 'lo=3', 'o=joint_solver.lst']
    completed = subprocess.run(command, cwd=output_dir, text=True, errors='replace',
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = completed.stdout or ''
    (output_dir / 'joint_solver.log').write_text(output, encoding='utf-8')
    print(output)
    require(completed.returncode == 0, f'GAMS failed with code {completed.returncode}; see {output_dir / "joint_solver.log"}.')
    # Read solver status before accepting any levels. A normal MIP solve must
    # meet the requested gap; a merely feasible incumbent is not sufficient.
    statuses = {}
    for line in (output_dir / 'joint_solutionstat.dat').read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2:
            statuses[parts[0]] = float(parts[1])
    # GAMS reports status 8 for a MIP optimum within the requested tolerance.
    gap = abs(statuses.get('OBJVAL', math.inf) - statuses.get('OBJEST', -math.inf))
    relative_gap = gap / max(1.0, abs(statuses.get('OBJVAL', 0.0)))
    statuses['relative_gap'] = relative_gap
    require(statuses.get('SOLVESTAT') == 1
            and (statuses.get('MODELSTAT') == 1
                 or (statuses.get('MODELSTAT') == 8 and relative_gap <= MIP_REL_GAP + 1e-12)),
            f'GAMS did not reach the requested optimality tolerance: {statuses}. See joint_solver.lst.')
    # Load primal values only; MIP marginal values may be NA. Round integer
    # values only within tolerance, then recheck the original model equations.
    for line in (output_dir / 'joint_solution.dat').read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        obj = symbol_map.bySymbol.get(parts[0])
        if obj is not None and obj.is_variable_type():
            level = float(parts[1])
            if obj.is_integer():
                require(abs(level - round(level)) <= 1e-5, f'Nonintegral solver level for {obj.name}: {level}.')
                level = round(level)
            obj.set_value(level)
    statuses['max_constraint_violation'] = validate_solution(model)
    require(math.isclose(pyo.value(model.joint_objective), statuses['OBJVAL'], rel_tol=1e-9, abs_tol=0.01),
            'Loaded objective does not match the GAMS result.')
    (output_dir / 'solver_status.json').write_text(json.dumps(statuses, indent=2), encoding='utf-8')
    return statuses


def write_joint_results(model, data, output_dir=RESULT_DIR):
    """Save production, routing, utilities and annual cost tables, then print the cost contributions."""
    output_dir = Path(output_dir)
    # Production and physical-flow outputs describe the same joint solution.
    supply = extract_first_mab_results(model)
    supply['site_location'] = supply['manufacturing_site'].map(data.site_location)
    supply.to_csv(output_dir / 'joint_supply_chain_results.csv', index=False)
    block = model.is_block
    pd.DataFrame([{'site': j, 'polymer': p, 'route': r, 'mode': l,
                   'tonnes_per_year': pyo.value(block.x[j, p, r, l])}
                  for j in data.sites for p, r in data.eligible for l in data.modes]).to_csv(
                      output_dir / 'joint_plastic_routing.csv', index=False)
    pd.DataFrame([{'site': j, 'route': r, 'mode': l,
                   'journeys': pyo.value(block.n[j, r, l])}
                  for j in data.sites for r in block.ROUTES for l in data.modes]).to_csv(
                      output_dir / 'joint_waste_transport.csv', index=False)
    pd.DataFrame([{'site': j, 'batches': pyo.value(model.site_batches[j]),
                   'grid_kwh': pyo.value(block.E[j, 'grid']), 'wind_kwh': pyo.value(block.E[j, 'wind']),
                   'reused_water_m3': pyo.value(block.u[j]), 'fresh_water_m3': pyo.value(block.f[j]),
                   'wastewater_direct_m3': pyo.value(block.qDIR[j]),
                   'wastewater_to_recovery_m3': pyo.value(block.qREU[j]),
                   'discharge_m3': pyo.value(block.d[j])} for j in data.sites]).to_csv(
                      output_dir / 'joint_utilities.csv', index=False)
    pd.DataFrame([{'site': j, 'route': r, 'selected': pyo.value(block.bio_y[j, r]),
                   'kg_per_year': pyo.value(block.bio_q[j, r])}
                  for j in data.sites for r in data.biomass_routes]).to_csv(
                      output_dir / 'joint_biomass.csv', index=False)
    # Monetary outputs: retain the original total, electricity deduction and
    # adjusted supply cost so the accounting can be checked from the CSV.
    costs = {name: pyo.value(getattr(model, name)) for name in [
        'supply_original_cost_gbp', 'embedded_electricity_deduction_gbp', 'supply_cost_gbp',
        'is_plastics_cost', 'is_transport_cost', 'is_electricity_cost',
        'is_water_cost', 'is_biomass_cost', 'is_cost_gbp', 'total_cost_gbp', 'shortage_penalty_gbp']}
    pd.DataFrame([{'metric': key, 'GBP_per_year': value} for key, value in costs.items()]).to_csv(
        output_dir / 'joint_cost_summary.csv', index=False)
    # Attribute costs within the single joint solve. Electricity is assigned
    # to IS after removing its embedded charge from the supply-chain costs.
    total = costs['total_cost_gbp']
    contributions = pd.DataFrame([
        {'component': 'Supply chain (after electricity deduction)',
         'GBP_per_year': costs['supply_cost_gbp']},
        {'component': 'Industrial symbiosis (including electricity)',
         'GBP_per_year': costs['is_cost_gbp']},
        {'component': 'Total annual cost', 'GBP_per_year': total},
    ])
    contributions['percent_of_total'] = (
        contributions['GBP_per_year'] / total * 100.0 if total != 0 else float('nan'))
    contributions.to_csv(output_dir / 'joint_cost_contributions.csv', index=False)
    print(supply.to_string(index=False))
    print('\nAnnual cost contributions (GBP 2026/year):')
    print(contributions.to_string(index=False, formatters={
        'GBP_per_year': '{:,.2f}'.format,
        'percent_of_total': '{:.2f}%'.format,
    }))
    if costs['shortage_penalty_gbp'] != 0:
        print(f'Additional unmet-demand penalty: {costs["shortage_penalty_gbp"]:,.2f} GBP/year')
        print(f'Joint objective including penalty: {pyo.value(model.joint_objective):,.2f} GBP/year')
    print(f'Results: {output_dir}')


def main():
    """Parse command-line options, build the joint model, then export only or solve and report results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-only', action='store_true', help='Write the one GAMS input without solving.')
    parser.add_argument('--output-dir', type=Path, default=RESULT_DIR)
    parser.add_argument('--gams-executable', help='Path to gams.exe if it is not on PATH.')
    args = parser.parse_args()
    model, data = build_joint_model()
    if args.export_only:
        path, _ = export_gams(model, args.output_dir)
        print(f'GAMS input: {path}')
    else:
        solve_joint_model(model, args.output_dir, args.gams_executable)
        write_joint_results(model, data, args.output_dir)


# Command-line entry point; importing this module does not start a solve.
if __name__ == '__main__':
    main()
