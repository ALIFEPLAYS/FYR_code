"""Solve the fixed-capacity joint mAb supply chain and IS MILP without PB caps."""
import hashlib
import inspect
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyomo.environ as pyo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from project_paths import SEQUENTIAL_DIR, SUPERPRO_DIR

sys.path.insert(0, str(SEQUENTIAL_DIR))
from mAb_SupplyChain_model import mAb_SupplyChain_v1 as supply
from MILP.code import run_master_MILP as sym

RESULTS = Path(__file__).resolve().parent / 'results'
COST_INPUT = (SUPERPRO_DIR / 'data/processed'
              / 'manufacturing_footprint_inputs_and_costs_20260703.xlsx')
TOL = 1e-5


def check_close(actual, expected, label, atol=TOL):
    if not np.isclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f'{label}: {actual} differs from {expected}')


def solve(model):
    solver = pyo.SolverFactory('highs')
    solver.options.update({'mip_rel_gap': 1e-12, 'mip_abs_gap': 1e-6,
                           'mip_feasibility_tolerance': 1e-7})
    result = solver.solve(model)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(str(result.solver))
    return result


def validate(model):
    """Check all active Pyomo rows and all used/fixed variables in original units."""
    max_row = max_bound = max_integer = 0.
    constraints = list(model.component_data_objects(pyo.Constraint, active=True))
    for row in constraints:
        value = pyo.value(row.body)
        lower, upper = row.lb, row.ub
        max_row = max(max_row, (lower - value) if lower is not None else 0.,
                      (value - upper) if upper is not None else 0.)
    for var in model.component_data_objects(pyo.Var):
        if var.value is None:  # unused original reporting variables
            continue
        if var.lb is not None:
            max_bound = max(max_bound, var.lb - var.value)
        if var.ub is not None:
            max_bound = max(max_bound, var.value - var.ub)
        if var.is_integer():
            max_integer = max(max_integer, abs(var.value - round(var.value)))
    if max(max_row, max_bound, max_integer) > TOL:
        raise ValueError(f'Feasibility check failed: {max_row}, {max_bound}, {max_integer}')
    return dict(active_rows=len(constraints), max_row_violation=max_row,
                max_bound_violation=max_bound, max_integrality_error=max_integer)


def completed(model, site):
    return sum(model.B_g[1, site, 'L', line, 'mAb', 'S1'] for line in range(1, 5))


def original_supply(data):
    # Benchmark step 1. Solve for sites, production lines, batches and deliveries
    # using the original manufacturing and distribution costs.
    kwargs = supply.build_first_mab_input_data()
    kwargs.update(scenario_set=kwargs['scenarios'], sales_activation=False)
    builder = supply.create_mab_supply_chain_model
    model = builder(**{k: kwargs[k] for k in inspect.signature(builder).parameters})
    result = supply.solve_mab_model(model, solver_name='highs', tee=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(str(result.solver))
    validate(model)
    for var in model.Slack.values():
        check_close(var.value, 0, 'Upstream shortfall')
        var.fix(0)  # All comparisons must deliver the same demand.
    for j in data.sites:
        check_close(pyo.value(completed(model, j)), data.batches[j], 'Upstream baseline')
        check_close(pyo.value(model.Y_j[1, j]), data.site_open[j], 'Upstream site')
    # Use the original optimum's product-assigned line configuration. Enabled
    # (Z) counts alone can include idle lines. W, time and batch variables stay free.
    for group in (model.Y_j, model.Y_ja, model.Y_k, model.Z_u, model.Z_d,
                  model.Z_g, model.M_u, model.M_d, model.M_g):
        for var in group.values():
            var.fix(round(var.value))
    return model


def accounting(data):
    params = json.loads(sym.INPUT_FILES['params'].read_text())
    fx = params['currency_and_price_year_conversions']['multipliers_to_nominal_GBP_2026']
    # Read the original USD power cost per batch. Matching the kWh confirms
    # that the IS electricity bill covers the same manufacturing demand.
    items = pd.read_excel(COST_INPUT, sheet_name='itemised_costs')
    power = items[(items.cost_group == 'utility') & (items.item_name == 'Std Power')]
    check_close(power.amount_per_batch.sum(), data.electricity_kwh_per_batch,
                'SuperPro and IS electricity inventories')
    if power.empty or not power.cost_per_batch_usd.gt(0).all():
        raise ValueError('Missing or invalid SuperPro power cost')
    return fx['USD_2019_nominal'], float(power.cost_per_batch_usd.sum())


def attach_is(model, core, fx, power_usd):
    """Lift the original sparse IS rows into the original supply Pyomo model."""
    data = core.data
    b = model.symbiosis = pyo.Block()
    low = {core.index['low_batch', j] for j in data.sites}
    # Keep IS decisions for batches, material flows, truck trips, electricity and water.
    # The old yes/no flags only chose which site received the 205-batch allocation.
    cols = [i for i in range(len(core.var_keys)) if i not in low]
    b.flow = pyo.Var(cols, domain=lambda _, i: pyo.Integers if core.integrality[i] else pyo.Reals,
                     bounds=lambda _, i: (float(core.lower[i]),
                                          float(core.upper[i]) if np.isfinite(core.upper[i]) else None))
    # Verify exactly the obsolete allocation equalities before removing them.
    patterns = [(dict.fromkeys(low, 1.), 1.)]
    for j in data.sites:
        patterns.append(({core.index['B', j]: 1., core.index['low_batch', j]: data.batch_delta},
                         data.batch_high * data.site_open[j]))
    # Copy the original IS rules except for the four allocation equations above.
    # Total batches fix annual output. Polymer balances send all waste to eligible routes.
    # Truck payloads limit tonnes per trip. Shared recycler limits cap total annual intake.
    # Biomass goes to one destination per site. Electricity must cover each site's batches.
    # Water balances track reuse and discharge within each site's treatment capacity.
    b.physical = pyo.ConstraintList()
    matrix = core.A.tocsr()
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row:row + 2]
        coeffs = dict(zip(matrix.indices[start:stop], matrix.data[start:stop]))
        lb, ub = core.b_l[row], core.b_u[row]
        if low.intersection(coeffs):
            if lb != ub or (coeffs, lb) not in patterns:
                raise ValueError(f'Unrecognised low-allocation constraint at row {row}')
            patterns.remove((coeffs, lb))
            continue
        expr = pyo.quicksum(float(v) * b.flow[i] for i, v in coeffs.items())
        b.physical.add((lb if np.isfinite(lb) else None, expr, ub if np.isfinite(ub) else None))
    if patterns:
        raise ValueError('Missing expected legacy allocation constraints')
    # New rule for each site. Its finished batches also set its IS batch count,
    # so moving production moves the associated waste and utility demand.
    b.link = pyo.Constraint(data.sites, rule=lambda _, j:
                            b.flow[core.index['B', j]] == completed(model, j))

    def expression(name):
        quantity = core.expressions[name]
        if np.any(quantity.coeff[list(low)]):
            raise ValueError('Legacy low-batch flags have acquired objective coefficients')
        return quantity.const + pyo.quicksum(
            float(quantity.coeff[i]) * b.flow[i] for i in np.flatnonzero(quantity.coeff))

    # Full IS cost includes the selected electricity bill and recovered-material income.
    b.cash = pyo.Expression(expr=expression('cost'))
    b.gwp_kg = pyo.Expression(expr=expression('gwp'))
    # Remove the embedded SuperPro electricity charge once to avoid counting power twice.
    b.power_removed = pyo.Expression(expr=fx * power_usd * sum(completed(model, j) for j in data.sites))
    model.private_cost = pyo.Expression(expr=fx * model.expected_cost - b.power_removed + b.cash)
    shortage_penalty = fx * (model.objective1.expr - model.expected_cost)
    model.objective1.deactivate()
    # Minimise economic cost plus the carbon value. Divide kg CO2e by 1,000
    # to apply the price per tonne. The fixed zero shortfall gives zero penalty.
    model.joint_objective = pyo.Objective(expr=model.private_cost + shortage_penalty
                                          + data.lifecycle_carbon_value_gbp_per_t * b.gwp_kg / 1000.)


def score(model, core, result, fx, power_usd):
    data = core.data
    # Score both plans in 2026 GBP on the same basis and check that every batch
    # count agrees across the two models before reporting delivered product.
    upstream = pyo.value(model.expected_cost) * fx
    removed = fx * power_usd * sum(pyo.value(completed(model, j)) for j in data.sites)
    for j in data.sites:
        check_close(pyo.value(completed(model, j)), result.x[core.index['B', j]], 'Shared site output')
    delivered = pyo.value(model.expected_sales['mAb'])
    check_close(delivered, data.total_product_kg, 'Delivered product')
    check_close(sum(pyo.value(v) for v in model.Slack.values()), 0., 'Demand shortfall')
    private = upstream - removed + result.metrics['cost']
    carbon = data.lifecycle_carbon_value_gbp_per_t * result.metrics['gwp'] / 1000.
    return dict(case=result.case_id, batches=data.total_batches, delivered_kg=delivered,
                supply_chain_gbp=upstream, product_freight_gbp=fx * pyo.value(model.transport_cost['S1']),
                power_overlap_removed_gbp=removed, is_private_gbp=result.metrics['cost'],
                private_cost_gbp=private, gwp_tco2e=result.metrics['gwp'] / 1000.,
                carbon_value_gbp=carbon, objective_gbp=private + carbon,
                freshwater_m3=result.metrics['freshwater'], residual_t=result.metrics['residual'],
                human_health_daly=sym.evaluate_human_health_daly(core, result),
                ecosystems_species_year=sym.evaluate_ecosystem_quality(core, result))


def extract_joint(model, core):
    x = np.zeros(len(core.var_keys))
    for i, var in model.symbiosis.flow.items():
        x[i] = pyo.value(var)
    metrics = {k: expr.evaluate(x) for k, expr in core.expressions.items()}
    return sym.SolveResult('joint', True, 0, 'optimal', x, metrics,
                           pyo.value(model.joint_objective), 'joint_objective')


def main():
    data = sym.load_model_data()
    fx, power_usd = accounting(data)
    model = original_supply(data)
    # Benchmark step 2. The original IS run chooses waste routes and can swap
    # the low-batch site within the inherited high/high/low pattern.
    original = sym.build_linear_model(data)
    with patch.dict(os.environ, IS_MILP_BACKEND='scipy'):
        benchmark = sym.solve_model(original, 'two_stage', 'policy_cost')
    if not benchmark.success:
        raise RuntimeError(benchmark.message)
    audit = {'two_stage_is': sym.solution_validation(original, benchmark)}
    if max(audit['two_stage_is'].values()) > TOL:
        raise ValueError('Original IS benchmark failed feasibility validation')
    baseline = score(model, original, benchmark, fx, power_usd)
    # With no contracted tonnages in the central inputs, permit up to the
    # demonstrated baseline throughput at each shared recycler, across all sites.
    bench_routes, _, _ = sym.solution_decisions(original, benchmark)
    caps = dict(data.route_capacities_t)
    for route in data.routes:
        if route.startswith('mechanical_recycling') and route not in caps:
            caps[route] = float(bench_routes.loc[bench_routes.route == route, 'tonnes_per_year'].sum())
    core = sym.build_linear_model(replace(data, route_capacities_t=caps))
    attach_is(model, core, fx, power_usd)

    # Replay the original plan in the integrated equations, including cash and GWP.
    for i, var in model.symbiosis.flow.items():
        var.fix(benchmark.x[i])
    solve(model)
    audit['two_stage_replay'] = validate(model)
    check_close(pyo.value(model.joint_objective), baseline['objective_gbp'], 'Benchmark replay', 0.01)
    for var in model.symbiosis.flow.values():
        var.unfix()
    # Joint solve. Batches and IS choices can now move together within fixed capacity.
    result = solve(model)
    audit['joint'] = validate(model)
    joint = extract_joint(model, core)
    optimum = score(model, core, joint, fx, power_usd)
    check_close(optimum['objective_gbp'], joint.raw_objective, 'Independent objective reconciliation', 0.01)
    if optimum['objective_gbp'] > baseline['objective_gbp'] + 0.01:
        raise ValueError('Joint solve worse than its feasible benchmark')
    # The solver's lower bound is the best possible score it has proved.
    # Require the reported optimum to match that bound within one penny.
    lower = float(result.problem.lower_bound)
    check_close(optimum['objective_gbp'], lower, 'Solver global lower bound', 0.01)
    audit.update(joint_solver='HiGHS', termination=str(result.solver.termination_condition),
                 objective_lower_bound_gbp=lower, objective_gap_gbp=optimum['objective_gbp'] - lower,
                 shared_recycler_caps_t_per_year=caps, usd2019_to_gbp2026=fx,
                 removed_power_usd_per_batch=power_usd, carbon_gbp_per_t=data.lifecycle_carbon_value_gbp_per_t)

    inputs = [*sym.INPUT_FILES.values(), COST_INPUT, Path(supply.__file__), Path(sym.__file__)]
    audit['input_sha256'] = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                             hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    tables = {}
    for is_core, plan in [(original, benchmark), (core, joint)]:
        routes, sites, mix = sym.solution_decisions(is_core, plan)
        # The legacy binary no longer describes a decision in the joint model.
        routes = routes.drop(columns='is_low_batch_location')
        sites = sites.drop(columns='is_low_batch_location')
        for name, frame in [('polymer_routing', routes), ('site_decisions', sites),
                            ('route_mix', mix), ('is_breakdown', sym.component_breakdown(is_core, plan))]:
            tables.setdefault(name, []).append(frame)
    tables['comparison'] = [pd.DataFrame([baseline, optimum])]
    stages = [('upstream', model.M_u, model.B_u), ('purification', model.M_d, model.B_d),
              ('fill_finish', model.M_g, model.B_g)]
    lines = []
    for stage, assigned, batches in stages:
        for key, var in assigned.items():
            _, site, _, line, _ = key
            lines.append(dict(stage=stage, site=site, line=line, assigned=int(var.value),
                              joint_batches=round(batches[(*key, 'S1')].value)))
    tables['manufacturing_lines'] = [pd.DataFrame(lines)]

    # Publish only after all solves, checks and report construction succeed.
    RESULTS.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=RESULTS) as tmp:
        for name, frames in tables.items():
            pd.concat(frames, ignore_index=True).to_csv(Path(tmp) / f'{name}.csv', index=False)
        (Path(tmp) / 'validation.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        for path in Path(tmp).iterdir():
            path.replace(RESULTS / path.name)
    print(tables['comparison'][0].to_string(index=False))
    print(f'Validated results in {RESULTS}')


if __name__ == '__main__':
    main()
