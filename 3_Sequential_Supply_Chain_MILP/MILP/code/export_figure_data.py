#!/usr/bin/env python3
"""Rebuild the four journal-figure CSV directories from this submission's run outputs."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MILP = Path(__file__).resolve().parents[1]
ROOT = MILP.parent
sys.path.insert(0, str(ROOT.parent))
from project_paths import LCA_DIR, SUPERPRO_DIR

RESULTS = MILP / 'results'
OUT = RESULTS / 'data'
CENTRAL = 'OPT_COST_PLUS_CARBON_VALUE'
POLICY = 'OPT_COST_PLUS_CARBON_VALUE_INDEPENDENT_COMPARATOR'


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Required current-run source is absent or empty: {path}')
    return path


def csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def supply_batches() -> float:
    selected = os.environ.get('IS_SUPPLY_CHAIN_FILE')
    if not selected:
        raise RuntimeError('IS_SUPPLY_CHAIN_FILE is required for figure data export')
    frame = csv(Path(selected))
    if 'mab_batches_per_year' not in frame.columns:
        raise ValueError(f'Supply-chain file lacks mab_batches_per_year: {selected}')
    batches = pd.to_numeric(frame['mab_batches_per_year'], errors='raise')
    if batches.isna().any() or (batches < 0).any() or batches.sum() <= 0:
        raise ValueError(f'Invalid annual batch allocation: {selected}')
    return float(batches.sum())


def write(frame: pd.DataFrame, folder: str, name: str) -> None:
    target = OUT / folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        raise ValueError(f'Refusing empty figure table: {target}')
    frame.to_csv(target, index=False, float_format='%.10g')
    print(f'{target.relative_to(ROOT)}: {len(frame)} rows')


def copy_current(source: Path, folder: str) -> None:
    write(csv(source), folder, source.name)


def model_data():
    source = MILP / 'code' / 'run_master_MILP.py'
    spec = importlib.util.spec_from_file_location('journal_master_milp', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_model_data()


def figure_01() -> None:
    routing = RESULTS / 'ecosystem_routing_20261015'
    for name in ('routing_case_category_breakdown.csv', 'routing_case_decisions.csv',
                 'routing_case_route_results.csv', 'routing_case_selected_lcia_factors.csv',
                 'routing_case_summary.csv'):
        copy_current(routing / name, 'figure_01')
    routes = csv(RESULTS / 'ecosystem_master_20261015' / 'selected_case_polymer_routing.csv')
    routes = routes.loc[routes.case_id.eq(CENTRAL)]
    if routes.empty:
        raise ValueError('Central MILP polymer routing is missing')
    data = model_data()
    rows = []
    for polymer, group in routes.groupby('polymer', sort=True):
        amount = 0.0
        for row in group.itertuples(index=False):
            key = (str(row.polymer), str(row.route))
            amount -= float(row.tonnes_per_year) * data.value_capture[key] * data.recovered_value_gbp_per_t[key] * data.recovery_yield[key]
        rows.append({'polymer': polymer, 'annual_value_gbp': amount})
    write(pd.DataFrame(rows), 'figure_01', 'recyclate_return_by_polymer.csv')
    # SP_master_dataset.xlsx is supplied manually for the manuscript, as requested.


def figure_02() -> None:
    routes = csv(RESULTS / 'ecosystem_master_20261015' / 'selected_case_polymer_routing.csv')
    routes = routes.loc[routes.case_id.eq(CENTRAL)]
    if routes.empty:
        raise ValueError('Central MILP polymer routing is missing')
    fate = routes.groupby(['polymer', 'route'], as_index=False)[['tonnes_per_year', 'recovered_tonnes', 'residual_tonnes']].sum()
    names = {'landfill_all': 'Landfill', 'incineration_efw_heat_electricity': 'EfW',
             'mechanical_recycling_polyolefins': 'Mechanical recycling',
             'mechanical_recycling_tpe_tubing': 'TPE recycling',
             'chemical_recycling_strap_hypothetical': 'STRAP'}
    fate['route'] = fate['route'].map(lambda x: names.get(x, x))
    write(fate, 'figure_02', 'figure_02_polymer_route_fate_mass.csv')

    # Reuse the exact LCA inventory builder and source workbook; it applies the
    # configured composition, tubing and defect assumptions without new masses.
    sys.path.insert(0, str(LCA_DIR))
    from lca_pipeline.config import load_settings
    from lca_pipeline.sut_inventory import build_sut_polymer_inventory
    workbook = require_file(SUPERPRO_DIR / 'data/processed/SP_master_dataset.xlsx')
    sp = pd.read_excel(workbook, sheet_name='Superpro_SR_ICR_processed')
    overall = pd.read_excel(workbook, sheet_name='SUT_overall_plastic')
    composition = pd.read_excel(workbook, sheet_name='SUT_plastic_comp_matrix', header=1)
    detail, _ = build_sut_polymer_inventory(sp, overall, composition, load_settings(LCA_DIR))
    annual_batches = supply_batches()
    stage = detail.groupby(['process_area', 'polymer'], as_index=False)['polymer_kg_per_batch'].sum()
    stage = stage.rename(columns={'process_area': 'process_stage', 'polymer_kg_per_batch': 'mass_t_per_year'})
    stage['process_stage'] = stage['process_stage'].replace({'Fill finish': 'Fill-finish'})
    stage['mass_t_per_year'] *= annual_batches / 1000
    expected = routes['tonnes_per_year'].sum()
    if not np.isclose(stage.mass_t_per_year.sum(), expected, rtol=1e-7, atol=1e-5):
        raise ValueError(f'SUT polymer stage mass {stage.mass_t_per_year.sum()} t/y differs from central MILP {expected} t/y')
    write(stage, 'figure_02', 'figure_02_process_polymer_mass.csv')

    eligibility = pd.read_excel(require_file(MILP / 'inputs/polymer_route_eligibility_20260722.xlsx'), sheet_name='polymer_route_eligibility')
    eligibility = eligibility.loc[eligibility['eligible'].fillna(False).astype(bool) & eligibility['base_enabled'].fillna(False).astype(bool)]
    eligibility = eligibility[['polymer', 'lca_pathway', 'eligible', 'base_enabled']].rename(columns={'lca_pathway': 'route'})
    eligibility['route'] = eligibility['route'].map(lambda x: names.get(x, x))
    eligibility['available'] = True
    eligibility['source'] = 'polymer_route_eligibility'
    write(eligibility[['polymer', 'route', 'available', 'eligible', 'base_enabled', 'source']], 'figure_02', 'figure_02_route_availability.csv')


def figure_03() -> None:
    subprocess.run([sys.executable, str(MILP / 'code/build_figure_03_data.py')], cwd=ROOT, check=True)
    copy_current(RESULTS / 'ecosystem_routing_20261015/routing_case_summary.csv', 'figure_03')


def figure_04() -> None:
    tables = RESULTS / 'epsilon_constraint_pareto_independent_cost_gwp/tables'
    front = csv(tables / 'pareto_front.csv')
    front = front.loc[front.point_status.eq('optimal')].copy()
    if front.empty:
        raise ValueError('No optimal Pareto points')
    plot = pd.DataFrame({'gwp_kt': front.gwp_tco2e_per_year / 1000,
                         'cost_million': front.cost_million_gbp_per_year,
                         'pareto_point_id': front.pareto_point_id})
    write(plot, 'figure_04', 'pareto_front_plot.csv')
    indices = np.unique(np.linspace(1, len(plot) - 2, min(15, max(len(plot) - 2, 1)), dtype=int)) if len(plot) > 2 else np.arange(len(plot))
    write(plot.iloc[indices].reset_index(drop=True), 'figure_04', 'pareto_front_markers.csv')
    anchors = csv(tables / 'anchor_solutions.csv').set_index('case_id')
    for key in ('ANCHOR_GWP_LEX_COST', 'ANCHOR_COST_LEX_GWP', POLICY):
        if key not in anchors.index:
            raise ValueError(f'Missing Pareto anchor: {key}')
    key_rows = []
    for key, label in [('ANCHOR_GWP_LEX_COST', 'GWP minimum'), (POLICY, 'Central IS optimum'),
                       ('ANCHOR_COST_LEX_GWP', 'Cost minimum')]:
        row = anchors.loc[key]
        key_rows.append({'key': key, 'label': label, 'gwp_kt': row.gwp_tco2e_per_year / 1000,
                         'cost_million': row.cost_million_gbp_per_year})
    write(pd.DataFrame(key_rows), 'figure_04', 'pareto_key_points.csv')
    gwp, central, cost = (anchors.loc[k] for k in ('ANCHOR_GWP_LEX_COST', POLICY, 'ANCHOR_COST_LEX_GWP'))
    comparison = csv(tables / 'policy_front_comparison.csv').iloc[0]
    batches = supply_batches()
    vals = [
        ('central_gwp', central.gwp_tco2e_per_year, 'tCO2e/yr', 'independent central carbon-valued optimum'),
        ('central_cost', central.cost_gbp_per_year, 'GBP/yr', 'annual network cost at central optimum'),
        ('central_carbon_value', comparison.central_carbon_value_gbp_per_tco2e, 'GBP/tCO2e', 'central objective carbon value'),
        ('gwp_min_gwp', gwp.gwp_tco2e_per_year, 'tCO2e/yr', 'GWP-minimum anchor'),
        ('gwp_min_cost', gwp.cost_gbp_per_year, 'GBP/yr', 'annual cost at GWP-minimum anchor'),
        ('left_branch_abatement', central.gwp_tco2e_per_year - gwp.gwp_tco2e_per_year, 'tCO2e/yr', 'GWP reduction from central to GWP-minimum anchor'),
        ('left_branch_incremental_cost', gwp.cost_gbp_per_year - central.cost_gbp_per_year, 'GBP/yr', 'cost increase from central to GWP-minimum anchor'),
        ('cost_min_gwp', cost.gwp_tco2e_per_year, 'tCO2e/yr', 'cost-minimum anchor'),
        ('cost_min_cost', cost.cost_gbp_per_year, 'GBP/yr', 'annual cost at cost-minimum anchor'),
        ('right_branch_extra_gwp', cost.gwp_tco2e_per_year - central.gwp_tco2e_per_year, 'tCO2e/yr', 'GWP increase from central to cost-minimum anchor'),
        ('right_branch_cost_saving', central.cost_gbp_per_year - cost.cost_gbp_per_year, 'GBP/yr', 'cost saving from central to cost-minimum anchor'),
        ('annual_batches', batches, 'batches/yr', 'supply-chain production allocation'),
        ('central_gwp_per_batch', central.gwp_kgco2e_per_year / batches, 'kgCO2e/batch', 'central GWP per batch'),
    ]
    numbers = {k: v for k, v, _, _ in vals}
    vals += [
        ('left_branch_marginal_cost', numbers['left_branch_incremental_cost'] / numbers['left_branch_abatement'], 'GBP/tCO2e', 'incremental cost per tonne abated'),
        ('right_branch_saving_per_extra_gwp', numbers['right_branch_cost_saving'] / numbers['right_branch_extra_gwp'], 'GBP/tCO2e', 'private-cost saving per additional tonne'),
    ]
    matching = front.loc[front.pareto_point_id.eq(comparison.nearest_pareto_point_id)]
    if len(matching) == 1:
        row = matching.iloc[0]
        vals += [('coincident_epsilon_cap', row.epsilon_cap_target_tco2e, 'tCO2e/yr', 'sampled epsilon cap nearest central solution'),
                 ('coincident_epsilon_actual_gwp', row.gwp_tco2e_per_year, 'tCO2e/yr', 'realised GWP at nearest sampled epsilon cap'),
                 ('coincident_epsilon_slack', row.gwp_cap_slack_kgco2e / 1000, 'tCO2e/yr', 'unused GWP cap at nearest sampled point')]
    write(pd.DataFrame(vals, columns=['quantity', 'value', 'unit', 'interpretation']), 'figure_04', 'pareto_branch_summary.csv')


def main() -> None:
    figure_01()
    figure_02()
    figure_03()
    figure_04()
    expected = {
        'figure_01': 6, 'figure_02': 3, 'figure_03': 11, 'figure_04': 4,
    }
    for folder, count in expected.items():
        found = list((OUT / folder).glob('*.csv'))
        if len(found) != count:
            raise ValueError(f'{folder}: expected {count} CSVs, found {len(found)}')
    print(f'Figure data complete: {OUT}')


if __name__ == '__main__':
    main()
