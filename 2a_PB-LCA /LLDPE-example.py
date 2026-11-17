"""Export LCI and unnormalised PB-LCIA for 1 kg LLDPE landfill (ecoinvent 3.12).

Run from any directory: python hh.py
Optional: python hh.py --project PROJECT --database DATABASE

Requires an already imported ecoinvent 3.12 cut-off database, bw2data 4,
bw2calc 2, numpy, openpyxl and pbaesa. PB methods and nitrogen exchanges are
set up in a dedicated copy, leaving the source project unchanged. LLDPE uses
the generic polyethylene waste
treatment dataset, matching the project's existing landfill mapping.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
import textwrap
from importlib.metadata import distribution
import hashlib
import json
import os
from contextlib import redirect_stdout
from io import StringIO

import bw2calc as bc
import bw2data as bd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


LOCATIONS = ('GB', 'RoW', 'RER')
ACTIVITY_NAME = 'treatment of waste polyethylene, sanitary landfill'
REFERENCE_PRODUCT = 'waste polyethylene'
VERSION = re.compile(r'(?<!\d)3[._-]12(?![.\d])')
PB_SUFFIX = '__lldpe_pb'
PB_CATEGORIES = (
    'Climate Change', 'Ocean Acidification', 'Change in Biosphere Integrity',
    'Phosphorus Cycle', 'Atmospheric Aerosol Loading', 'Freshwater Use',
    'Stratospheric Ozone Depletion', 'Land-system Change', 'Nitrogen Cycle',
)


def select_database(project_name=None, database_name=None):
    """Select one existing 3.12 cut-off database; never substitute another version."""
    projects = [project.name for project in bd.projects]
    if project_name and project_name not in projects:
        raise ValueError(f'Brightway project does not exist: {project_name}')
    matches, available = [], []
    for project in ([project_name] if project_name else projects):
        if not project_name and PB_SUFFIX in project:
            continue  # Generated PB projects are not alternative source databases.
        bd.projects.set_current(project)
        for name, metadata in bd.databases.items():
            available.append(f'{project}: {name}')
            if database_name and name != database_name:
                continue
            description = f'{name} {metadata.get("version", "")} {metadata.get("system_model", "")}'
            if (VERSION.search(description) and 'biosphere' not in name.lower()
                    and ('cutoff' in description.lower() or 'cut-off' in description.lower())):
                matches.append((project, name))
    if len(matches) != 1:
        if not matches:
            reason = ('No ecoinvent 3.12 cut-off activity database was found. '
                      'Import that licensed database into Brightway first. '
                      'Installing pbaesa does not install ecoinvent. No results were generated.')
        else:
            reason = 'Multiple matching databases found; specify --project and --database.'
        raise ValueError(reason + '\nAvailable databases:\n  ' + '\n  '.join(available))
    project, database = matches[0]
    bd.projects.set_current(project)
    return project, database


def prepare_pb_project(source_project, database):
    """Register official pbaesa methods once in a dedicated, provenance-checked copy."""
    import pandas as pd
    import pbaesa

    package = distribution('pbaesa')
    factor_file = Path(package.locate_file('pbaesa/data/Characterization Factors_for_eco3101.xlsx'))
    table = pd.read_excel(factor_file, sheet_name='Characterization Factors').iloc[1:]
    if table['Code'].isna().any() or table['Code'].duplicated().any():
        raise ValueError('Missing or duplicate pbaesa factor-table codes.')
    biospheres = [name for name in bd.databases if 'biosphere' in name.lower() and VERSION.search(name)]
    if len(biospheres) != 1:
        raise ValueError(f'Expected one ecoinvent 3.12 biosphere: {biospheres}')
    biosphere = biospheres[0]
    flows = {str(flow['code']): flow for flow in bd.Database(biosphere)}
    missing = set(table['Code'].astype(str)) - set(flows)
    if missing:
        raise ValueError(f'{len(missing)} PB factor codes are missing from {biosphere}.')
    for category in PB_CATEGORIES[:-1]:
        if not np.isfinite(pd.to_numeric(table[category], errors='coerce')).all():
            raise ValueError(f'Invalid PB factors for {category}.')
    review = []
    for _, row in table.iterrows():
        flow = flows[str(row['Code'])]
        if str(row['Name']).strip() != flow.get('name', ''):
            review.append([str(row['Code']), str(row['Name']), flow.get('name'),
                           'Same identifier; name differs between factor table and ecoinvent 3.12'])
    provenance = {
        'source_project': source_project, 'database': database,
        'pbaesa_version': package.version,
        'factor_sha256': hashlib.sha256(factor_file.read_bytes()).hexdigest(),
        'source_database_metadata': json.loads(json.dumps(dict(bd.databases), default=str)),
    }
    target = source_project + PB_SUFFIX
    methods = [('Planetary Boundaries', category) for category in PB_CATEGORIES]
    if target in bd.projects:
        bd.projects.set_current(target)
        marker = Path(bd.projects.dir) / 'lldpe_pb_provenance.json'
        if not marker.exists() or json.loads(marker.read_text(encoding='utf-8')) != provenance:
            raise ValueError(f'PB project {target} has missing or changed provenance. '
                             'Use a fresh source-project copy; do not reuse stale results.')
    else:
        print(f'Copying Brightway project to {target}', flush=True)
        bd.projects.copy_project(target, switch=True)
        marker = Path(bd.projects.dir) / 'lldpe_pb_provenance.json'
        marker.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    if not all(method in bd.methods for method in methods):
        print('Registering PB methods and nitrogen inventory in the copied project', flush=True)
        previous_directory = Path.cwd()
        try:
            # The package checks cwd first. Force the verified bundled factor file.
            os.chdir(factor_file.parent)
            with redirect_stdout(StringIO()):
                pbaesa.create_pbaesa_methods(biosphere_db=bd.Database(biosphere))
        finally:
            os.chdir(previous_directory)
    if not all(method in bd.methods for method in methods):
        raise ValueError('PB setup did not register all nine expected methods.')
    return target, methods, review, provenance


def reference_amount(activity):
    """Use the actual reference-production sign for waste-treatment demand."""
    production = [exchange for exchange in activity.production()
                  if exchange.input.key == activity.key]
    if len(production) != 1:
        raise ValueError(f'Expected one self-reference production exchange: {activity.key}')
    amount = float(production[0]['amount'])
    if not math.isfinite(amount) or amount == 0:
        raise ValueError(f'Invalid reference production amount: {amount}')
    if activity.get('unit') not in ('kilogram', 'kg'):
        raise ValueError(f'Expected kilogram reference unit: {activity.key}')
    return amount


def flow_metadata(flow):
    categories = tuple(flow.get('categories', ()))
    return [flow.get('name', ''), categories[0] if categories else '',
            ' / '.join(categories[1:]), flow.get('unit', ''),
            flow.get('database', ''), str(flow.get('code', ''))]


def calculate_inventory(activity, methods=()):
    production = reference_amount(activity)
    demand = math.copysign(1.0, production)
    lca = bc.LCA({activity: demand})
    lca.lci()
    matrix = lca.inventory.tocsr(copy=True)
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all():
        raise ValueError(f'Non-finite inventory values: {activity.key}')
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    column = lca.dicts.activity[activity.id]
    direct_totals = matrix[:, column].toarray().ravel()
    # Include rows with contributions even if positive/negative entries cancel.
    recorded_rows = set(np.flatnonzero(np.diff(matrix.indptr)).tolist())
    raw_exchanges = list(activity.biosphere())
    # Preserve explicitly recorded zero-valued direct exchanges as well.
    for exchange in raw_exchanges:
        if exchange.input.id in lca.dicts.biosphere:
            recorded_rows.add(lca.dicts.biosphere[exchange.input.id])
    rows = []
    for row in sorted(recorded_rows):
        flow = bd.get_node(id=lca.dicts.biosphere.reversed[row])
        rows.append([*flow_metadata(flow), float(totals[row]), float(direct_totals[row])])
    rows.sort(key=lambda row: (row[1], row[2], row[0], row[5]))
    scale = float(lca.supply_array[column])
    direct = []
    for exchange in raw_exchanges:
        amount = float(exchange['amount'])
        if not math.isfinite(amount):
            raise ValueError('Non-finite direct biosphere exchange.')
        direct.append([*flow_metadata(exchange.input), amount,
                       amount / abs(production), amount * scale])
    pb_results, coverage = [], []
    for method in methods:
        lca.switch_method(method)
        lca.lcia()
        score = float(lca.score)
        if not math.isfinite(score):
            raise ValueError(f'Non-finite PB score for {method}')
        factors = np.asarray(lca.characterization_matrix.diagonal()).ravel()
        reconstructed = float(np.dot(factors, totals))
        if not math.isclose(score, reconstructed, rel_tol=1e-9, abs_tol=1e-30):
            raise ValueError(f'PB flow-sum check failed for {method}')
        explicit = set()
        for entry in bd.Method(method).load():
            key = entry[0]
            explicit.add(bd.get_node(id=int(key)).id if isinstance(key, (int, np.integer))
                         else bd.get_node(database=key[0], code=key[1]).id)
        absent = 0
        for row in sorted(recorded_rows):
            flow_id = lca.dicts.biosphere.reversed[row]
            if flow_id not in explicit:
                absent += 1
                flow = bd.get_node(id=flow_id)
                coverage.append([method[1], *flow_metadata(flow), float(totals[row]),
                                 'No explicit factor; not necessarily a missing impact pathway'])
        pb_results.append([method[1], score, bd.methods[method].get('unit', ''),
                           ' / '.join(method), len(recorded_rows) - absent, absent,
                           reconstructed, None])
    if methods:
        # An independent second solve verifies that the per-kg result scales linearly.
        lca.redo_lci({activity.id: 2 * demand})
        for method, result in zip(methods, pb_results):
            lca.switch_method(method)
            lca.lcia()
            if not math.isclose(float(lca.score), 2 * result[1], rel_tol=1e-8, abs_tol=1e-30):
                raise ValueError(f'PB 2 kg scaling check failed for {method}')
            result[-1] = 'Passed: 2 kg score = 2 x 1 kg score'
    return rows, direct, production, demand, pb_results, coverage


def add_table(workbook, name, headers, rows):
    sheet = workbook.create_sheet(name)
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    sheet.freeze_panes = 'B2'
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 42
    for cell in sheet[1]:
        cell.font = Font(name='Arial', size=10, bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='243B53')
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    for column in sheet.columns:
        width = min(65, max(18, max(len(str(cell.value or '')) for cell in column) + 2))
        sheet.column_dimensions[get_column_letter(column[0].column)].width = width
        for cell in column[1:]:
            cell.font = Font(name='Arial', size=10)
            cell.alignment = Alignment(vertical='top', wrap_text=isinstance(cell.value, str))
            if isinstance(cell.value, str):
                cell.data_type = 's'  # Dataset labels must remain literal text.
            elif isinstance(cell.value, float):
                cell.number_format = '0.000000000E+00'
    for row in sheet.iter_rows(min_row=2):
        lines = max((len(textwrap.wrap(str(cell.value or ''),
                     width=max(10, int(sheet.column_dimensions[cell.column_letter].width) - 3)))
                     for cell in row), default=1)
        sheet.row_dimensions[row[0].row].height = max(18, 14 * lines + 4)
    return sheet


def export_workbook(output, project, database, candidates, methods=(), review=(), provenance=None):
    workbook = Workbook()
    workbook.remove(workbook.active)
    summary = [
        ['Functional unit', 'Treatment of 1 kg LLDPE waste in sanitary landfill'],
        ['Database', database], ['Brightway project', project],
        ['Version and system model', 'ecoinvent 3.12 cut-off'],
        ['Material mapping', 'LLDPE represented by generic waste polyethylene landfill treatment'],
        ['Scope', 'Landfill treatment and its upstream supply chain; no separate virgin plastic production, supplier proxy or delivery stage added'],
        ['Geographies', 'Exact GB, RoW and RER matches only; no regional substitution'],
        ['Flow quantities', 'Signed elementary-flow quantities in their own units per 1 kg waste treated; no significance threshold'],
        ['LCIA', 'pbaesa Planetary Boundaries: raw characterised indicators per kg waste treated. No SOS division, allocation, normalisation or weighting.' if methods else 'None'],
        ['Direct quantities', 'Direct exchanges show source amount, normalised amount per kg treated and solved contribution including any feedback'],
        ['Inventory quantities', 'Total includes the direct column. Do not add direct quantities again. Zero totals may reflect cancellation'],
        ['Source', 'Local licensed ecoinvent 3.12 database; dataset and flow codes are preserved'],
    ]
    if methods:
        summary.extend([
            ['PB factor basis', 'pbaesa factors developed for ecoinvent 3.10.1, applied by matching identifiers to 3.12. Results are not a complete compatibility certification.'],
            ['PB indicators', 'Nine separate indicator results in method-specific units; no combined score. Novel entities are not implemented by pbaesa.'],
            ['Nitrogen', 'Official pbaesa N_supply exchanges added only in the copied project. Inventory now includes this additional flow when present upstream.'],
            ['Mapping review', f'{len(review)} flow-name differences listed in PB mapping review; unresolved semantic changes can affect interpretation.'],
            ['Coverage', 'PB coverage lists inventory flows without an explicit factor per method. Explicit zero factors are distinguished from absent factors. Counts are not impact-weighted coverage.'],
            ['PB software', provenance['pbaesa_version']],
            ['PB factor SHA256', provenance['factor_sha256']],
            ['Source project', provenance['source_project']],
            ['PB method source', 'https://github.com/RWTH-LTT/pbaesa'],
        ])
    add_table(workbook, 'Summary', ['Item', 'Details'], summary)
    statuses, inventories, directs, indicators, coverage = [], [], [], [], []
    for location in LOCATIONS:
        matches = [activity for activity in candidates if activity.get('location') == location]
        if len(matches) != 1:
            status = 'Unavailable: no exact dataset' if not matches else 'Ambiguous: multiple datasets; review Candidates'
            statuses.append([location, status, None, None, None, None])
            for method in methods:
                indicators.append([location, method[1], None, bd.methods[method].get('unit', ''),
                                   status, ' / '.join(method), None, None, None, None])
            continue
        activity = matches[0]
        print(f'Calculating {location}: {activity.key}', flush=True)
        rows, direct, production, demand, pb_rows, missing_factors = calculate_inventory(activity, methods)
        code = str(activity['code'])
        inventories.extend([[location, code, *row] for row in rows])
        directs.extend([[location, code, *row] for row in direct])
        statuses.append([location, 'Calculated', code, production, demand, len(rows)])
        indicators.extend([[location, r[0], r[1], r[2], 'Calculated', *r[3:]] for r in pb_rows])
        coverage.extend([[location, *row] for row in missing_factors])
    add_table(workbook, 'Locations',
              ['Location', 'Status', 'Activity code', 'Reference production amount',
               'Signed demand (kg)', 'Inventory flow rows'], statuses)
    if methods:
        add_table(workbook, 'PB indicators',
                  ['Location', 'PB indicator', 'Result per kg waste treated', 'Indicator unit',
                   'Status', 'Method', 'Flow rows with explicit factor', 'Flow rows without factor',
                   'Reconstructed from flow totals', 'Scaling check'], indicators)
        add_table(workbook, 'PB coverage',
                  ['Location', 'PB indicator', 'Elementary flow', 'Compartment', 'Subcompartment',
                   'Flow unit', 'Flow database', 'Flow code', 'Amount per kg waste treated', 'Note'], coverage)
        add_table(workbook, 'PB mapping review',
                  ['Flow code', 'Factor-table name', 'Ecoinvent 3.12 name', 'Review'], review)
    base_headers = ['Location', 'Activity code', 'Elementary flow', 'Compartment',
                    'Subcompartment', 'Flow unit', 'Flow database', 'Flow code']
    add_table(workbook, 'Elementary flows', base_headers +
              ['Total per kg waste treated', 'Direct contribution within total'], inventories)
    add_table(workbook, 'Direct exchanges', base_headers +
              ['Source exchange amount', 'Direct per kg waste treated', 'Contribution in solved LCI'], directs)
    add_table(workbook, 'Candidates',
              ['Location', 'Activity name', 'Reference product', 'Unit', 'Database', 'Activity code'],
              [[a.get('location'), a.get('name'), a.get('reference product'), a.get('unit'),
                a.get('database'), str(a.get('code'))] for a in candidates])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Complete the zip before replacing the delivered file. If Excel has it open,
    # retain the computed workbook so no second LCA calculation is needed.
    pending = output.with_name('.' + output.name + '.pending')
    workbook.save(pending)
    try:
        pending.replace(output)
    except PermissionError as error:
        raise PermissionError(f'Close {output.name} in Excel, then run python hh.py --finish-save. '
                              f'Completed results are preserved at {pending}') from error
    for location, status, *_ in statuses:
        print(f'{location}: {status}')
    print(f'Saved: {output}')
    return all(row[1] == 'Calculated' for row in statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', help='Existing Brightway project containing ecoinvent 3.12.')
    parser.add_argument('--database', help='Existing ecoinvent 3.12 cut-off activity database.')
    parser.add_argument('--finish-save', action='store_true',
                        help='Save an already calculated pending workbook after closing Excel.')
    args = parser.parse_args()
    output = Path(__file__).resolve().with_name('LLDPEtest.xlsx')
    if args.finish_save:
        pending = output.with_name('.' + output.name + '.pending')
        try:
            if not pending.is_file():
                raise ValueError('No pending workbook exists. Run python hh.py to calculate results.')
            pending.replace(output)
        except (ValueError, PermissionError) as error:
            print(f'Cannot finish save: {error}', file=sys.stderr)
            return 1
        print(f'Saved: {output}')
        return 0
    original_project = bd.projects.current
    try:
        project, database = select_database(args.project, args.database)
        project, methods, review, provenance = prepare_pb_project(project, database)
        candidates = [a for a in bd.Database(database)
                      if a.get('name') == ACTIVITY_NAME
                      and a.get('reference product') == REFERENCE_PRODUCT]
        candidates.sort(key=lambda a: (str(a.get('location')), str(a.get('code'))))
        complete = export_workbook(output,
                                   project, database, candidates, methods, review, provenance)
        return 0 if complete else 2
    except (ValueError, PermissionError) as error:
        print(f'Cannot complete export: {error}', file=sys.stderr)
        return 1
    finally:
        if bd.projects.current != original_project:
            bd.projects.set_current(original_project)


if __name__ == '__main__':
    raise SystemExit(main())
