"""Calculate cumulative PB impacts for every row of the master route workbook.

Run: python PB_LCIA.py
Requires the same Python/Brightway/pbaesa environment as hh.py. The default
source is ../2_LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx.
Output: PB_LCIA_perkg.xlsx beside this script (native route units are retained).

The master export omits activity demand quantities. Recover each single-activity
stage's signed demand from its GWP factor, independently verify it against its
fossil-resource factor, then calculate PB scores from the actual LCI. This is
NOT a conversion of GWP into PB impacts. A changed or incompatible source fails
validation instead of silently generating different route factors.

The GWP-only SUT fabrication adder has no elementary-flow inventory and is
explicitly excluded. 'Other' production uses the master's mass-weighted mix.
PB setup is delegated to hh.prepare_pb_project, which protects the source project.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import bw2calc as bc
import bw2data as bd
import numpy as np
from openpyxl import load_workbook

from hh import PB_CATEGORIES, prepare_pb_project, select_database


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / '2_LCA/outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx'
ROUTE_FIELDS = ('source_module', 'stream_category', 'optimisation_flow',
                'material_or_service', 'pathway', 'optimisation_quantity_unit')
STAGE_FIELDS = ROUTE_FIELDS + ('life_cycle_stage', 'eol_sink', 'database', 'code')
MANUAL_CODE = 'SUT_MANUFACTURING_GWP_ADDER'


# Verified ecoinvent 3.12 cut-off mappings from the SuperPro flow register.
SUPPLEMENTAL_ACTIVITIES = (
    ('Centrifuge cleaning effluent',
     'treatment of wastewater, average, wastewater treatment', 'Europe without Switzerland',
     'cubic meter', '853722e89539c9b5fa11ddb5a151683e'),
    ('NaOH, active mass',
     'market for sodium hydroxide, without water, in 50% solution state', 'RER',
     'kilogram', '7525bc2877b280a42663999cd025cf0c'),
    ('H₃PO₄, active mass',
     'market for phosphoric acid, industrial grade, without water, in 85% solution state', 'GLO',
     'kilogram', 'f84cfa144416d3c33a7229be84b15702'),
    ('Acetic acid',
     'market for acetic acid', 'GLO',
     'kilogram', '89ddfe4d99001c74fe5fdba00443ef30'),
    ('Dry sodium chloride',
     'market for sodium chloride, powder', 'GLO',
     'kilogram', '8e9e8b96777e25cc19aa7c310f7f3104'),
    ('Na₂HPO₄',
     'market for disodium phosphate', 'GLO',
     'kilogram', '00a4b7a531491d2211a784cea7ae2651'),
    ('Na₃PO₄',
     'market for trisodium phosphate', 'GLO',
     'kilogram', 'ba595ef81d4278e5b9fb6d55ea0db436'),
    ('Brine chemical make-up',
     'market for calcium chloride', 'RER',
     'kilogram', 'eb63804716531cd4717ffeaad09894ba'),
    ('Brine cooling duty, temperature proxy to review',
     'market for cooling energy, at -25 °C', 'GLO',
     'megajoule', '852575286dfeaacb8e1a226cc5389fbc'),
    ('Sterile air, compression only if pressure fits',
     'market for compressed air, 600 kPa gauge', 'RER',
     'cubic meter', '551efd52b08e7d48266202f80b3fb284'),
    ('Labware, only if polypropylene is confirmed',
     'market for polypropylene, granulate', 'RER',
     'kilogram', 'b0abb5042456d17f58a95600bd9eebd7'),
    ('Labware, only if polystyrene is confirmed',
     'market for polystyrene, general purpose', 'RER',
     'kilogram', 'efa6057bb5875d23a36a903eb65e23a8'),
)
DIRECT_RELEASES = (
    ('Phosphorus', ('water', 'surface water'), 'b2631209-8374-431e-b7d5-56c96c6b6d79'),
    ('Phosphate', ('water', 'surface water'), '1727b41d-377e-43cd-bc01-9eaba946eccb'),
    ('Ammonium', ('water', 'surface water'), '13331e67-6006-48c4-bdb4-340c12010036'),
    ('Nitrate', ('water', 'surface water'), '7ce56135-2ca5-4fba-ad52-d62a34bfeb35'),
    ('COD, Chemical Oxygen Demand', ('water', 'surface water'), 'fc0b5c85-3b49-42c2-a3fd-db7e57b696e3'),
    ('BOD5, Biological Oxygen Demand', ('water', 'surface water'), '70d467b6-115e-43c5-add2-441de9411348'),
    ('Carbon dioxide, fossil', ('air',), '349b29d1-3e58-4c66-98b9-9d1a076efd2e'),
    ('Carbon dioxide, non-fossil', ('air',), 'eba59fd6-f37e-41dc-9ca3-c7ea22d602c7'),
)


def identity(row, fields=ROUTE_FIELDS):
    return tuple(row.get(field) for field in fields)


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'Non-finite quantity: {value}')
    return number


def close(actual, expected, label, rel_tol=2e-7):
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=1e-25):
        raise ValueError(f'{label}: calculated {actual:.12g}, source {expected:.12g}')


def read_source(path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    def records(name):
        values = iter(workbook[name].values)
        headers = next(values)
        return [dict(zip(headers, row)) for row in values if any(v is not None for v in row)]
    try:
        routes, stages = records('route_factor_matrix'), records('lca_stage_breakdown')
    finally:
        workbook.close()
    keys = [identity(row) for row in routes]
    if not routes or len(set(keys)) != len(keys):
        raise ValueError('Missing or duplicate route identities in master workbook.')
    if set(identity(row) for row in stages) != set(keys):
        raise ValueError('Route matrix and stage breakdown contain different routes.')
    return routes, stages


class Scores:
    """Reuse one factorised technosphere; check scores against elementary totals."""
    def __init__(self, activities, methods):
        self.methods = methods
        self.lca = bc.LCA({activity: 1.0 for activity in activities})
        self.lca.lci(factorize=True)

    def calculate(self, demand):
        self.lca.redo_lci(demand)
        totals = np.asarray(self.lca.inventory.sum(axis=1)).ravel()
        if not np.isfinite(totals).all():
            raise ValueError('Non-finite elementary-flow inventory.')
        scores = []
        for method in self.methods:
            self.lca.switch_method(method)
            self.lca.lcia()
            score = finite(self.lca.score)
            rebuilt = float(np.dot(self.lca.characterization_matrix.diagonal(), totals))
            close(rebuilt, score, f'Elementary-flow sum for {method}')
            scores.append(score)
        return np.array(scores)


def recover_demands(routes, stages, database):
    """Recover and independently validate quantities missing from the export."""
    # These are endpoint conversions of the same GWP-only fabrication estimate,
    # not additional physical activities or elementary-flow inventories.
    stages = [r for r in stages if not (
        r['database'] == 'foreground assumption' and r['code'] in {
            'SUT_MANUFACTURING_CLIMATE_ECOSYSTEM_ADDER',
            'SUT_MANUFACTURING_CLIMATE_HEALTH_ADDER'})]
    grouped = defaultdict(dict)
    for row in stages:
        key, category = identity(row, STAGE_FIELDS), row['impact_category']
        if category in grouped[key]:
            raise ValueError(f'Duplicate stage/category: {key}, {category}')
        grouped[key][category] = row
    for group in grouped.values():
        if 'gwp' not in group:
            raise ValueError('A stage has no GWP anchor; cannot reconstruct its demand.')
    activity_keys = {(r['database'], r['code']) for r in stages if r['database'] == database}
    unsupported = {(r['database'], r['code']) for r in stages
                   if r['database'] != database
                   and (r['database'], r['code']) not in
                   {('derived', 'derived'), ('foreground assumption', MANUAL_CODE)}}
    if unsupported:
        raise ValueError(f'Unsupported stage mappings: {unsupported}')
    activities = {key: bd.get_node(database=key[0], code=key[1]) for key in activity_keys}
    method_fields = ('gwp_method', 'fossil_resource_method')
    methods = []
    for field in method_fields:
        labels = {row[field] for row in routes}
        if len(labels) != 1:
            raise ValueError(f'Inconsistent source method: {field}')
        method = tuple(labels.pop().split(' | '))
        if method not in bd.methods:
            raise ValueError(f'Missing validation method: {method}')
        methods.append(method)
    engine = Scores(list(activities.values()), methods)
    unit_scores = {}
    for i, (key, activity) in enumerate(sorted(activities.items()), 1):
        print(f'Validating activity {i}/{len(activities)}: {activity["name"]}', flush=True)
        unit_scores[key] = engine.calculate({activity.id: 1.0})

    stage_demands = {}
    for key, group in grouped.items():
        row = group['gwp']
        if row['database'] != database:
            continue
        activity_key = (row['database'], row['code'])
        if row['unit'] != activities[activity_key].get('unit'):
            raise ValueError(f'Activity unit changed: {activity_key}')
        if 'fossil_resource_use' not in group:
            raise ValueError(f'No independent fossil-resource check for {key}')
        expected = np.array([finite(group[c]['impact_factor'])
                             for c in ('gwp', 'fossil_resource_use')])
        unit = unit_scores[activity_key]
        anchor = 0 if unit[0] != 0 else 1
        if unit[anchor] == 0:
            raise ValueError(f'Cannot identify demand from zero validation scores: {key}')
        amount = expected[anchor] / unit[anchor]
        for index in range(2):
            close(amount * unit[index], expected[index], f'Signed demand validation: {key}')
        stage_demands[key] = {activity_key: amount}

    # Reconstruct the same known-polymer mass mix used by plastics_lca.py.
    polymer_rows = {}
    for route in routes:
        if route['source_module'] == 'sut_plastics' and route['material_or_service'] != 'Other':
            material = route['material_or_service']
            if material in polymer_rows:
                close(finite(route['represented_quantity_per_batch']),
                      finite(polymer_rows[material]['represented_quantity_per_batch']),
                      f'Polymer mass consistency: {material}')
            polymer_rows[material] = route
    total_mass = sum(finite(r['represented_quantity_per_batch']) for r in polymer_rows.values())
    mix = defaultdict(float)
    for material, route in polymer_rows.items():
        matches = [k for k, g in grouped.items()
                   if identity(g['gwp']) == identity(route)
                   and g['gwp']['life_cycle_stage'] == 'virgin_polymer_production']
        if len(matches) != 1 or total_mass <= 0:
            raise ValueError(f'Ambiguous polymer production mix for {material}')
        weight = finite(route['represented_quantity_per_batch']) / total_mass
        for activity_key, amount in stage_demands[matches[0]].items():
            mix[activity_key] += weight * amount
    for key, group in grouped.items():
        row = group['gwp']
        if row['database'] == 'derived':
            if row['material_or_service'] != 'Other' or row['life_cycle_stage'] != 'virgin_polymer_production':
                raise ValueError(f'Unsupported derived stage: {key}')
            for index, category in enumerate(('gwp', 'fossil_resource_use')):
                value = sum(amount * unit_scores[a][index] for a, amount in mix.items())
                close(value, finite(group[category]['impact_factor']), f'Other polymer mix: {category}')
            stage_demands[key] = dict(mix)

    demands, exclusions = {}, {}
    for route in routes:
        key = identity(route)
        members = [(k, g) for k, g in grouped.items() if identity(g['gwp']) == key]
        for category in ('gwp', 'fossil_resource_use'):
            summed = sum(finite(g[category]['impact_factor']) for _, g in members if category in g)
            close(summed, finite(route[f'net_{category}_factor_per_optimisation_unit']),
                  f'Master route reconciliation: {key}, {category}')
        demand = defaultdict(float)
        omitted = []
        for stage_key, group in members:
            row = group['gwp']
            if row['code'] == MANUAL_CODE:
                omitted.append(finite(row['impact_factor']))
                continue
            for activity_key, amount in stage_demands[stage_key].items():
                demand[activity_key] += amount
        if not demand:
            raise ValueError(f'No inventory-backed stages for {key}')
        demands[key] = dict(demand)
        exclusions[key] = sum(omitted) if omitted else None
    return demands, exclusions, activity_keys


def supplemental_rows(engine, database, methods):
    rows = []
    for label, name, location, unit, code in SUPPLEMENTAL_ACTIVITIES:
        activity = bd.get_node(database=database, code=code)
        if (activity['name'], activity['location'], activity['unit']) != (name, location, unit):
            raise ValueError(f'Activity mapping changed: {activity.key}')
        production = finite(activity.rp_exchange()['amount'])
        if production == 0:
            raise ValueError(f'Zero reference production: {activity.key}')
        demand = math.copysign(1.0, production)
        scores = engine.calculate({activity.id: demand})
        rows.append([label, name, unit, *scores.tolist(), f'Unit activity inventory ({location})',
                     'additional activity', f'activity__{code}', None, None])

    # Characterise one kg released directly into its receiving compartment.
    # The diagonal is the same one used by Scores.calculate for cumulative inventories.
    lca = engine.lca
    releases = []
    for name, compartment, code in DIRECT_RELEASES:
        flow = bd.get_node(database='ecoinvent-3.12-biosphere', code=code)
        if (flow['name'], tuple(flow['categories']), flow['unit']) != (name, compartment, 'kilogram'):
            raise ValueError(f'Elementary-flow mapping changed: {flow.key}')
        releases.append((flow, []))
    for method in methods:
        lca.switch_method(method)
        diagonal = lca.characterization_matrix.diagonal()
        for flow, scores in releases:
            scores.append(finite(diagonal[lca.dicts.biosphere[flow.id]]))
    for flow, scores in releases:
        rows.append([flow['name'], ' / '.join(flow['categories']), flow['unit'], *scores,
                     'Direct release into receiving compartment', 'elementary exchange',
                     f"release__{flow['code']}", None, None])
    return rows


# Embedded exporter keeps the requested Python file sufficient for regeneration.
# Uses the installed artifact runtime when available; ordinary Python installations
# without it use the same openpyxl table writer already used by hh.py.
EXPORT_JS = r'''
import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const payload = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const wb = Workbook.create();
for (const [index, spec] of payload.sheets.entries()) {
  const sh = wb.worksheets.add(spec.name);
  const rows = [spec.headers, ...spec.rows];
  sh.getRangeByIndexes(0, 0, rows.length, spec.headers.length).values = rows;
  const used = sh.getUsedRange();
  used.format.font = {name:'Arial', size:10, color:'#243B53'};
  used.format.verticalAlignment = 'center';
  used.format.rowHeight = index === 0 ? 46 : 42;
  used.format.wrapText = true;
  for (let j=0;j<spec.headers.length;j++) {
    sh.getRangeByIndexes(0,j,rows.length,1).format.columnWidth = spec.widths[j];
  }
  sh.getRangeByIndexes(0,0,1,spec.headers.length).format = {
    fill:'#243B53', font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},
    wrapText:true, horizontalAlignment:'center', verticalAlignment:'center', rowHeight:66
  };
  sh.showGridLines = false;
  sh.freezePanes.freezeRows(1);
  if (index === 0) {
    sh.freezePanes.freezeColumns(2);
    sh.getRange(`D2:L${rows.length}`).setNumberFormat('0.000000000E+00');
    sh.getRange(`Q2:Q${rows.length}`).setNumberFormat('0.00');
    sh.tables.add(`A1:Q${rows.length}`,true,'PBRouteFactors');
    sh.getRange(`M2:M${rows.length}`).conditionalFormats.add('containsText', {
      text:'Excluded',format:{fill:'#FFF2CC'}
    });
  }
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'PB impacts!A1:F5',tableMaxRows:5,tableMaxCols:6})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NUM!|#NAME\\?',
  options:{useRegex:true,maxResults:10}})).ndjson);
for (const [sheetName,range,file] of [
  ['PB impacts','A1:F8','pb_preview.png'],
  ['PB impacts','G1:M8','pb_indicators_preview.png'],
  ['Method notes','A1:B15','pb_notes_preview.png']]) {
  const preview = await wb.render({sheetName,range,scale:1.4,format:'png'});
  await fs.writeFile(`${process.argv[4]}/${file}`,new Uint8Array(await preview.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(process.argv[3]);
'''


def export_workbook(output, sheets):
    runtime = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
    node = runtime / 'bin/node.exe'
    modules = runtime / 'node_modules'
    pending = output.with_name('.' + output.name + '.pending.xlsx')
    output.parent.mkdir(parents=True, exist_ok=True)
    if node.exists() and (modules / '@oai/artifact-tool').exists():
        # Keep previews in the OS temp directory, not among the deliverables.
        work = Path(tempfile.mkdtemp(prefix='pb_lcia_'))
        junction = work / 'node_modules'
        quoted = lambda path: "'" + str(path).replace("'", "''") + "'"
        subprocess.run(['powershell', '-NoProfile', '-Command',
                        '$ErrorActionPreference = \'Stop\'; New-Item -ItemType Junction -Path '
                        + quoted(junction) + ' -Target ' + quoted(modules) + ' | Out-Null'], check=True)
        (work / 'export.mjs').write_text(EXPORT_JS, encoding='utf-8')
        (work / 'payload.json').write_text(json.dumps({'sheets': sheets}), encoding='utf-8')
        rendered = work / 'PB_LCIA_perkg.xlsx'
        subprocess.run([str(node), str(work / 'export.mjs'), str(work / 'payload.json'),
                        str(rendered), str(work)], check=True)
        shutil.copyfile(rendered, pending)
        print(f'Workbook previews: {work}', flush=True)
    else:
        from openpyxl import Workbook
        from hh import add_table
        workbook = Workbook()
        workbook.remove(workbook.active)
        for spec in sheets:
            add_table(workbook, spec['name'], spec['headers'], spec['rows'])
        workbook.save(pending)
    check = load_workbook(pending, read_only=True, data_only=True)
    try:
        for spec in sheets:
            actual = list(check[spec['name']].values)
            expected = [spec['headers'], *spec['rows']]
            if len(actual) != len(expected):
                raise ValueError('Export row count differs from calculated results.')
            for arow, erow in zip(actual, expected):
                for a, e in zip(arow, erow):
                    if isinstance(e, (int, float)):
                        close(finite(a), float(e), 'Saved workbook numerical check')
                    elif a != e:
                        raise ValueError(f'Export value changed: {a!r} versus {e!r}')
    finally:
        check.close()
    pending.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=HERE / 'PB_LCIA_perkg.xlsx')
    args = parser.parse_args()
    source = args.source.resolve()
    routes, stages = read_source(source)
    projects = {r['brightway_project'] for r in routes}
    if len(projects) != 1:
        raise ValueError('Master workbook must use one Brightway source project.')
    original = bd.projects.current
    try:
        project, database = select_database(projects.pop())
        demands, excluded, activity_keys = recover_demands(routes, stages, database)
        print('Preparing verified PB methods in the copied project', flush=True)
        pb_project, methods, review, provenance = prepare_pb_project(project, database)
        activities = {key: bd.get_node(database=key[0], code=key[1]) for key in activity_keys}
        engine = Scores(list(activities.values()), methods)
        unit_scores = {key: engine.calculate({act.id: 1.0}) for key, act in activities.items()}
        rows = []
        for index, route in enumerate(routes, 1):
            key = identity(route)
            demand = {activities[a].id: q for a, q in demands[key].items()}
            scores = engine.calculate(demand)
            summed = sum(q * unit_scores[a] for a, q in demands[key].items())
            doubled = engine.calculate({a: 2*q for a, q in demand.items()})
            for category, score, subtotal, double in zip(PB_CATEGORIES, scores, summed, doubled):
                # Independent sparse solves can differ slightly for tiny PB scores.
                close(float(subtotal), float(score), f'Stage sum: {key}, {category}', rel_tol=1e-5)
                close(float(double), 2*float(score), f'Two-unit scaling: {key}, {category}', rel_tol=1e-5)
            scope = ('Excluded fabrication proxy: no elementary-flow inventory'
                     if excluded[key] is not None else 'All source stages represented')
            rows.append([route['material_or_service'], route['pathway'],
                         route['optimisation_quantity_unit'], *scores.tolist(), scope,
                         route['source_module'], route['optimisation_flow'], index + 1, excluded[key]])
            print(f'Calculated {index}/{len(routes)}: {route["material_or_service"]} / {route["pathway"]}', flush=True)
        rows.extend(supplemental_rows(engine, database, methods))
        notes = [
            ['Basis', 'Master routes are followed by additional activity and direct-release factors. Results are per ONE unit in column C, including kg, m3, kWh and MJ as applicable.'],
            ['Additional activities', 'One native unit of the named provider. Waste treatment uses its reference production sign. Chemical quantities use active mass. Cooling, compression and polymer activities are conditional proxies.'],
            ['Direct releases', 'One kg emitted to the stated compartment. A zero can indicate an absent or zero method factor. Use post-treatment releases and account for emissions already in the treatment inventory.'],
            ['Unresolved supplies', 'HEPES, TRIS base/HCl, polysorbate 80, media, inoculum, Protein A and HIC resins, and diafiltration membranes require supplier inventories or justified proxies.'],
            ['Calculation', 'Sum of characterised elementary flows across each route and its upstream supply chain, including signed avoided-product credits. No normalisation, allocation or weighting.'],
            ['Scope of plastic routes', 'Includes mapped virgin production, cleaning, end-of-life and credits. Excludes the manual SUT fabrication proxy for every PB because its elementary-flow inventory is unavailable. These are partial route totals.'],
            ['Excluded proxy column', 'Column Q records the excluded source GWP proxy in kg CO2-equivalent per route unit. It is not a PB impact and is not added to any PB column.'],
            ['Demand reconstruction', 'Signed activity amounts recovered from source stage GWP divided by the same activity/method unit score; independently checked against fossil-resource factors before PB calculation from the actual inventory.'],
            ['Other plastics', 'Virgin production is the mass-weighted mix of the known polymers using source represented quantities, counted once per material. Verified against both source indicators.'],
            ['Nitrogen', 'Official pbaesa N_supply additions in the copied project only. Its fertiliser-name selection controls coverage; zero is not proof that no nitrogen pathway exists.'],
            ['Compatibility', 'pbaesa factors are based on ecoinvent 3.10.1 and matched by identifier to 3.12. This is not a full semantic compatibility certification.'],
            ['Mapping review', f'{len(review)} factor-flow name differences were identified by hh.prepare_pb_project.'],
            ['Novel entities', 'Not implemented by pbaesa. No numerical result is invented; the nine implemented PB indicators are shown.'],
            ['Validation', f'{len(routes)} routes: source GWP/fossil stage totals reconciled; demand factors cross-checked; PB stage sums and elementary-flow sums verified; two-unit scaling verified.'],
            ['Numerical tolerances', 'Source-factor checks: relative 2e-7. Independent PB stage-sum and scaling checks: relative 1e-5 for sparse-solver precision. Absolute floor: 1e-25.'],
            ['Source workbook', str(source)],
            ['Source SHA256', hashlib.sha256(source.read_bytes()).hexdigest()],
            ['Brightway project', pb_project],
            ['pbaesa version', provenance['pbaesa_version']],
            ['Factor SHA256', provenance['factor_sha256']],
            ['Method source', 'https://github.com/RWTH-LTT/pbaesa'],
        ]
        for method in methods:
            notes.append([method[1], bd.methods[method].get('unit', '') + ' per route unit (column C)'])
        headers = ['Material or service', 'Environmental pathway', 'Per-unit basis',
                   *PB_CATEGORIES, 'Inventory scope', 'Source module', 'Optimisation flow',
                   'Master Excel row', 'Excluded fabrication GWP (kg CO2e/unit)']
        sheets = [dict(name='PB impacts', headers=headers, rows=rows,
                       widths=[24, 49, 18, *([23]*9), 52, 27, 34, 16, 26]),
                  dict(name='Method notes', headers=['Item', 'Details'], rows=notes, widths=[28, 115])]
        export_workbook(args.output.resolve(), sheets)
        print(f'Saved and verified {len(rows)} factors x {len(methods)} PB indicators: {args.output.resolve()}')
    finally:
        if bd.projects.current != original:
            bd.projects.set_current(original)


if __name__ == '__main__':
    main()
