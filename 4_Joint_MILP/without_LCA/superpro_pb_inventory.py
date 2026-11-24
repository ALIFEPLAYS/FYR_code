"""Per-batch foreground quantities for the supplementary ecoinvent PB factors."""

import math

import pandas as pd


def extract_batch_inventory(report_dir, assumptions=None):
    """Read the original SuperPro report sheets. Blank scored quantities mean unresolved."""
    assumptions = dict(assumptions or {})
    reports = {}
    for report in ('SR', 'EIR', 'ICR', 'IDR', 'EMS'):
        reports[report] = pd.read_excel(
            report_dir / f'mab_fedbatch_1000_sut_{report}.xls',
            sheet_name='Table p. 1', header=None, dtype=str, keep_default_na=False)

    def cell(report, row, column):
        return reports[report].iat[row - 1, ord(column) - ord('A')].strip()

    def check(report, row, column, label):
        if cell(report, row, column) != label:
            raise ValueError(f'Unexpected SuperPro layout at {report}!{column}{row}, expected {label!r}.')

    def number(report, row, column, label, label_col='B'):
        check(report, row, label_col, label)
        value = float(cell(report, row, column).replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid quantity at {report}!{column}{row}: {value}')
        return value

    records = []

    def record(key, code, unit, quantity, source, source_quantity, source_unit, basis):
        status = 'included' if quantity is not None else 'unresolved'
        if quantity is None and key in assumptions:
            quantity = assumptions.pop(key)
            status = 'supplied'
            basis += ' Additional per-batch quantity supplied in the factor unit.'
        if quantity is not None and (type(quantity) not in (float, int)
                                     or not math.isfinite(quantity) or quantity < 0):
            raise ValueError(f'Invalid PB quantity for {key}: {quantity!r}')
        prefix = 'release' if '-' in code else 'activity'
        records.append(dict(component=key, optimisation_flow=f'{prefix}__{code}', quantity_unit=unit,
                            quantity_per_batch=quantity, status=status, source=source,
                            source_quantity_per_batch=source_quantity, source_unit=source_unit, basis=basis))

    check('SR', 1100, 'G', 'INPUT')
    for component, row, name, code in [
        ('acetic_acid', 1101, 'Acetic-Acid', '89ddfe4d99001c74fe5fdba00443ef30'),
        ('disodium_phosphate', 1110, 'Na2HPO4', '00a4b7a531491d2211a784cea7ae2651'),
        ('phosphoric_acid', 1113, 'Phosphoric Acid', 'f84cfa144416d3c33a7229be84b15702'),
        ('sodium_chloride', 1115, 'Sodium Chloride', '8e9e8b96777e25cc19aa7c310f7f3104'),
        ('sodium_hydroxide', 1116, 'Sodium Hydroxid', '7525bc2877b280a42663999cd025cf0c'),
        ('trisodium_phosphate', 1117, 'Sodium Phosphat', 'ba595ef81d4278e5b9fb6d55ea0db436'),
    ]:
        mass = number('SR', row, 'G', name)
        record(component, code, 'kilogram', mass, f'SR G{row}', mass, 'kg active ingredient',
               'Process-wide component input, including mixtures and cleaning. WFI accounted separately.')

    check('EIR', 1281, 'B', 'P-11/2h CIP-1')
    check('EIR', 1281, 'J', 'Component Flow\n(kg/batch)')
    cip = number('EIR', 1285, 'J', 'TOTAL', 'C')
    record('cip_effluent', '853722e89539c9b5fa11ddb5a151683e', 'cubic meter', cip / 1000,
           'EIR J1285', cip, 'kg wastewater', 'Non-stream P-11 cleaning waste. Assume 1000 kg/m3 '
           'and generic treatment without reuse credit. Batch sheet charge independently totals 4.20 m3.')

    check('SR', 1073, 'F', 'S-141')
    check('SR', 1073, 'I', 'Product')
    inlet = number('SR', 1081, 'F', 'Total Enthalpy (kW-h)')
    outlet = number('SR', 1081, 'I', 'Total Enthalpy (kW-h)')
    check('IDR', 3185, 'B', 'Operation Name and Type: Freeze Bags (Batch Freezing)')
    efficiency = number('IDR', 3203, 'O', 'Efficiency') / 100
    if not 0 < efficiency <= 1 or outlet > inlet:
        raise ValueError('Invalid P-27 cooling balance or efficiency.')
    duty = (inlet - outlet) * 3.6 / efficiency
    record('brine_cooling', '852575286dfeaacb8e1a226cc5389fbc', 'megajoule', duty,
           'SR F1081/I1081 and IDR O3203', inlet - outlet, 'kWh product heat removed',
           'P-27 cooling supplied = enthalpy drop x 3.6 / freezer efficiency. '
           'Assume a separate cooling service at -25 C, the mean of the batch-sheet brine temperatures.')

    batches = number('ICR', 13, 'V', 'Number of Batches Per Year')
    if not batches:
        raise ValueError('Expected positive SuperPro reference batches per year.')
    check('ICR', 1567, 'B', 'CaCl2 Brine')
    brine = number('ICR', 1569, 'O', 'Filling') / batches
    record('brine_makeup', 'eb63804716531cd4717ffeaad09894ba', 'kilogram', None,
           'ICR O1569/V13', brine, 'tonnes circulating brine', 'Requires kg of fresh CaCl2 make-up.')
    check('SR', 505, 'M', 'Sterile Air')
    air = number('SR', 523, 'M', 'TOTAL (L/batch)') / 1000
    pressure = number('SR', 511, 'M', 'Pressure (bar)')
    record('compressed_air', '551efd52b08e7d48266202f80b3fb284', 'cubic meter', None,
           'SR M523/M511', air, 'm3 at process inlet', f'Stream pressure {pressure:g} bar. Requires '
           'equivalent 600 kPa gauge service volume and confirmation of electricity coverage.')
    counts = []
    for row, name in [(774, '100 mL T-Flask'), (779, '14 mL Test Tube'), (784, '2.2 L Roller Bottle')]:
        check('ICR', row, 'B', name)
        counts.append(number('ICR', row + 3, 'O', 'TOTAL') / batches)
    labware = ', '.join(f'{x:g} {name}' for x, name in zip(counts, ('flasks', 'tubes', 'bottles')))
    for key, code in [('labware_pp', 'b0abb5042456d17f58a95600bd9eebd7'),
                      ('labware_ps', 'efa6057bb5875d23a36a903eb65e23a8')]:
        record(key, code, 'kilogram', None, 'ICR O777/O782/O787 and V13', sum(counts),
               'shared labware item count', f'{labware} per batch. Requires net kg of this polymer.')

    # The EIR loads enter treatment. The generic treatment inventory already has releases.
    for key, row, name, source_unit, code in [
        ('effluent_p', 1061, 'TP', 'kg P', 'b2631209-8374-431e-b7d5-56c96c6b6d79'),
        ('effluent_phosphate', 1061, 'TP', 'kg P', '1727b41d-377e-43cd-bc01-9eaba946eccb'),
        ('effluent_ammonium', 1059, 'NH3', 'kg N', '13331e67-6006-48c4-bdb4-340c12010036'),
        ('effluent_nitrate', 1060, 'NO3_NO2', 'kg N', '7ce56135-2ca5-4fba-ad52-d62a34bfeb35'),
        ('effluent_cod', 1054, 'COD', 'kg oxygen demand', 'fc0b5c85-3b49-42c2-a3fd-db7e57b696e3'),
        ('effluent_bod5', 1057, 'BOD5', 'kg oxygen demand', '70d467b6-115e-43c5-add2-441de9411348'),
    ]:
        load = number('EIR', row, 'G', name)
        record(key, code, 'kilogram', None, f'EIR G{row}', load, source_unit + ' entering treatment',
               'Requires additional release to surface water after treatment and reconciliation with '
               'the generic inventory. Reported TP omits phosphorus from cleaning acid.')

    co2 = number('EMS', 434, 'D', 'Carb. Dioxide')
    fraction = assumptions.pop('fossil_co2_fraction', 0.0)
    if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
        raise ValueError('fossil_co2_fraction must be a number between 0 and 1.')
    for key, code, share in [('vent_co2_fossil', '349b29d1-3e58-4c66-98b9-9d1a076efd2e', fraction),
                             ('vent_co2_nonfossil', 'eba59fd6-f37e-41dc-9ca3-c7ea22d602c7', 1 - fraction)]:
        record(key, code, 'kilogram', co2 * share, 'EMS D434', co2, 'kg total vent CO2',
               f'Incubation and fermentation CO2. Assumed fossil share {fraction:g}.')
    if assumptions:
        raise ValueError(f'Unknown PB quantity inputs: {sorted(assumptions)}')
    inventory = pd.DataFrame(records)
    amounts = inventory.set_index('component')['quantity_per_batch']
    if amounts['effluent_p'] > 0 and amounts['effluent_phosphate'] > 0:
        raise ValueError('Enter the additional phosphorus release on one mass basis, P or phosphate.')
    return inventory
