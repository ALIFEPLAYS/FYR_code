"""Independent cross-check of the central joint result over all feasible batch splits."""
import json

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.optimize import Bounds, LinearConstraint, milp

from run_joint_milp import RESULTS, accounting, check_close, completed, original_supply, solve, sym


def main():
    data = sym.load_model_data()
    upstream = original_supply(data)
    fx, _ = accounting(data)
    # Use the saved benchmark cost as the starting point. The saved joint answer
    # is the comparison target after all candidate allocations have been scored.
    summary = pd.read_csv(RESULTS / 'comparison.csv').set_index('case')
    joint = summary.loc['joint']
    base = summary.loc['two_stage']
    sites = data.sites
    # With fixed lines and total production, the only varying upstream term
    # is site-to-storage freight. Read each unit price from the original solve.
    freight = {j: fx * pyo.value(upstream.C_freight_jk[1, j, 'k1', 'S1']) / data.batches[j]
               for j in sites}
    # Measure capacity by maximising one site's batches at a time, with the
    # original line and processing-time constraints still in force.
    upstream.objective1.deactivate()
    maxima = []
    for j in sites:
        upstream.capacity_probe = pyo.Objective(expr=completed(upstream, j), sense=pyo.maximize)
        solve(upstream)
        maxima.append(round(pyo.value(completed(upstream, j))))
        upstream.del_component(upstream.capacity_probe)

    # Apply the same shared recycler limits used in the joint run.
    audit = json.loads((RESULTS / 'validation.json').read_text())
    data.route_capacities_t.update(audit['shared_recycler_caps_t_per_year'])
    core = sym.build_linear_model(data)
    matrix = core.A.tocsr()
    lower, upper = core.b_l.copy(), core.b_u.copy()
    lb, ub = core.lower.copy(), core.upper.copy()
    # Keep the original IS formulation and pin each site to the split under
    # test by changing only the RHS of its legacy high/low equality.
    allocation_rows = []
    for j in sites:
        low = core.index['low_batch', j]
        batch = core.index['B', j]
        rows = np.flatnonzero(matrix[:, low].toarray().ravel())
        matches = [i for i in rows if matrix[i, batch] == 1 and matrix[i, low] == data.batch_delta]
        if len(matches) != 1:
            raise ValueError(f'Cannot identify original allocation equation for {j}')
        allocation_rows.append(matches[0])
        lb[low] = ub[low] = int(data.batches[j] == data.batch_low)
    expr = core.expressions['policy_cost']
    tested = []
    # Try all whole-batch counts for the first two sites within their capacities.
    # The third supplies the balance needed to reach 861 batches in this case.
    total = round(data.total_batches)
    for a in range(max(0, total - maxima[1] - maxima[2]), maxima[0] + 1):
        for b in range(max(0, total - a - maxima[2]), min(maxima[1], total - a) + 1):
            split = (a, b, total - a - b)
            for j, n, row in zip(sites, split, allocation_rows):
                lower[row] = upper[row] = n + data.batch_delta * lb[core.index['low_batch', j]]
            # For this fixed split, choose the cheapest IS plan including carbon value.
            # Waste routes, trips, power and water can change for each candidate.
            raw = milp(expr.coeff, integrality=core.integrality, bounds=Bounds(lb, ub),
                       constraints=LinearConstraint(matrix, lower, upper),
                       options={'mip_rel_gap': 1e-10})
            if not raw.success:
                raise RuntimeError(f'IS solve failed for {split}: {raw.message}')
            # Adjust product freight for the new split. Keep the original power charge
            # out of the supply subtotal because IS already prices electricity.
            delta_freight = sum(freight[j] * (n - data.batches[j]) for j, n in zip(sites, split))
            score = base.supply_chain_gbp - base.power_overlap_removed_gbp + delta_freight + expr.evaluate(raw.x)
            tested.append((score, split))
    # Rank all feasible splits by combined annual cost including carbon. Check both
    # the minimum value and the site counts against the joint solver.
    tested.sort()
    best_score, best_split = tested[0]
    check_close(best_score, joint.objective_gbp, 'Exhaustive minimum vs joint objective', 0.01)
    decisions = pd.read_csv(RESULTS / 'site_decisions.csv').query("case_id == 'joint'").set_index('site')
    for j, n in zip(sites, best_split):
        check_close(decisions.loc[j, 'annual_batches'], n, 'Independently verified allocation')
    report = dict(allocations_checked=len(tested), site_order=sites, site_capacity_batches=maxima,
                  best_split=best_split, minimum_gbp=best_score, second_best_split=tested[1][1],
                  margin_to_next_allocation_gbp=tested[1][0] - best_score)
    path = RESULTS / 'allocation_check.json'
    path.with_suffix('.tmp').write_text(json.dumps(report, indent=2) + '\n')
    path.with_suffix('.tmp').replace(path)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
