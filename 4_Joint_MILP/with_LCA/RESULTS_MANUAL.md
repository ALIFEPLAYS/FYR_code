# Joint MILP results manual

Moving eight batches from Oxford and eight from Cambridge to Slough saves **£1,700 per year in economic costs**. The network delivers the same 4,003.65 kg of antibody. Both plans use the same waste treatments, but the joint plan needs two fewer trips to the specialist plastics recycler.

## Compare the annual plans

The benchmark runs Cian’s supply chain model first, then its industrial symbiosis (IS) model, using the workbooks in this project. The joint model chooses batches and waste routes together. Both columns use the current input data, 2026 pounds and the same carbon valuation. The paper used an earlier plastics inventory, so its published totals differ from this benchmark.

| Production site | Two-stage batches | Joint batches | Change | Modelled capacity |
|---|---:|---:|---:|---:|
| Oxford | 328 | 320 | −8 | 328 |
| Cambridge | 328 | 320 | −8 | 328 |
| Slough | 205 | 221 | +16 | 242 |
| **Total** | **861** | **861** | **0** | |

| Annual measure | Two-stage | Joint | Joint minus two-stage |
|---|---:|---:|---:|
| Finished-product freight | £405,815 | £407,016 | +£1,200 |
| Waste transport | £38,643 | £35,742 | −£2,901 |
| **Total economic cost**, including annualised capital | **£479,757,297** | **£479,755,597** | **−£1,700** |
| Modelled emissions | 4,629.93 t CO₂e | 4,628.67 t CO₂e | −1.27 t CO₂e |
| Value assigned to those emissions | £1,521,923 | £1,521,507 | −£416 |
| **Economic cost plus carbon valuation** | **£481,279,221** | **£481,277,104** | **−£2,116** |

Freight and waste transport are already included in total economic cost. The last row is the quantity the solver minimises. Differences use the unrounded results. [Cost and impact totals](results/comparison.csv)

## Why 320 batches matters

At 328 batches, Oxford and Cambridge each generate about 102.4 tonnes of TPE, a rubber-like plastic. A truck carries 20 tonnes, so each site needs six trips. At 320 batches, each produces just under 100 tonnes and can use five trips. Slough takes the extra batches within its existing trips. Waste trucks travel **1,120 fewer vehicle-kilometres**, accounting for the modelled emissions saving. The extra product freight is priced, but its emissions are outside the supplied LCA data. [Site decisions](results/site_decisions.csv) · [Routes and truck distances](results/route_mix.csv)

| Waste or resource decision | Two-stage | Joint |
|---|---:|---:|
| Polyolefin plastics sent for recycling | 464.81 t | 464.81 t |
| TPE sent for recycling | 268.81 t | 268.81 t |
| Plastic sent directly to landfill | 18.18 t | 18.18 t |
| Saleable plastic recovered after processing | 560.01 t | 560.01 t |
| Diesel waste trips, all routes | 43 | 41 |
| Treated water reused | 5,812.52 m³ | 5,812.52 m³ |
| Manufacturing electricity | 100% wind | 100% wind |
| Residual biomass | External digestion for steam | External digestion for steam |

The landfill row counts direct shipments. The **191.79 tonnes of total residual plastic** in each case also include losses during recycling.

## How the model counts manufacturing electricity once

The supply-chain subtotal includes SuperPro’s original manufacturing electricity bill. The IS subtotal includes the electricity actually selected for each site. The objective subtracts the original bill once so that only the selected electricity is paid for. The existing electricity-demand constraint still requires grid and wind purchases to cover the batches made at each site.

> Minimise supply-chain cost − embedded manufacturing electricity charge + full net IS cost + carbon valuation.

Supply-chain cost includes manufacturing, annualised capital, storage and product freight. Net IS cost includes electricity, water, waste treatment and transport, less recovered-material income. Carbon valuation is modelled tonnes of CO₂e times £328.714056 per tonne. The original shortage penalty is zero because all demand-zone minimums must be met.

**Example.** Suppose supply-chain cost is £100 including £10 for electricity. The selected electricity costs £12 and other IS activities cost £6. Economic cost is £100 − £10 + (£12 + £6) = **£108**, before carbon valuation.

In the results CSV, subtract `power_overlap_removed_gbp` from `supply_chain_gbp` and add `is_private_gbp` to reproduce the economic cost. [Joint objective](run_joint_milp.py)

## Which rules determine the result?

The operating decisions include whole batches per site and line, product shipments, tonnes sent down each waste route, whole truck trips, electricity purchases by source and water reuse or discharge.

| Origin | Rule | Purpose |
|---|---|---|
| Original supply chain | Keep line processing times, stage balances, shipments, storage flows and demand-zone requirements. | Every counted batch must be producible and deliverable. |
| Original IS | Keep 861 total batches, per-batch waste and utility balances, eligible treatment routes, truck payloads, wastewater capacity and biomass destination rules. | All generated waste and resource use must be accounted for. |
| Modified for joint optimisation | Remove the low-site flags and four equations requiring two sites at 328 batches and one at 205. | Allows intermediate allocations such as 320 / 320 / 221. |
| New link | At each site, set IS batches equal to completed fill-and-finish batches. | Moving production also moves the associated waste and utility demand. |
| Fixed for this comparison | Fix the benchmark sites, scales and assigned lines. Fix demand shortfall at zero. | Both plans meet demand with the same equipment. |
| Existing recycler limit activated | Where the inputs lack a contracted capacity, cap each shared recycler at its benchmark intake across all three sites. | Tests reallocation with no increase above demonstrated model throughput. Actual commercial capacities would need separate evidence. |

## What the validation establishes

The main script replays the two-stage decisions in the joint model and checks that the combined objective agrees within a penny. It then releases the decisions, solves the joint model, checks its 504 active constraints and compares its score with the solver’s proven lower bound. [Solver checks](results/validation.json)

The second script tests all 741 batch splits separately. For each split it optimises the IS decisions and adjusts factory-to-storage freight, the only upstream cost that varies in this fixed three-site case. It finds the same optimum. The next split, **320 / 319 / 222**, costs only **£43.99 more per year** including carbon. That small margin matters when interpreting the precise allocation. Both calculations rely on the same cost and emissions inputs. [Allocation check](check_allocations.py) · [Verified best and next-best splits](results/allocation_check.json)
