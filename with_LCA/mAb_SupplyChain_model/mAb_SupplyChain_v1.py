# -*- coding: utf-8 -*-
"""
Deterministic mAb supply-chain network design model.

Adapted from the original supply-chain optimisation formulation created by
ms3916 on 27 November 2023.

Main assumptions and changes for this thesis case
-------------------------------------------------
1. The model represents a hypothetical multi-product commercial mAb network,
   rather than demand for one named therapeutic product.

2. Aggregate annual demand is set to 4,000 kg mAb/year. This represents an
   illustrative medium-to-large commercial mAb producer operating a portfolio
   of established products.

3. A single representative planning period of one year is modelled. The model
   is therefore a single-period annual network-design problem rather than a
   fixed-horizon or rolling-horizon multi-period formulation.

4. Upfront SuperPro capital investment is converted to an equivalent annual
   cost using a capital recovery factor. A 15-year equipment/facility life and
   an 8% annual discount rate are assumed.

5. Manufacturing yield, process duration, cycle time, annual throughput,
   labour cost, variable cost, facility cost and capital cost are taken from
   the SuperPro 1,000 L single-use fed-batch mAb process.

6. The SuperPro process produces approximately 4.65 kg final mAb per batch,
   with a 2.50-day recipe cycle time and 121 batches per upstream line per
   operating year.

7. Three candidate manufacturing sites, one storage node and five equally
   sized demand zones are retained from the source network structure.

8. The first deterministic case includes manufacturing, capital, storage and
   freight costs. Industrial-symbiosis and waste-routing decisions are added
   in a subsequent model stage.

9. Cost results are reported in the SuperPro 2019 USD basis. No inflation
   adjustment is currently applied.

10. Candidate-site and demand-zone distances remain illustrative case-study
    inputs and should not be interpreted as actual pharmaceutical locations.
"""

# Preamble and packages
import os
import shutil
from pathlib import Path
from time import process_time

import numpy as np
import pandas as pd
import pyomo.environ as pyo

t1_start = process_time()


# GAMS solver configuration
GAMS_RESOURCE_DIR = Path("/Library/Frameworks/GAMS.framework/Resources")
if GAMS_RESOURCE_DIR.exists():
    os.environ["PATH"] = str(GAMS_RESOURCE_DIR) + os.pathsep + os.environ.get("PATH", "")

DEFAULT_SOLVER_NAME = os.environ.get("IS_SUPPLY_SOLVER", "auto").strip().lower()


# %% mAb supply-chain optimisation model formulation
def create_mab_supply_chain_model(scenario_set, scenario_probability, r_u, alpha_u, alpha_d, alpha_g, x_g, C_LABju,
                                  C_LABjd, C_LABjg, C_CAPju, C_CAPjd, C_CAPjg, C_FACju, C_FACjd, C_FACjg, C_VARju,
                                  C_VARjd, C_VARjg, sales_activation, total_demand_mab):
    ## Universal sets

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
        return model.cost[s, i] == + sum(
            (model.TC_OPj[t, s, i] + model.TC_freight[t, s] + model.TC_OPk[t, s]) for t in T) + 100 * sum(
            (model.Y_j[1, j]) for j in J)

    model.COST = pyo.Constraint(S, I, rule=cost_rule)

    def capital_risk_rule(model, s, i):
        return model.capital_risk[s, i] == sum(model.TC_CAPj[t, s, i] for t in T)

    model.capital_riskkk = pyo.Constraint(S, I, rule=capital_risk_rule)

    def transport_rule(model, s):
        return model.transport_cost[s] == sum((model.TC_freight[t, s]) for t in T)

    model.transport_cost_cons = pyo.Constraint(S, rule=transport_rule)

    def manufacturing_rule(model, s, i):
        return model.manufacturing_cost[s, i] == sum((model.TC_OPj[t, s, i]) for t in T)

    model.manufacturing_cost_cons = pyo.Constraint(S, I, rule=manufacturing_rule)

    def storage_rule(model, s):
        return model.storage_cost[s] == sum((model.TC_OPk[t, s]) for t in T)

    model.storage_cost_cons = pyo.Constraint(S, rule=storage_rule)

    def total_capital_cost_rule(model, t, s, i):
        return model.TC_CAPj[t, s, i] == sum((model.C_CAPj[t, j, s, i]) for j in J)

    model.total_capital_cost_constraint = pyo.Constraint(T, S, I, rule=total_capital_cost_rule)

    def capital_cost_rule(model, t, j, s, i):
        return model.C_CAPj[t, j, s, i] == sum(
            (C_CAPju[a, i, s] * model.M_u[t, j, a, u, i]) for a in A for u in U) + sum(
            (C_CAPjd[a, i, s] * model.M_d[t, j, a, d, i]) for a in A for d in D) + sum(
            (C_CAPjg[a, i, s] * model.M_g[t, j, a, g, i]) for a in A for g in G)

    model.capital_cost = pyo.Constraint(T, J, S, I, rule=capital_cost_rule)

    def total_operating_cost_rule(model, t, s, i):
        return model.TC_OPj[t, s, i] == sum((model.C_OPj[t, j, s, i]) for j in J)

    model.total_operating_cost = pyo.Constraint(T, S, I, rule=total_operating_cost_rule)

    def total_varlab_cost_rule(model, s, i):
        return model.production_cost[s, i] == sum(
            (C_LABju[a, i, s] * model.B_u[t, j, a, u, i, s] + C_VARju[a, i, s] * model.B_u[t, j, a, u, i, s]) for a in A
            for u in U for j in J for t in T) + sum(
            (C_LABjd[a, i, s] * model.B_d[t, j, a, d, i, s] + C_VARjd[a, i, s] * model.B_d[t, j, a, d, i, s]) for a in A
            for t in T for d in D for j in J) + + sum(
            (C_LABjg[a, i, s] * model.B_g[t, j, a, g, i, s] + C_VARjg[a, i, s] * model.B_g[t, j, a, g, i, s]) for a in A
            for g in G for j in J for t in T)

    model.total_varlab_cost = pyo.Constraint(S, I, rule=total_varlab_cost_rule)

    def operating_cost_rule(model, t, j, s, i):
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
        return model.TC_OPk[t, s] == sum((model.C_OPk[t, k, s]) for k in K)

    model.total_operating_storage_cost = pyo.Constraint(T, S, rule=total_operating_storage_cost_rule)

    def operating_storage_cost_rule(model, t, k, s):
        return model.C_OPk[t, k, s] == sum(
            (t_s * model.Q_kl[t, k, l, a, i, s] * x_g[a, i, s]) for a in A for i in I for l in L)

    model.operating_storage_cost = pyo.Constraint(T, K, S, rule=operating_storage_cost_rule)

    def total_freight_cost_rule(model, t, s):
        return model.TC_freight[t, s] == sum((model.C_freight_jk[t, j, k, s]) for j in J for k in K) + sum(
            (model.C_freight_kl[t, k, l, s]) for k in K for l in L)

    model.total_freight_cost = pyo.Constraint(T, S, rule=total_freight_cost_rule)

    def freight_cost_jk_rule(model, t, j, k, s):
        return model.C_freight_jk[t, j, k, s] == sum(
            (D_jk[j, k] * t_c * x_g[a, i, s] / product_unit[i] * model.Q_jk[t, j, k, a, i, s]) for a in A for i in I)

    model.freight_jk_cost = pyo.Constraint(T, J, K, S, rule=freight_cost_jk_rule)

    def freight_cost_kl_rule(model, t, k, l, s):
        return model.C_freight_kl[t, k, l, s] == sum(
            (D_kl[k, l] * t_c * x_g[a, i, s] / product_unit[i] * model.Q_kl[t, k, l, a, i, s]) for a in A for i in I)

    model.freight_kl_cost = pyo.Constraint(T, K, L, S, rule=freight_cost_kl_rule)

    #     #### Enough costs

    def manufacturing_existence_rule(model, t, j):
        return sum((model.Y_ja[t, j, a]) for a in A) == model.Y_j[t, j]

    model.existence = pyo.Constraint(T, J, rule=manufacturing_existence_rule)

    def usp_scale_selection_rule(model, t, j, a, u, s):
        return model.Z_u[t, j, a, u] <= model.Y_ja[t, j, a]

    model.usp_scale_selection = pyo.Constraint(T, J, A, U, S, rule=usp_scale_selection_rule)

    def dsp_scale_selection_rule(model, t, j, a, d, s):
        return model.Z_d[t, j, a, d] <= model.Y_ja[t, j, a]

    model.dsp_scale_selection = pyo.Constraint(T, J, A, D, S, rule=dsp_scale_selection_rule)

    def ff_scale_selection_rule(model, t, j, a, g, s):
        return model.Z_g[t, j, a, g] <= model.Y_ja[t, j, a]

    model.ff_scale_selection = pyo.Constraint(T, J, A, G, S, rule=ff_scale_selection_rule)

    def degeneracy_usp_rule(model, t, j, a, u):
        if u > 1:
            return model.Z_u[t, j, a, u] <= model.Z_u[t, j, a, u - 1]
        return pyo.Constraint.Skip

    model.degeneracy_usp = pyo.Constraint(T, J, A, U, rule=degeneracy_usp_rule)

    def degeneracy_dsp_rule(model, t, j, a, d):
        if d > 1:
            return model.Z_d[t, j, a, d] <= model.Z_d[t, j, a, d - 1]
        return pyo.Constraint.Skip

    model.degeneracy_dsp = pyo.Constraint(T, J, A, D, rule=degeneracy_dsp_rule)

    def degeneracy_ff_rule(model, t, j, a, g):
        if g > 1:
            return model.Z_g[t, j, a, g] <= model.Z_g[t, j, a, g - 1]
        return pyo.Constraint.Skip

    model.degeneracy_ff = pyo.Constraint(T, J, A, G, rule=degeneracy_ff_rule)

    def production_order_usp_rule(model, t, j, a, u, i, s):
        if u > 1:
            return model.P_u[t, j, a, u, i, s] <= model.P_u[t, j, a, u - 1, i, s]
        return pyo.Constraint.Skip

    model.production_order_usp = pyo.Constraint(T, J, A, U, I, S, rule=production_order_usp_rule)

    def production_order_dsp_rule(model, t, j, a, d, i, s):
        if d > 1:
            return model.P_d[t, j, a, d, i, s] <= model.P_d[t, j, a, d - 1, i, s]
        return pyo.Constraint.Skip

    model.production_order_dsp = pyo.Constraint(T, J, A, D, I, S, rule=production_order_dsp_rule)

    def production_order_ff_rule(model, t, j, a, g, i, s):
        if g > 1:
            return model.P_g[t, j, a, g, i, s] <= model.P_g[t, j, a, g - 1, i, s]
        return pyo.Constraint.Skip

    model.production_order_ff = pyo.Constraint(T, J, A, G, I, S, rule=production_order_ff_rule)

    def usp_product_allocation_rule(model, t, j, a, u, i):
        return model.M_u[t, j, a, u, i] <= model.Z_u[t, j, a, u]

    model.usp_product_allocation = pyo.Constraint(T, J, A, U, I, rule=usp_product_allocation_rule)

    def dsp_product_allocation_rule(model, t, j, a, d, i):
        return model.M_d[t, j, a, d, i] <= model.Z_d[t, j, a, d]

    model.dsp_product_allocation = pyo.Constraint(T, J, A, D, I, rule=dsp_product_allocation_rule)

    def ff_product_allocation_rule(model, t, j, a, g, i):
        return model.M_g[t, j, a, g, i] <= model.Z_g[t, j, a, g]

    model.ff_product_allocation = pyo.Constraint(T, J, A, G, I, rule=ff_product_allocation_rule)

    def usp_product_compatible_rule(model, t, j, a, u, i, s):
        return model.W_u[t, j, a, u, i, s] <= model.M_u[t, j, a, u, i]

    model.usp_product_compatible = pyo.Constraint(T, J, A, U, I, S, rule=usp_product_compatible_rule)

    def dsp_product_compatible_rule(model, t, j, a, d, i, s):
        return model.W_d[t, j, a, d, i, s] <= model.M_d[t, j, a, d, i]

    model.dsp_product_compatible = pyo.Constraint(T, J, A, D, I, S, rule=dsp_product_compatible_rule)

    def ff_product_compatible_rule(model, t, j, a, g, i, s):
        return model.W_g[t, j, a, g, i, s] <= model.M_g[t, j, a, g, i]

    model.ff_product_compatible = pyo.Constraint(T, J, A, G, I, S, rule=ff_product_compatible_rule)

    def network_link_jk_rule(model, t, j, k):
        return model.X_jk[t, j, k] <= model.Y_j[t, j]

    model.network_link_jk = pyo.Constraint(T, J, K, rule=network_link_jk_rule)

    def network_link_kl_rule(model, t, k, l):
        return model.X_kl[t, k, l] <= model.Y_k[t, k]

    model.network_link_kl = pyo.Constraint(T, K, L, rule=network_link_kl_rule)

    def sourcing_link_jk_rule(model, t, k):
        return sum((model.X_jk[t, j, k]) for j in J) >= model.Y_k[t, k]

    model.sourcing_link_jk = pyo.Constraint(T, K, rule=sourcing_link_jk_rule)

    def inventory_ud_rule(model, t, j, a, i, s):
        if t == 1:
            return model.I_u[t, j, a, i, s] == sum((model.B_u[t, j, a, u, i, s]) for u in U) - sum(
                (model.B_d[t, j, a, d, i, s]) for d in D)
        return model.I_u[t, j, a, i, s] == model.I_u[t - 1, j, a, i, s] + sum(
            (model.B_u[t, j, a, u, i, s]) for u in U) - sum((model.B_d[t, j, a, d, i, s]) for d in D)

    model.inventory_ud = pyo.Constraint(T, J, A, I, S, rule=inventory_ud_rule)

    def inventory_dg_rule(model, t, j, a, i, s):
        if t == 1:
            return model.I_d[t, j, a, i, s] == + sum((model.B_d[t, j, a, d, i, s]) for d in D) - sum(
                (model.B_g[t, j, a, g, i, s]) for g in G)
        return model.I_d[t, j, a, i, s] == model.I_d[t - 1, j, a, i, s] + sum(
            (model.B_d[t, j, a, d, i, s]) for d in D) - sum((model.B_g[t, j, a, g, i, s]) for g in G)

    model.inventory_dg = pyo.Constraint(T, J, A, I, S, rule=inventory_dg_rule)

    def inventory_j_rule(model, t, j, a, i, s):
        if t == 1:
            return model.I_g[t, j, a, i, s] == + sum((model.B_g[t, j, a, g, i, s]) for g in G) - sum(
                (model.Q_jk[t, j, k, a, i, s]) for k in K)
        return model.I_g[t, j, a, i, s] == model.I_g[t - 1, j, a, i, s] + sum(
            (model.B_g[t, j, a, g, i, s]) for g in G) - sum((model.Q_jk[t, j, k, a, i, s]) for k in K)

    model.inventory_j = pyo.Constraint(T, J, A, I, S, rule=inventory_j_rule)

    def inventory_k_rule(model, t, k, a, i, s):
        if t == 1:
            return model.I_k[t, k, a, i, s] == + sum((model.Q_jk[t, j, k, a, i, s]) for j in J) - sum(
                (model.Q_kl[t, k, l, a, i, s]) for l in L)
        return model.I_k[t, k, a, i, s] == model.I_k[t - 1, k, a, i, s] + sum(
            (model.Q_jk[t, j, k, a, i, s]) for j in J) - sum((model.Q_kl[t, k, l, a, i, s]) for l in L)

    model.inventory_k = pyo.Constraint(T, K, A, I, S, rule=inventory_k_rule)

    def inventory_l_rule(model, t, l, a, i, s):
        if t == 1:
            return model.I_l[t, l, a, i, s] == + sum((model.Q_kl[t, k, l, a, i, s]) for k in K) - model.S_l[
                t, l, a, i, s]
        return model.I_l[t, l, a, i, s] == model.I_l[t - 1, l, a, i, s] + sum(
            (model.Q_kl[t, k, l, a, i, s]) for k in K) - model.S_l[t, l, a, i, s]

    model.inventory_l = pyo.Constraint(T, L, A, I, S, rule=inventory_l_rule)

    def transport_flow_jk_rule(model, t, j, k, i, s):
        return model.transport_jk[t, j, k, i, s] == sum(
            (model.Q_jk[t, j, k, a, i, s] * x_g[a, i, s] / product_unit[i]) for a in A)

    model.transport_flow_jk = pyo.Constraint(T, J, K, I, S, rule=transport_flow_jk_rule)

    def transport_flow_kl_rule(model, t, k, l, i, s):
        return model.transport_kl[t, k, l, i, s] == sum(
            (model.Q_kl[t, k, l, a, i, s] * x_g[a, i, s] / product_unit[i]) for a in A)

    model.transport_flow_kl = pyo.Constraint(T, K, L, I, S, rule=transport_flow_kl_rule)

    def sales_product_rule(model, t, l, i, s):
        return model.Sales[t, l, i, s] == sum((model.S_l[t, l, a, i, s] * x_g[a, i, s] / product_unit[i]) for a in A)

    model.sales_product = pyo.Constraint(T, L, I, S, rule=sales_product_rule)

    def usp_time_allocation_max_rule(model, t, j, a, u, i, s):
        return model.T_u[t, j, a, u, i, s] <= T_MAX * model.W_u[t, j, a, u, i, s]

    model.usp_time_allocation_max = pyo.Constraint(T, J, A, U, I, S, rule=usp_time_allocation_max_rule)

    def usp_time_allocation_min_rule(model, t, j, a, u, i, s):
        return T_MINu * model.W_u[t, j, a, u, i, s] <= model.T_u[t, j, a, u, i, s]

    model.usp_time_allocation_min = pyo.Constraint(T, J, A, U, I, S, rule=usp_time_allocation_min_rule)

    def max_utilisation_rule(model, t, j, a, u, s, i):
        return model.T_u[t, j, a, u, i, s] - H_ja <= 0

    model.max_utilisation_rule = pyo.Constraint(T, J, A, U, S, I, rule=max_utilisation_rule)

    def dsp_time_allocation_max_rule(model, t, j, a, d, i, s):
        return model.T_d[t, j, a, d, i, s] <= T_MAX * model.W_d[t, j, a, d, i, s]

    model.dsp_time_allocation_max = pyo.Constraint(T, J, A, D, I, S, rule=dsp_time_allocation_max_rule)

    def dsp_time_allocation_min_rule(model, t, j, a, d, i, s):
        return T_MINu * model.W_d[t, j, a, d, i, s] <= model.T_d[t, j, a, d, i, s]

    model.dsp_time_allocation_min = pyo.Constraint(T, J, A, D, I, S, rule=dsp_time_allocation_min_rule)

    def ff_time_allocation_max_rule(model, t, j, a, g, i, s):
        return model.T_g[t, j, a, g, i, s] <= T_MAX * model.W_g[t, j, a, g, i, s]

    model.ff_time_allocation_max = pyo.Constraint(T, J, A, G, I, S, rule=ff_time_allocation_max_rule)

    def ff_time_allocation_min_rule(model, t, j, a, g, i, s):
        return model.T_g[t, j, a, g, i, s] >= T_MINu * model.W_g[t, j, a, g, i, s]

    model.ff_time_allocation_min = pyo.Constraint(T, J, A, G, I, S, rule=ff_time_allocation_min_rule)

    def ff_batches_rule(model, t, j, a, g, i, s):
        return model.B_g[t, j, a, g, i, s] == model.W_g[t, j, a, g, i, s] + 1 / r_g * (
                    model.T_g[t, j, a, g, i, s] - alpha_g[s, i] * model.W_g[
                t, j, a, g, i, s])  #### Need to recheck this expression

    model.ff_batches = pyo.Constraint(T, J, A, G, I, S, rule=ff_batches_rule)

    def usp_batches_rule(model, t, j, a, u, i, s):
        return model.B_u[t, j, a, u, i, s] == model.W_u[t, j, a, u, i, s] + 1 / r_u[s, i] * (
                    model.T_u[t, j, a, u, i, s] - alpha_u[s, i] * model.W_u[t, j, a, u, i, s])

    model.usp_batches = pyo.Constraint(T, J, A, U, I, S, rule=usp_batches_rule)

    def dsp_batches_rule(model, t, j, a, d, i, s):
        return model.B_d[t, j, a, d, i, s] == model.W_d[t, j, a, d, i, s] + 1 / r_d * (
                    model.T_d[t, j, a, d, i, s] - alpha_d[s, i] * model.W_d[t, j, a, d, i, s])

    model.dsp_batches = pyo.Constraint(T, J, A, D, I, S, rule=dsp_batches_rule)

    def usp_mass_produced_rule(model, t, j, a, u, i, s):
        return model.P_u[t, j, a, u, i, s] == model.B_u[t, j, a, u, i, s] * x_u[a, i, s] / product_unit[i]

    model.usp_mass_produced = pyo.Constraint(T, J, A, U, I, S, rule=usp_mass_produced_rule)

    def dsp_mass_produced_rule(model, t, j, a, d, i, s):
        return model.P_d[t, j, a, d, i, s] == model.B_d[t, j, a, d, i, s] * x_d[a, i, s] / product_unit[i]

    model.dsp_mass_produced = pyo.Constraint(T, J, A, D, I, S, rule=dsp_mass_produced_rule)

    def ff_mass_produced_rule(model, t, j, a, g, i, s):
        return model.P_g[t, j, a, g, i, s] == model.B_g[t, j, a, g, i, s] * x_g[a, i, s] / product_unit[i]

    model.ff_mass_produced = pyo.Constraint(T, J, A, G, I, S, rule=ff_mass_produced_rule)

    def min_transport_jk_rule(model, t, j, k, s):
        return Q_min * model.X_jk[t, j, k] <= sum((model.Q_jk[t, j, k, a, i, s]) for i in I for a in A)

    model.min_transport_jk = pyo.Constraint(T, J, K, S, rule=min_transport_jk_rule)

    def max_transport_jk_rule(model, t, j, k, s):
        return sum((model.Q_jk[t, j, k, a, i, s]) for i in I for a in A) <= Q_max * model.X_jk[t, j, k]

    model.max_transport_jk = pyo.Constraint(T, J, K, S, rule=max_transport_jk_rule)

    def min_transport_kl_rule(model, t, k, l, s):
        return Q_min * model.X_kl[t, k, l] <= sum((model.Q_kl[t, k, l, a, i, s]) for i in I for a in A)

    model.min_transport_kl = pyo.Constraint(T, K, L, S, rule=min_transport_kl_rule)

    def max_transport_kl_rule(model, t, k, l, s):
        return sum((model.Q_kl[t, k, l, a, i, s]) for i in I for a in A) <= Q_max * model.X_kl[t, k, l]

    model.max_transport_kl = pyo.Constraint(T, K, L, S, rule=max_transport_kl_rule)

    def inventory_max_usp_rule(model, t, j, a, i, s):
        return model.I_u[t, j, a, i, s] <= 0

    model.inventory_max_usp = pyo.Constraint(T, J, A, I, S, rule=inventory_max_usp_rule)

    def inventory_max_dsp_rule(model, t, j, a, i, s):
        return model.I_d[t, j, a, i, s] <= 0

    model.inventory_max_dsp = pyo.Constraint(T, J, A, I, S, rule=inventory_max_dsp_rule)

    def inventory_max_ff_rule(model, t, j, a, i, s):
        return model.I_g[t, j, a, i, s] <= 0

    model.inventory_max_ff = pyo.Constraint(T, J, A, I, S, rule=inventory_max_ff_rule)

    def inventory_max_stor_rule(model, t, k, a, i, s):
        return model.I_k[t, k, a, i, s] <= 0

    model.inventory_max_stor = pyo.Constraint(T, K, A, I, S, rule=inventory_max_stor_rule)

    def inventory_max_demand_rule(model, t, l, a, i, s):
        return model.I_l[t, l, a, i, s] <= 0

    model.inventory_max_demand = pyo.Constraint(T, L, A, I, S, rule=inventory_max_demand_rule)

    def expected_cost_rule(model):
        return model.expected_cost == sum(
            (phi[s] * (model.cost[s, i] + model.capital_risk[s, i])) for i in I for s in S)

    model.expect_cost = pyo.Constraint(rule=expected_cost_rule)

    def missed_deliveries_rule(model, t, l, i, s):
        if sales_activation == True:
            return D_l[t, l, i] <= model.Sales[t, l, i, s]
        return D_l[t, l, i] <= model.Sales[t, l, i, s] + model.Slack[t, l, i, s]

    model.missed_deliveries = pyo.Constraint(T, L, I, S, rule=missed_deliveries_rule)

    def actual_demand(model, t, l, i):
        return + model.demand[t, l, i] == D_l[t, l, i]

    model.actual = pyo.Constraint(T, L, I, rule=actual_demand)

    def objective_rule(model):
        if sales_activation == True:
            return model.expected_cost
        return model.expected_cost + sum((model.Slack[t, l, i, s] * 1e12) for t in T for l in L for i in I for s in S)

    model.objective1 = pyo.Objective(rule=objective_rule)

    def expected_sales_rule(model, i):
        return model.expected_sales[i] == sum((phi[s] * model.Sales[t, l, i, s]) for s in S for t in T for l in L)

    model.expected_saleszzz = pyo.Constraint(I, rule=expected_sales_rule)

    return model


# %% Direct mAb parameter assignment and first deterministic case
#
# This first version deliberately removes the paper-reproduction loops for
# best/median/worst/stochastic/random cases. No waste or routing quantities
# are included in this file.


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


def solve_mab_model(model, solver_name=DEFAULT_SOLVER_NAME, tee=True):
    """Solve the unchanged supply-chain formulation with an available MILP solver."""
    candidates = ("gams:cplex", "highs") if solver_name == "auto" else (solver_name,)
    for candidate in candidates:
        if candidate == "gams:cplex" and shutil.which("gams") is None:
            continue
        solver = pyo.SolverFactory(candidate)
        if solver is not None and solver.available(exception_flag=False):
            print(f"Solver: {candidate}")
            if candidate == "gams:cplex":
                return solver.solve(
                    model, tee=tee, keepfiles=False,
                    symbolic_solver_labels=True,
                )
            # Match the tight optimality standard used for the central MILP.
            solver.options["mip_rel_gap"] = 1e-9
            return solver.solve(model, tee=tee)
    raise RuntimeError(
        f"No available supply-chain MILP solver for {solver_name!r}. "
        "Install highspy or configure a working GAMS/CPLEX interface."
    )


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


def run_first_mab_case():
    """Build, solve, print, and export the first deterministic mAb case."""
    data = build_first_mab_input_data()

    model = create_mab_supply_chain_model(
        scenario_set=data["scenarios"],
        scenario_probability=data["scenario_probability"],
        r_u=data["r_u"],
        alpha_u=data["alpha_u"],
        alpha_d=data["alpha_d"],
        alpha_g=data["alpha_g"],
        x_g=data["x_g"],
        C_LABju=data["C_LABju"],
        C_LABjd=data["C_LABjd"],
        C_LABjg=data["C_LABjg"],
        C_CAPju=data["C_CAPju"],
        C_CAPjd=data["C_CAPjd"],
        C_CAPjg=data["C_CAPjg"],
        C_FACju=data["C_FACju"],
        C_FACjd=data["C_FACjd"],
        C_FACjg=data["C_FACjg"],
        C_VARju=data["C_VARju"],
        C_VARjd=data["C_VARjd"],
        C_VARjg=data["C_VARjg"],
        sales_activation=False,
        total_demand_mab=data["total_demand_mab"],
    )

    print(f"Solving first deterministic mAb supply-chain case (solver={DEFAULT_SOLVER_NAME})...")
    results = solve_mab_model(
        model,
        solver_name=DEFAULT_SOLVER_NAME,
        tee=True,
    )

    termination = results.solver.termination_condition
    if termination not in {pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible}:
        raise RuntimeError(f"Solver terminated with condition: {termination}")

    result_table = extract_first_mab_results(model)
    output_path = Path(__file__).resolve().parent / "results/mab_supply_chain_superpro_v2_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(output_path, index=False)

    print("\nSelected mAb network and production allocation:")
    print(result_table.to_string(index=False))
    annualised_cost = pyo.value(model.expected_cost)
    annual_supply = pyo.value(model.expected_sales["mAb"])

    print(
        f"\nExpected annualised total cost: "
        f"{annualised_cost:,.2f} USD/year"
    )
    print(
        f"Expected mAb supplied: "
        f"{annual_supply:,.2f} kg/year"
    )
    print(
        f"Annualised cost per kg supplied: "
        f"{annualised_cost / annual_supply:,.2f} USD/kg"
    )
    print(f"Results written to: {output_path}")

    return model, results, result_table


if __name__ == "__main__":
    run_first_mab_case()