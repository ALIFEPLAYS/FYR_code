"""Run Alfie's unchanged economic-only model using the shared project inputs."""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from project_paths import LCA_DIR, ROOT, SEQUENTIAL_DIR, SUPERPRO_DIR


def load_model_module():
    spec = importlib.util.spec_from_file_location('alfie_joint_cost', HERE / 'testjoint.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Adapt locations only; retain the original model formulation and solver.
    module.CODE_DIR = HERE
    module.PROJECT_ROOT = HERE
    module.REPO_ROOT = ROOT
    module.PARAMETER_DIR = SEQUENTIAL_DIR / 'parameters'
    module.RESULT_DIR = HERE / 'results'
    module.INPUT_FILES = {
        'eligibility': SEQUENTIAL_DIR / 'MILP/inputs/polymer_route_eligibility_20260722.xlsx',
        'lca': Path(os.environ.get('IS_LCA_ROUTE_FACTOR_FILE', str(
            LCA_DIR / 'outputs/master_lca_route_factor_database_ecoinvent_3_12.xlsx'))),
        'params': SEQUENTIAL_DIR / 'parameters/MILP_central_params.json',
        'itemised_costs': SUPERPRO_DIR / 'data/processed/itemised_costs_20261026.csv',
    }
    return module


if __name__ == '__main__':
    load_model_module().main()
