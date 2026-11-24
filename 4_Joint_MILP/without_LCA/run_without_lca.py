"""Run Alfie's unchanged economic-only model using the shared project inputs."""
import importlib.util
import json
import os
import platform
import sys
from importlib.metadata import version
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
        # Regenerated 23 Nov 2026 from extract_manufacturing_footprint_and_costs.ipynb; byte-identical to itemised_costs_20261024.csv. Original 20261026 file not found.
        'itemised_costs': SUPERPRO_DIR / 'data/processed/itemised_costs_regenerated_20261123.csv',
    }
    # Record the Python environment in solver_status.json after every solve.
    # testjoint.py stays unchanged; this wraps its solve function only.
    solve = module.solve_joint_model

    def solve_and_record_environment(model, output_dir=module.RESULT_DIR, gams_executable=None):
        statuses = solve(model, output_dir, gams_executable)
        statuses['environment'] = {
            'python': platform.python_version(),
            'python_executable': sys.executable,
            **{package: version(package) for package in ('pyomo', 'pandas', 'openpyxl')},
        }
        (Path(output_dir).resolve() / 'solver_status.json').write_text(
            json.dumps(statuses, indent=2), encoding='utf-8')
        return statuses

    module.solve_joint_model = solve_and_record_environment
    return module


if __name__ == '__main__':
    load_model_module().main()
