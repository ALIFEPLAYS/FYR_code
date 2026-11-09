"""Shared file locations for the four-section project layout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUPERPRO_DIR = ROOT / '1_SuperPro_Results'
LCA_DIR = ROOT / '2_LCA'
SEQUENTIAL_DIR = ROOT / '3_Sequential_Supply_Chain_MILP'
JOINT_DIR = ROOT / '4_Joint_MILP'
