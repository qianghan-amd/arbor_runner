"""The backend name local.yaml files and run_config.json records used before 2026-11-21 (`backend: crusoe_slurm`).
It is the generic Slurm backend (slurm.py); new configs say `backend: slurm` and name their cluster."""
from .slurm import Slurm, remote, slurm_time, to_minutes  # noqa: F401  (tests and old callers import them from here)


class CrusoeSlurm(Slurm):
    name = "crusoe_slurm"


BACKEND = CrusoeSlurm
