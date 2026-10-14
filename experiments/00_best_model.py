# %% [markdown]
# # 00_best_model — Stable Best-Submission Dispatcher
#
# **Team: parKINGson**
# **Members:** Damien CATAU, Malo GERVAISE, Esteban GAUBERT, Maximilien STUDER
#
# This file serves as the canonical entry point for the winning model.
# It dynamically imports and executes `run()` from whichever experiment
# achieved the lowest patient-grouped cross-validation RMSE.
#
# Inputs:
#   None (delegates to the selected experiment's run() function).
# Outputs:
#   Generates `submissions/best_submission.csv` and pushes the report to Skore Hub.

# %%
import importlib.util
import sys
from pathlib import Path

# ↓ Change this filename to point at the current winning experiment.
WINNER = "13_super_moe_final_sota.py"

# --- loader (do not edit below this line) ---
_winner_path = Path(__file__).parent / WINNER
_spec = importlib.util.spec_from_file_location("_winner", _winner_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules["_winner"] = _module
_spec.loader.exec_module(_module)
run = _module.run

if __name__ == "__main__":
    run()
