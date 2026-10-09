# %% [markdown]
# # 00_best_model — stable best-submission dispatcher
#
# This file contains no model code. It loads `run()` from whichever numbered
# experiment currently holds the lowest GroupKFold RMSE and calls it.
#
# To nominate a new winner, change the one WINNER line below.
# The filename (00_best_model) and Hub key (best_submission) never change.

# %%
import importlib.util
import sys
from pathlib import Path

# ↓ Change this filename to point at the current winning experiment.
WINNER = "07_tuning.py"

# --- loader (do not edit below this line) ---
_winner_path = Path(__file__).parent / WINNER
_spec = importlib.util.spec_from_file_location("_winner", _winner_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules["_winner"] = _module
_spec.loader.exec_module(_module)
run = _module.run

if __name__ == "__main__":
    run()
