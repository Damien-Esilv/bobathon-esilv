"""pytest configuration: disable MPS before torch loads to avoid fork-segfault."""

import os

# Must be set before torch is imported by any test module.
os.environ["DISABLE_MPS_FOR_TESTS"] = "1"
