"""
Roll production back to an earlier model version.

    python rollback.py <version>

Example:
    python rollback.py model_20261016_093000

Find version names in models/registry.csv.
"""

import os
import shutil
import sys

PRODUCTION_MODEL = "models/production.pkl"


def main():
    if len(sys.argv) != 2:
        print("Usage: python rollback.py <version>   (see models/registry.csv)")
        sys.exit(1)
    source = f"models/{sys.argv[1]}.pkl"
    if not os.path.exists(source):
        print(f"Not found: {source}")
        sys.exit(1)
    shutil.copy(source, PRODUCTION_MODEL)
    print(f"Rolled back: {PRODUCTION_MODEL} is now {source}")


if __name__ == "__main__":
    main()
