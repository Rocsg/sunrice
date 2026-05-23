"""Allow ``python -m marvel_view.scripts.water_conductance``."""
import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
