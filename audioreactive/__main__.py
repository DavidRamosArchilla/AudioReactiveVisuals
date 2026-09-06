"""Package entry point: python -m audioreactive."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
