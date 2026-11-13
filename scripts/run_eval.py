"""Run retrieval evaluation, or draft behavior checks with --behavior."""

import sys

if __name__ == "__main__":
    if "--behavior" in sys.argv:
        from clone.evaluation.behavior_cli import main
    else:
        from clone.evaluation.cli import main
    main()
