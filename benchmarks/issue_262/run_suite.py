"""Run the three-policy streaming comparison in isolated processes."""

import sys

from benchmarks.issue_246.run_suite import main

if __name__ == "__main__":
    if "--directions" not in sys.argv:
        sys.argv.extend([
            "--directions",
            "isotropic,ellipsoidal,streaming",
        ])
    main()
