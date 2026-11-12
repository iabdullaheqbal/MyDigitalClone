"""Compare sparse, dense, and hybrid retrieval against the same dataset."""

from clone.evaluation.cli import main

if __name__ == "__main__":
    main(compare_modes=True)
