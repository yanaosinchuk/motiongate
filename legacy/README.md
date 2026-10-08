# Legacy prototype

This directory preserves the original classroom prototype for provenance and comparison with the current implementation.

The initial motion-detection script was based on the tutorial source that was referenced in the original file:

- https://itsourcecode.com/free-projects/python-projects/motion-detection-opencv-python-with-source-code

The production implementation in `src/motiongate/` is a separate, tested redesign with explicit interfaces, scheduler guarantees, validation, reproducible benchmarks and documented failure modes. New development should target the package under `src/`, not this legacy script.
