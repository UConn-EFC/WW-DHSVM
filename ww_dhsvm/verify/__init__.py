"""Verification layer: checks that a finished DHSVM case is correct, not just runnable.

A case can run to the end, close its water balance and match the outlet
hydrograph while an input is wrong inside the basin.  This package holds
the checks that look for such errors in the files WW-DHSVM writes.

The layer has two parts, and the package keeps them apart:

* the **engine** (``ww_dhsvm/verify/core.py``): one finding format, a
  check registry and a runner.  It knows nothing about DHSVM, so that a
  later step can add other models;
* the **DHSVM rules**: the contract files in ``contracts/dhsvm/`` (one
  YAML file per input group) and the checks in the ``dhsvm_*`` modules
  of this package.

This ``__init__`` imports none of the DHSVM-specific modules.  Import the
modules you need by name.
"""
