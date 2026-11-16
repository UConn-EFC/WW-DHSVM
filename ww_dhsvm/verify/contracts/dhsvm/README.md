# DHSVM input contract

This folder holds the contract for DHSVM cases: what each input file must
look like for DHSVM to read it as intended.  The checks in
`ww_dhsvm/verify/dhsvm_*.py` are written against it.

There is one YAML file per input group:

| File | Covers | Check-ID prefix |
|---|---|---|
| `terrain.yaml` | `dem.bin`, `mask.bin` | `TER` |
| `streams.yaml` | `stream.network.dat`, `stream.map.dat`, `stream.class.dat` | `NET` |
| `config.yaml` | the DHSVM config file; the state files | `CFG` |
| `soils.yaml` | `soil.bin`, `soild.bin`, the soil table | `SOIL` |
| `vegetation.yaml` | `veg.bin`, the vegetation table | `VEG` |
| `forcing.yaml` | the met station files | `MET` |

For each file, a contract gives the format, data type, shape, units and
valid range, and the rules that tie the file to other files.  Every rule
names the ID of the check that enforces it.  Where a rule relies on what
DHSVM does, it cites the source as `file:line` in `pnnl/DHSVM-PNNL` at
`e6f333d` (`DHSVM/sourcecode/`).

`failures.yaml`, the catalog of known silent failures, also lives here.

The folder is empty until the contract and catalog tasks add their files.
Everything specific to DHSVM stays here and in the `dhsvm_*` modules; the
engine in `ww_dhsvm/verify/core.py` stays free of it, and other models
will get sibling folders.
