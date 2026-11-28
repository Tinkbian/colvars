# OpenMM LDDM feasibility run

This example ports the two alchemical legs of BFEE3 LDDM for prepared Amber
inputs. It uses BFEE3's quaternion-based geometry templates, the six fitted
harmonic restraints, and protein translation and orientation restraints.
The solute is one molecule with B-column flag +1.

## Run

Install the resident CUDA build of OpenMM–Colvars, OpenMM, NumPy, SciPy and
cuPotential. Put the resident Python module and this repository's
`openmm/python` directory on `PYTHONPATH`. Select a single CUDA device with
`CUDA_VISIBLE_DEVICES`. Use the interpreter that can load both CUDA plugins.

```bash
python openmm/examples/lddm/workflow.py \
    --inputs /path/to/3flz/input \
    --bfee /path/to/BFEE3 --out /path/to/new/results \
    --bound-intervals 10 --solvent-intervals 5 --window-steps 20000 --discard 5000
```

The common preparation is 5 ns of complex equilibration, 5 ns of complex
re-equilibration, and 5 ns of solvent-leg equilibration. Each backend starts
from the same prepared coordinates, velocities and box. Each subsequent
state continues its own backend's trajectory. Completed stages can be reused
with the same arguments. Partial stages keep their logs and data for inspection.
To run one implementation throughout preparation and FEP, add `--mode cpu`,
`--mode gpu` or `--mode cupotential` and choose a separate output directory.

The complex needs `complex.parm7`, `complex.pdb`, `complex.ndx`,
`complex.xyz` and `fep.pdb`. The solvent leg accepts the BFEE3
`ligandOnly.parm7`, `ligandOnly.pdb`, `fep_ligandOnly.pdb` triplet. If
only the BFEE3 strip recipe is present, put AmberTools `cpptraj` on PATH
or supply `--cpptraj /path/to/cpptraj`. The driver builds the missing topology
in the output directory. Use `--solvent-inputs` to reuse a prepared triplet.
The 3FLZ solvent-leg comparison uses this 49,906-atom system. The separate
3,865-atom `ligo` pair in the input contracts close to 3 nm under NPT.
cuPotential 0.3.5 requires each box edge to stay at least three times the
1 nm cutoff throughout integration and pressure trials.

## Coupling and sampling

All three modes integrate on OpenMM CUDA in mixed precision. `cpu` selects
CPU Colvars, `gpu` selects GPU atom groups, and `cupotential` selects GPU
atom groups and the NVIDIA nonbonded force. `result.json` records the actual
backend. For these BFEE3 inputs the six scalar CVs and bias updates run on
the host in the resident GPU build. The free ligand has no geometric bias.

The coupling parameter decreases from 1 to 0. The bound restraint fraction
increases from 0 to 1. Ten bound intervals therefore have eleven sampled
states. Five solvent intervals have six. Every 50 steps, interior states
alternate their energy projection to the next or previous state. Endpoints
project to their only neighbour. Each CSV row records both physical and
restraint work for the same configuration in kcal/mol.
The schedule follows BFEE3's 0.5 electrostatic start, 0.7 van der Waals end,
and NAMD's default 5 Å² soft-core shift and intraligand annihilation. A
ligand-only PME correction gives the correct linear electrostatic energy
including self and image terms. The analytically projected restraint energy
uses the same geometry and centers as the force used in dynamics.

## Numerical conventions

The OpenMM protocol uses 300 K Langevin-middle integration, 2 fs, damping
1/ps, constrained hydrogen bonds, rigid water and an isotropic Monte Carlo
barostat at 1.01325 bar. The initial periodic vectors come from each
supplied PDB's CRYST1 record. BFEE3's NAMD template uses a Langevin piston,
heavy-atom thermostatting and reciprocal electrostatics every other step.
The OpenMM integrator thermostats all mobile atoms and evaluates PME each
step. AMBER uses a 1.0 nm cutoff and switching from 0.9 nm. The alchemical
LJ pair terms use the NAMD potential switch. The environment uses the
OpenMM backend's switch. All FEP modes request PME tolerance 0.0005 as
required by cuPotential 0.3.5. Native shared preparation requests 0.00001.
The native primary force uses Ewald PME. cuPotential converts that force
with its default C2 Prolate splitting. The ligand-only PME correction and
soft-core pair terms continue to use OpenMM kernels in all three modes.
When starting cuPotential directly from a PDB, the matching native OpenMM
force handles minimization, including any Reference-platform fallback.
Dynamics then uses the cuPotential force.

The complex equilibration records CVs every 50 steps. Histograms from those
frames use BFEE3's bin widths, modal centers and three-point fit. This
subsamples the histogram relative to BFEE3's per-step histogram bias and
avoids adding adaptive history to Monte Carlo volume trials.

`check.py` evaluates five fixed-coordinate lambda states per leg. Supply
`--namd-coor` with NAMD's binary coordinates after its zero-step constraint
projection when comparing the engines. `summary.json` contains each leg's
throughput (integration and work sampling, excluding setup), counts per
direction, forward/reverse hysteresis, BAR, midpoint bidirectional FEP and
the BFEE3 standard-state correction. Short runs assess execution and speed.

The 3FLZ validation includes the [completed window counts and timings](../../validation/3flz-lddm-run.json) and [fixed-coordinate comparisons with NAMD](../../validation/3flz-lddm.json).

[BFEE3 source used here](https://github.com/fhh2626/BFEE3/tree/ce8a98a94f97d48cfef48c8212d2f1239a4cbfe4)
· [NAMD free energy parameters](https://www.ks.uiuc.edu/Research/namd/3.0.1/ug/node57.html)
· [OpenMM integration and pressure coupling](https://docs.openmm.org/latest/userguide/application/02_running_sims.html)
· [NVIDIA conversion and box requirements](https://docs.nvidia.com/cuda/cupotential/nonbonded-force)
