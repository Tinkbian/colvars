# Upstream provenance

The initial OpenMM proxy, Force wrapper design, Python binding pattern, and PDB
selection code are adapted from the `openmm/` directory of Colvars pull request
979, commit 2ada654ecc7218409b66628f2a26c9deaba2f325.

The imported interface is distributed under GPL-3.0 (see LICENSE). The Colvars
core keeps its existing LGPL-3.0 license. This integration deliberately excludes
the unrelated FEP, TABF-integrator, and AlchemicalPME implementations.

## Source comparison, 2026-11-19

The first audit started from `Tinkbian/colvars:chinfo` commit
`8379ea411672171af23544e3186cac2507c413d2`. The second audit compared
the interface at `585762a039665a344670d4678c88ee8e214002a2` and its
local revisions against the same upstream source. The GitHub API in the
first audit and `git ls-remote https://github.com/Colvars/colvars.git refs/pull/979/head`
in the second both identified `2ada654ecc7218409b66628f2a26c9deaba2f325`.
[PR #979](https://github.com/Colvars/colvars/pull/979) ·
[Pinned source](https://github.com/Colvars/colvars/tree/2ada654ecc7218409b66628f2a26c9deaba2f325/openmm)

| Area | Official PR #979 | chinfo implementation and reason |
| --- | --- | --- |
| Python and build | SWIG exposes a C++ OpenMM Force, file and string configurations, explicit charges, and restart settings. The PR also builds alchemical and FEP interfaces. | Retains the Force wrapper and configuration model. The local CMake project builds the bundled Colvars sources with a selected OpenMM SDK. The Python API adds per-Context values, bias-state export, update counts, backend diagnostics and output flushing. |
| Units, atom IDs and PDB | Uses GROMACS units and one-based Colvars atom numbers. Its OpenMM PDB loader returns the numeric coordinates from the PDB fields. | The OpenMM PDB loader converts the Å fields to nm. Selected atom slots map to zero-based System particles. Charges are explicit per-particle values in elementary-charge units. |
| Periodic boundaries | Reads Context box vectors and updates Colvars boundaries when successive boxes differ beyond the PR's tolerance. | Checks the current box, prepares a reduced lattice for tilted cells and rejects unrepresentable cell heights before updating adaptive history. CUDA gathering also reconstructs reordered and wrapped coordinates, including mixed-precision corrections. |
| Force groups | The deferred CUDA post-computation saves the owner's group mask when constructed. | The force callback uses OpenMM's per-Context group number. Editing the System transitions to the new group when the caller creates or reinitializes a Context. |
| CUDA force callback | A deferred post-computation flushes the queue, reads host positions for the system, and uploads a full force array. | A direct ForceImpl callback gathers selected positions and scatters selected forces. The mapping follows OpenMM reorder events. |
| Repeated queries | A cache compares step, box, selected positions and alchemical parameters. | The adapter reevaluates the current geometry on each request. The core distinguishes state updates from evaluations on the saved bias surface. This supports changing a position or box at fixed step count. |
| Restart and history | OpenMM step counts are mapped to Colvars steps and the force exposes restart-file settings. | Checks the saved step and required adaptive records. The first restored evaluation reuses the saved bias, including pending hills and moving-restraint state. Consecutive evaluations are required while advancing history-dependent biases. |
| Physical total forces | Deferred computation, ordering checks and group restrictions support sampling for certain configurations. | The selected-atom bridge runs before all physical forces are guaranteed to have completed. ABF requires ordering, group-completeness checks and unbiased total-force sampling on the correct step. The proxy rejects this request. |
| Extended dynamics and TABF | The PR supplies alchemical parameters and a dedicated TABF integrator. | Repeat-safe updates for the extra dynamical variables would require integrator support. Extended-Lagrangian CVs are rejected. |
| CV execution and bias | The PR's CUDA force path evaluates Colvars using host coordinates. | GPU atom groups and native RMSD/coordination components reuse the branch's CUDA implementation. Eligible host COM components consume reduced centers. Single scalar COM components can send a three-component center force back through mapped memory for on-device mass weighting. The distance and bias calculations in the benchmark still run on the host. |

The audit compared implementation and API behavior from source. Runtime
tests exercised the chinfo plugin on OpenMM Reference, CPU, and single-device
CUDA. OpenCL/HIP, multi-device CUDA, engine-provided replica messaging,
alchemical parameter propagation and the TABF integrator are outside the
chinfo OpenMM interface. File-based replica exchange of hill records is
supported. Distributed replica jobs and full NAMD/GROMACS integrations
were outside the runtime matrix. The measured reference for throughput is
bare OpenMM or the named earlier chinfo build. PR #979 was not benchmarked.
[Review findings](REVIEW.md) · [Validation results](VALIDATION.md)
