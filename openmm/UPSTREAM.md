# Upstream provenance

The default OpenMM interface is based on commit
`2ada654ecc7218409b66628f2a26c9deaba2f325` of Colvars PR #979.
The interface is GPL-3.0 under [LICENSE](LICENSE). The Colvars core retains its
existing license. [Pinned upstream sources](https://github.com/Colvars/colvars/tree/2ada654ecc7218409b66628f2a26c9deaba2f325/openmm)

| Area | Default build | Optional resident build |
| --- | --- | --- |
| OpenMM interface | PR force, proxy, Python wrapper and deferred CUDA callback in their original directories. Defaults to nm and kJ/mol. | Shares the public force and TABF integrator. Adds selected-atom transfers, backend diagnostics, selectable real units and adaptive-state export in `resident/`. |
| TABF | The dedicated integrator publishes the current physical-energy derivative before evaluating the bias and commits the requested alpha after the step. The PR's custom-integrator fallback remains available. | Uses the same integration operations with endpoint handling for report queries and paired restarts from earlier chinfo runs. |
| Alchemy | PR's AlchemicalPMEForce, CUDA and Reference implementations, SWIG bindings and Python FEP helpers. A local adapter calls the older ReferencePME C API for OpenMM 8.4. | Build the default interface for the PR's alchemical and FEP workflows. |
| Core and build | Uses the existing chinfo core. The OpenMM proxy holds the pending alpha locally. The build selects Tcl stubs according to the available core sources. | Also builds the chinfo CUDA atom groups and supported CV kernels. |
| Restart | PR file-prefix and restart frequency interface. | Adds paired OpenMM checkpoint and Colvars-state restore for the Chignolin production workflow. |

The dedicated-integrator derivative regression is tested with a constant
physical energy of 24 kJ/mol at alpha 2. The first derivative must be
−6 kJ/mol. The Chignolin pair-force test checks all particles against an
analytic harmonic bias, including the atom shared by two distances.
[Shared interface tests](tests/test_pr979_interface.py)

The earlier chinfo reviews and benchmarks predate the restoration of the PR
backend. [Historical review](REVIEW.md) and [historical validation](VALIDATION.md).
