# OpenMM interface for Colvars

The default build follows [Colvars PR #979](https://github.com/Colvars/colvars/pull/979).
It provides `ColvarsForce`, `ColvarsTABFIntegrator`, `AlchemicalPMEForce` and
the Python `openmm_fep` package. Colvars input uses GROMACS units (nm and
kJ/mol). On CUDA, the default Colvars proxy reads coordinates to the host and
returns atomic bias forces through the PR's deferred callback.

## Build the PR interface

From the repository root, activate the OpenMM environment and select the
Lepton source from the matching OpenMM release. The build requires a C++17
compiler, CMake, Python development headers and SWIG compatible with OpenMM.

```bash
cmake -S openmm -B build-pr979 -G Ninja \
  -DOPENMM_ROOT="$CONDA_PREFIX" -DPython_EXECUTABLE="$(command -v python)" \
  -DLEPTON_DIR=/path/to/openmm/libraries/lepton \
  -DOPENMM_COLVARS_PYTHON_INSTALL_DIR="$PWD/install-pr979"
cmake --build build-pr979 --parallel
cmake --install build-pr979
PYTHONPATH="$PWD/install-pr979" python -m pytest -q openmm/tests
```

For the PR's CUDA implementation, add `-DOPENMM_COLVARS_ENABLE_CUDA_KERNELS=ON`
and supply the CUDA toolkit headers and the matching OpenMM CUDA plugin. For
conda layouts where discovery needs a hint, pass `-DOPENMM_CUDA_LIBRARY=...`
and `-DOPENMM_COLVARS_CUDA_TOOLKIT_INCLUDE_DIR=...`. To exercise the GPU, add
`--openmm-platform Reference,CPU,CUDA --require-cuda` to the test command.

## Chignolin T+eABF with selected-atom CUDA transfers

Use a separate build directory and installation path for
`-DOPENMM_COLVARS_BACKEND=resident -DOPENMM_COLVARS_ENABLE_CUDA=ON`.
Both builds use the same public force configuration and TABF stepping
interface. The resident backend also accepts the original Chignolin input in
Å and kcal/mol with `force.setUnitSystem("real")`. Its CUDA bridge evaluates
supported distance components on device. Extended alpha, ABF and metadynamics
history are updated on the host. It provides matching OpenMM and Colvars
restart data for continuing this workflow. Select the installed module for
each simulation with `PYTHONPATH`.

See [resident build and restart instructions](RESIDENT.md) and the
[source comparison](UPSTREAM.md).
