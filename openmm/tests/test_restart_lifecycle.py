"""A loaded adaptive step must be evaluated without repeating its update."""

import multiprocessing
import re

import numpy as np
import openmm as mm
import pytest

from openmm_colvars import ColvarsForce
from test_openmm_colvars import state_values


CONFIG = """
units gromacs
colvarsTrajFrequency 0
colvarsRestartFrequency 0
colvar {
  name d
  width 0.1
  lowerBoundary 0
  upperBoundary 2
  distance {
    group1 { atomNumbers 1 }
    group2 { atomNumbers 2 }
  }
}
metadynamics {
  name meta
  colvars d
  hillWeight 2
  gaussianSigmas 0.1
  newHillFrequency 1
  stepZeroData on
  useGrids on
  gridsUpdateFrequency 1
}
"""


def context_with_restart(engine, output, state_file=None, load_source="prefix", config=CONFIG):
    if state_file is not None and load_source == "configuration":
        config += f"\ndefaultInputStateFile {state_file}\n"
    force = ColvarsForce.fromConfigString(config, str(output))
    force.setUseGpu(engine.use_gpu)
    force.setUsesPeriodicBoundaryConditions(False)
    if state_file is not None and load_source == "prefix":
        force.setRestartInputPrefix(str(state_file))
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(24.0)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName(engine.platform), engine.properties)
    context.setPositions([[0, 0, 0], [0.65, 0, 0]])
    return context, integrator, force


def assert_same_state(actual, expected, engine):
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=engine.tolerances[1])
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=engine.tolerances[2])


def test_unevaluated_state_cannot_be_saved(engine, tmp_path):
    context, integrator, force = context_with_restart(engine, tmp_path / "initial")
    with pytest.raises(RuntimeError, match="Evaluate energy or forces"):
        force.getBiasState(context)
    assert force.getBiasUpdateCount(context) == 0
    assert state_values(context)[0] == pytest.approx(2.0, abs=engine.tolerances[1], rel=0)
    assert "lastHillStep 0" in force.getBiasState(context)


def check_incomplete_restart(engine, tmp_path, missing):
    original, integrator, force = context_with_restart(engine, tmp_path / "original")
    expected = state_values(original)
    snapshot = force.getBiasState(original)
    header, objects = snapshot.split("colvar {", 1)
    colvar_block, bias = ("colvar {" + objects).split("metadynamics {", 1)
    missing_step = "Restart requires a configuration block with a saved step"
    missing_cv = "Colvars restart contains no matching collective variable"
    contents, error = {
        "empty": ("", missing_step),
        "step": (colvar_block + "metadynamics {" + bias, missing_step),
        "objects": (header, missing_cv),
        "bias": (header + colvar_block, "Colvars restart is missing adaptive bias meta"),
        "colvar": (header + "metadynamics {" + bias, missing_cv),
        "truncated": (snapshot.rstrip()[:-1],
                      'corrupt restart information for "metadynamics" bias "meta": no matching brace'),
    }[missing]
    state_file = tmp_path / "invalid.state"
    state_file.write_text(contents)
    with pytest.raises(mm.OpenMMException, match=re.escape(error)):
        context_with_restart(engine, tmp_path / "restored", state_file)
    assert_same_state(state_values(original), expected, engine)
    assert force.getBiasUpdateCount(original) == 1


@pytest.mark.parametrize("missing", ["empty", "step", "objects", "bias", "colvar", "truncated"])
def test_incomplete_restart_cannot_suppress_first_update(engine, tmp_path, missing):
    if engine.platform != "CUDA":
        check_incomplete_restart(engine, tmp_path, missing)
        return
    # OpenMM 8.6.1 retains CUDA allocations when Context construction throws.
    # A separate native process releases those allocations after each rejection.
    process = multiprocessing.get_context("spawn").Process(
        target=check_incomplete_restart, args=(engine, tmp_path, missing))
    process.start()
    process.join(120)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("CUDA restart rejection timed out")
    assert process.exitcode == 0


def test_restart_allows_new_unbiased_colvar(engine, tmp_path):
    original, integrator, force = context_with_restart(engine, tmp_path / "original")
    state_values(original)
    state_file = tmp_path / "valid.state"
    state_file.write_text(force.getBiasState(original).rstrip())
    extra = CONFIG[CONFIG.index("colvar {"):CONFIG.index("metadynamics {")].replace("name d", "name other")
    restored, integrator2, force2 = context_with_restart(
        engine, tmp_path / "restored", state_file, config=CONFIG + extra)
    assert state_values(restored)[0] == pytest.approx(2.0, abs=engine.tolerances[1])
    assert force2.getBiasUpdateCount(restored) == 0


def test_invalid_clock_does_not_commit_an_adaptive_step(engine, tmp_path):
    original, integrator, force = context_with_restart(engine, tmp_path / "original")
    original.setStepCount(2)
    expected = state_values(original)
    saved = force.getBiasState(original)
    state_file = tmp_path / "clock.state"
    state_file.write_text(saved)
    context, integrator2, force2 = context_with_restart(engine, tmp_path / "restored", state_file)
    assert_same_state(state_values(context), expected, engine)

    context.setStepCount(2**63 - 1)
    with pytest.raises(mm.OpenMMException, match="Colvars step range"):
        state_values(context)
    context.setStepCount(1)
    integrator2.setStepSize(float("nan"))
    with pytest.raises(mm.OpenMMException, match="finite integration timestep"):
        state_values(context)
    assert force2.getBiasState(context) == saved
    assert force2.getBiasUpdateCount(context) == 0
    integrator2.setStepSize(0.001)
    assert state_values(context)[0] == pytest.approx(4.0, abs=engine.tolerances[1], rel=0)
    assert force2.getBiasUpdateCount(context) == 1


@pytest.mark.parametrize("load_source", ["prefix", "configuration"])
def test_restored_step_is_not_deposited_twice(engine, tmp_path, load_source):
    original, integrator, force = context_with_restart(engine, tmp_path / "original")
    integrator.setStepSize(0.002)
    for step in range(3):
        original.setStepCount(step)
        expected = state_values(original)
    np.testing.assert_allclose(expected[0], 6.0, rtol=0, atol=engine.tolerances[1])
    state_file = tmp_path / "续算-Δ.colvars.state"
    state_file.write_text(force.getBiasState(original))

    restored, restored_integrator, restored_force = context_with_restart(
        engine, tmp_path / "restored", state_file, load_source)
    assert restored_force.getBiasState(restored) == state_file.read_text()
    # The caller may continue with an independently restored OpenMM step count.
    # The initial query still represents the saved Colvars step (2).
    restored.setStepCount(37)
    assert_same_state(state_values(restored), expected, engine)
    assert_same_state(state_values(restored), expected, engine)
    assert restored_force.getBiasUpdateCount(restored) == 0
    snapshot = restored_force.getBiasState(restored)
    assert int(re.search(r"configuration\s*\{\s*step\s+(\d+)", snapshot)[1]) == 2

    # This change exceeds the restart geometry-check threshold. Only the first
    # loaded geometry is checked; arbitrary later same-step queries are valid.
    changed_positions = [[0, 0, 0], [0.85, 0, 0]]
    original.setPositions(changed_positions)
    restored.setPositions(changed_positions)
    expected = state_values(original)
    assert_same_state(state_values(restored), expected, engine)
    assert restored_force.getBiasUpdateCount(restored) == 0

    original.setStepCount(3)
    restored.setStepCount(38)
    expected = state_values(original)
    assert_same_state(state_values(restored), expected, engine)
    assert_same_state(state_values(restored), expected, engine)
    assert restored_force.getBiasUpdateCount(restored) == 1
    snapshot = restored_force.getBiasState(restored)
    assert int(re.search(r"configuration\s*\{\s*step\s+(\d+)", snapshot)[1]) == 3
    del restored, restored_integrator, original, integrator
