"""Metadynamics restart fidelity and read-only serialization regressions."""

import math
import re

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from openmm_colvars import ColvarsForce


def configuration(*, grids, keep_hills=False, upper=2.0, step_zero=True, well_tempered=False):
    grid_options = f"gridsUpdateFrequency 100\n  keepHills {'on' if keep_hills else 'off'}" if grids else ""
    tempered = "wellTempered on\n  biasTemperature 1000" if well_tempered else ""
    return f"""
units gromacs
colvarsTrajFrequency 0
colvarsRestartFrequency 0
colvar {{
  name d
  width 0.1
  lowerBoundary 0
  upperBoundary {upper}
  distance {{
    group1 {{ atomNumbers 1 }}
    group2 {{ atomNumbers 2 }}
  }}
}}
metadynamics {{
  name meta
  colvars d
  hillWeight 2
  gaussianSigmas 0.1
  newHillFrequency 1
  stepZeroData {'on' if step_zero else 'off'}
  useGrids {'on' if grids else 'off'}
  {grid_options}
  {tempered}
}}
"""


def make_context(engine, prefix, config, restart=None):
    system = mm.System()
    system.addParticle(12.0)
    system.addParticle(24.0)
    force = ColvarsForce.fromConfigString(config, str(prefix))
    force.setUseGpu(engine.use_gpu)
    force.setUsesPeriodicBoundaryConditions(False)
    if restart is not None:
        force.setRestartInputPrefix(str(restart))
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(engine.platform), engine.properties)
    return context, integrator, force


def evaluate(context, distance, step=None):
    if step is not None:
        context.setStepCount(step)
    context.setPositions([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)))


def assert_same(left, right, engine):
    np.testing.assert_allclose(left[0], right[0], rtol=0, atol=engine.tolerances[1])
    np.testing.assert_allclose(left[1], right[1], rtol=0, atol=engine.tolerances[2])


@pytest.mark.parametrize("legacy", [False, True])
def test_projected_edge_hill_survives_expanded_grid_restart(engine, tmp_path, legacy):
    fixed = configuration(grids=True, upper=0.8)
    if legacy:
        fixed = fixed.replace("newHillFrequency 1", "newHillFrequency 100")
    expanding = fixed.replace("width 0.1", "width 0.1\n  expandBoundaries on")
    source, _, bias = make_context(engine, tmp_path/"fixed", fixed)
    evaluate(source, 0.76, 0)  # Project a hill at the original upper edge.
    first = tmp_path/"first.state"
    first.write_text(bias.getBiasState(source))

    grown, _, bias = make_context(engine, tmp_path/"grown", expanding, first)
    evaluate(grown, 0.76, 0)
    evaluate(grown, 0.82, 1)  # Move the upper boundary to 1.0.
    serialized = bias.getBiasState(grown)
    assert serialized.count("hill {") == (1 if legacy else 2)
    if legacy:
        # Pre-tail grid checkpoints store only projected hills.
        assert "unprojectedHills" not in serialized
        serialized = re.sub(r"lastHillStep \d+\n", "", serialized)
        serialized = re.sub(r"version \d{4}-\d{2}-\d{2}", "version 2024-06-01", serialized)
    checkpoint = tmp_path/"expanded.state"
    checkpoint.write_text(serialized)
    expected = evaluate(grown, 1.03)
    assert expected[0] > 0.04  # The projected boundary hill still acts outside.

    restored, _, bias = make_context(engine, tmp_path/"restored", expanding, checkpoint)
    evaluate(restored, 0.82, 1)
    assert_same(evaluate(restored, 1.03), expected, engine)
    assert bias.getBiasState(restored).count("hill {") == serialized.count("hill {")


def test_legacy_unknown_keep_hills_does_not_classify_every_hill_as_an_edge(engine, tmp_path):
    config = configuration(grids=True, keep_hills=True, upper=0.8)
    original, _, bias = make_context(engine, tmp_path/"keep-all", config)
    evaluate(original, 0.65, 0)  # Interior, but keepHills writes it explicitly.
    legacy = bias.getBiasState(original).replace("keepHills on\n", "")
    legacy = re.sub(r"lastHillStep \d+\n", "", legacy)
    legacy = re.sub(r"version \d{4}-\d{2}-\d{2}", "version 2020-06-01", legacy)
    assert legacy.count("hill {") == 1
    checkpoint = tmp_path/"legacy.state"
    checkpoint.write_text(legacy)
    restored, _, _ = make_context(engine, tmp_path/"legacy", config.replace("keepHills on", "keepHills off"),
                                  checkpoint)
    evaluate(restored, 0.65, 0)
    assert evaluate(restored, 0.82)[0] == pytest.approx(0.0, abs=engine.tolerances[1])


@pytest.mark.parametrize("legacy", [False, True])
def test_no_grid_restart_retains_analytic_hills(engine, tmp_path, legacy):
    config = configuration(grids=False, step_zero=False)
    original = make_context(engine, tmp_path/"original", config)
    context, _, force = original
    for step in (0, 1, 2):
        evaluate(context, 0.6, step)
    expected = evaluate(context, 0.63)
    gaussian = math.exp(-0.5*(0.03/0.1)**2)
    np.testing.assert_allclose(expected[0], 4.0*gaussian, rtol=0, atol=engine.tolerances[1])
    snapshot = force.getBiasState(context)
    assert snapshot.count("hill {") == 2
    assert "keepHills on" in snapshot
    if legacy:
        # States written before this fix omitted keepHills despite storing all
        # analytic hills. They must remain readable.
        snapshot = snapshot.replace("keepHills on\n", "")
    saved = tmp_path/"saved.colvars.state"
    saved.write_text(snapshot)
    restored = make_context(engine, tmp_path/"restored", config, saved)
    assert_same(evaluate(restored[0], 0.63, 2), expected, engine)
    assert restored[2].getBiasState(restored[0]).count("hill {") == 2
    assert_same(evaluate(restored[0], 0.67), evaluate(context, 0.67), engine)


@pytest.mark.parametrize("keep_hills", [False, True])
@pytest.mark.parametrize("center,probe,upper", [(0.6, 0.63, 2.0), (0.76, 0.83, 0.8)])
def test_grid_snapshot_preserves_pending_hills_and_forces(engine, tmp_path, keep_hills, center, probe, upper):
    config = configuration(grids=True, keep_hills=keep_hills, upper=upper)
    original = make_context(engine, tmp_path/"original", config)
    context, _, force = original
    # Step zero projects its hill. Steps one and two remain analytic until 100.
    for step in (0, 1, 2):
        evaluate(context, center, step)
    expected = evaluate(context, probe)
    if probe > upper:
        # Both projected boundary hills and pending hills are analytic outside
        # the grid. In particular, pending boundary hills must occur only once.
        energy = 6.0*math.exp(-0.5*((probe-center)/0.1)**2)
        np.testing.assert_allclose(expected[0], energy, rtol=0, atol=engine.tolerances[1])
        np.testing.assert_allclose(expected[1][0, 0], -energy*(probe-center)/0.1**2,
                                   rtol=0, atol=engine.tolerances[2])
    snapshot = force.getBiasState(context)
    assert "unprojectedHills 2" in snapshot
    assert snapshot == force.getBiasState(context)
    assert_same(evaluate(context, probe), expected, engine)
    saved = tmp_path/"saved.colvars.state"
    saved.write_text(snapshot)
    restored = make_context(engine, tmp_path/"restored", config, saved)
    assert_same(evaluate(restored[0], probe, 2), expected, engine)
    # Continue both histories and eventually project pending hills. Exact
    # agreement catches lost and duplicated hills as well as stale iterators.
    for step in range(3, 102):
        assert_same(evaluate(restored[0], probe, step), evaluate(context, probe, step), engine)
    assert_same(evaluate(restored[0], center), evaluate(context, center), engine)


@pytest.mark.parametrize("keep_hills", [False, True])
def test_legacy_fully_projected_grid_restart(engine, tmp_path, keep_hills):
    config = configuration(grids=True, keep_hills=keep_hills, upper=0.8)
    original = make_context(engine, tmp_path/"original", config)
    expected = evaluate(original[0], 0.76, 100)
    snapshot = original[2].getBiasState(original[0])
    assert "unprojectedHills" not in snapshot
    saved = tmp_path/"legacy.colvars.state"
    saved.write_text(snapshot)
    restored = make_context(engine, tmp_path/"restored", config, saved)
    assert_same(evaluate(restored[0], 0.76, 100), expected, engine)
    assert_same(evaluate(restored[0], 0.83), evaluate(original[0], 0.83), engine)


@pytest.mark.parametrize("grids", [False, True])
def test_well_tempered_height_includes_unprojected_bias(engine, tmp_path, grids):
    config = configuration(grids=grids, well_tempered=True)
    original = make_context(engine, tmp_path/"original", config)
    context, _, force = original
    energy = evaluate(context, 0.63, 0)[0]
    thermal = 0.00831446261815324*1000.0
    for step in (1, 2):
        # The new hill is centered on the current CV, hence contributes its
        # entire well-tempered height. Gridded older hills can be approximate,
        # but each pending hill's height must use the complete acting bias.
        expected = energy + 2.0*math.exp(-energy/thermal)
        energy = evaluate(context, 0.63, step)[0]
        np.testing.assert_allclose(energy, expected, rtol=0, atol=engine.tolerances[1])
    before = evaluate(context, 0.66)
    snapshot = force.getBiasState(context)
    assert_same(evaluate(context, 0.66), before, engine)
    saved = tmp_path/"well_tempered.colvars.state"
    saved.write_text(snapshot)
    restored = make_context(engine, tmp_path/"restored", config, saved)
    assert_same(evaluate(restored[0], 0.66, 2), before, engine)
    assert_same(evaluate(restored[0], 0.66, 3), evaluate(context, 0.66, 3), engine)


def test_frequent_pmf_output_includes_pending_hills_without_projecting(engine, tmp_path):
    base_config = configuration(grids=True)
    frequent_config = base_config.replace("colvarsRestartFrequency 0", "colvarsRestartFrequency 1")
    frequent_config = frequent_config.replace("newHillFrequency 1", "newHillFrequency 1\n  outputFreq 1")
    frequent = make_context(engine, tmp_path/"frequent", frequent_config)
    silent = make_context(engine, tmp_path/"silent", base_config)
    for step, center in ((0, 0.65), (1, 0.63)):
        assert_same(evaluate(frequent[0], center, step), evaluate(silent[0], center, step), engine)
    assert "unprojectedHills 1" in frequent[2].getBiasState(frequent[0])
    assert_same(evaluate(frequent[0], 0.67), evaluate(silent[0], 0.67), engine)

    table = np.loadtxt(tmp_path/"frequent.pmf")
    positions, pmf = table[:, 0], table[:, 1]
    potential = sum(2.0*np.exp(-0.5*((positions-center)/0.1)**2) for center in (0.65, 0.63))
    expected_pmf = potential.max()-potential
    # Compare bins close to the hills, away from Colvars' Gaussian truncation.
    selected = np.abs(positions-0.65) < 0.3
    np.testing.assert_allclose(pmf[selected], expected_pmf[selected], rtol=0,
                               atol=max(1.0e-6, engine.tolerances[1]))
