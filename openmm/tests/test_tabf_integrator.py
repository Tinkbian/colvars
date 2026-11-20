"""Analytical force splitting and adaptive history for coupled T+eABF dynamics."""

import re

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from openmm_colvars import ColvarsForce, ColvarsTABFIntegrator
from conftest import Engine


DT = 0.002  # ps
ALPHA = 2.0
MASS = 12.0  # Da
P0, P1 = 0.3, 1.102  # nm
PHYSICAL, BIAS = 3, 6


def alpha_config(units, mass):
    return f"""
units {units}
colvarsTrajFrequency 0
colvarsRestartFrequency 0
colvar {{
  name a
  width 0.05
  lowerBoundary 1
  upperBoundary 4
  extendedLagrangian on
  extendedMass {mass}
  extendedLangevinDamping 0
  tabfAlpha {{ }}
}}
"""


def make_tabf(engine, prefix, config, units="gromacs", state_path=None, box_length=None):
    system = mm.System()
    system.addParticle(MASS)
    system.addParticle(MASS)
    physical = mm.CustomExternalForce("24 + 12*x")
    physical.addParticle(0, [])
    physical.setForceGroup(PHYSICAL)
    system.addForce(physical)

    bias = ColvarsForce.fromConfigString(config, str(prefix))
    bias.setUnitSystem(units)
    bias.setUseGpu(engine.use_gpu)
    bias.setUsesPeriodicBoundaryConditions(box_length is not None)
    bias.setTargetTemperature(300)
    bias.setRandomSeed(709)
    bias.setForceGroup(BIAS)
    if state_path is not None:
        bias.setRestartInputPrefix(str(state_path))
    system.addForce(bias)
    if box_length is not None:
        system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*row) for row in box_length*np.eye(3)))
    integrator = ColvarsTABFIntegrator(DT, 300, 0, ALPHA, PHYSICAL, BIAS)
    context = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName(engine.platform), engine.properties)
    context.setPositions([[P0, 0, 0], [P1+(box_length or 0.0), 0, 0]])
    context.setVelocities([[0, 0, 0], [0, 0, 0]])
    return context, integrator, bias


def kinematics(context):
    state = context.getState(getPositions=True, getVelocities=True)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
    return x, v


@pytest.mark.parametrize("units", ["gromacs", "real"])
def test_tabf_force_split_and_energy_derivative(engine, tmp_path, units):
    # The real-unit input describes exactly the same spring and fictitious mass.
    length = 10.0 if units == "real" else 1.0
    energy = 4.184 if units == "real" else 1.0
    alpha_mass = 500.0  # kJ/mol fs^2
    config = alpha_config(units, alpha_mass/energy) + f"""
colvar {{
  name d
  width {length}
  distance {{
    group1 {{ atomNumbers 1 }}
    group2 {{ atomNumbers 2 }}
  }}
}}
harmonic {{
  colvars d
  centers {0.5*length}
  forceConstant {8.0/energy}
}}
abf {{
  colvars a
  applyBias off
  stepZeroData on
}}
"""
    box_length = 2.0
    context, integrator, bias = make_tabf(engine, tmp_path / "split", config, units,
                                          box_length=box_length)
    x = np.array([P0, P1+box_length])
    v = np.zeros(2)
    alpha, alpha_v = ALPHA, 0.0
    derivative_sum = 0.0
    dt_fs = 1000*DT
    for step in (1, 2):
        # f0 is divided by the pre-step alpha. The bias is a full impulse at x(t+dt).
        half_kick = 0.5*DT*(-12.0)/(MASS*alpha)
        v[0] += half_kick
        x += DT*v
        v[0] += half_kick
        bias_force = 8.0*(x[1]-x[0]-box_length-0.5)
        v += DT*bias_force*np.array([1, -1])/MASS
        derivative = -(24.0+12.0*x[0])/alpha**2
        derivative_sum += derivative
        alpha_v -= dt_fs*derivative/alpha_mass
        alpha += dt_fs*alpha_v

        integrator.step(1)
        actual_x, actual_v = kinematics(context)
        tol = max(engine.tolerances[0], 1.e-10)
        np.testing.assert_allclose(actual_x[:, 0], x, rtol=0, atol=tol)
        np.testing.assert_allclose(actual_v[:, 0], v, rtol=0, atol=tol)
        assert integrator.getAlphaDerivative() == pytest.approx(derivative, abs=engine.tolerances[1], rel=0)
        assert integrator.getAlpha() == pytest.approx(alpha, abs=max(tol, 1.e-9), rel=0)
        assert bias.getBiasUpdateCount(context) == step
        state = bias.getBiasState(context)
        counts, = abf_arrays(state, "samples")
        averages, = abf_arrays(state, "gradient")
        assert counts.sum() == step
        # State grids contain the mean dE/dalpha in the selected Colvars units.
        assert energy*np.dot(counts, averages) == pytest.approx(derivative_sum, abs=2.e-4, rel=0)


ADAPTIVE = alpha_config("gromacs", 50000.0) + """
colvar {
  name d
  width 0.005
  lowerBoundary 0.7
  upperBoundary 0.9
  extendedLagrangian on
  extendedFluctuation 0.014
  extendedTimeConstant 24
  extendedLangevinDamping 0
  distance {
    group1 { atomNumbers 1 }
    group2 { atomNumbers 2 }
  }
}
abf {
  name tabf
  colvars a
  fullSamples 2
}
abf {
  name eabf
  colvars d
  fullSamples 2
}
metadynamics {
  name meta
  colvars d
  newHillFrequency 2
  hillWeight 0.2
  gaussianSigmas 0.01
  useGrids off
}
"""


def abf_arrays(snapshot, key):
    # A Colvars state stores each grid as whitespace-separated rows after its key.
    pattern = rf"^{key}[ \t]*\r?\n((?:[ \t]+[-+.eE\d][-.+ eE\d\t]*\r?\n)+)"
    return [np.fromstring(grid, sep=" ") for grid in re.findall(pattern, snapshot, re.M)]


@pytest.mark.parametrize("frequency", [0, 3])
def test_tabf_multidimensional_abf_projection(tmp_path, frequency):
    engine = Engine("Reference", False, "double")
    config = ADAPTIVE.split("abf {", 1)[0] + f"""
abf {{
  colvars a d
  stepZeroData on
  pABFintegrateFreq {frequency}
}}
"""
    if frequency:
        with pytest.raises(Exception, match="Projected ABF is unavailable"):
            make_tabf(engine, tmp_path / "projected", config)
    else:
        context, integrator, bias = make_tabf(engine, tmp_path / "unprojected", config)
        integrator.step(2)
        counts, = abf_arrays(bias.getBiasState(context), "samples")
        assert counts.sum() == 2


def reporter_queries(context):
    context.getState(getEnergy=True)
    context.getState(getForces=True, groups={BIAS})
    context.getState(getEnergy=True, getForces=True, groups={BIAS})


@pytest.mark.parametrize("problem, message", [
    ("missing", "tabfAlpha driver"), ("duplicate", "tabfAlpha driver"),
    ("mass", "positive finite mass"), ("bounds", "Initial TABF alpha"),
])
def test_tabf_invalid_driver(tmp_path, problem, message):
    # Reference exercises the common validation without repeatedly creating invalid CUDA Contexts.
    engine = Engine("Reference", False, "double")
    config = alpha_config("gromacs", 0.0 if problem == "mass" else 500.0)
    if problem == "missing":
        config = """colvar {
 name d
 distance {
   group1 { atomNumbers 1 }
   group2 { atomNumbers 2 }
 }
}
"""
    elif problem == "duplicate":
        config += config[config.index("colvar {"):].replace("name a", "name another_alpha")
    elif problem == "bounds":
        config = config.replace("lowerBoundary 1", "lowerBoundary 3")
    with pytest.raises(Exception, match=message):
        context, integrator, bias = make_tabf(engine, tmp_path / "invalid", config)
        reporter_queries(context)


@pytest.mark.parametrize("problem,message", [("clock", "endpoint step"), ("alpha", "alpha differs")])
def test_tabf_mismatched_checkpoint(tmp_path, problem, message):
    engine = Engine("Reference", False, "double")
    config = alpha_config("gromacs", 500.0)
    source, first, bias = make_tabf(engine, tmp_path / "source", config)
    first.step(2)
    checkpoint = source.createCheckpoint()
    state_file = tmp_path / "endpoint.state"
    state_file.write_text(bias.getBiasState(source))
    del source, first
    restored, second, bias2 = make_tabf(engine, tmp_path / "restored", config,
                                        state_path=state_file)
    restored.loadCheckpoint(checkpoint)
    if problem == "clock":
        restored.setStepCount(1)
    else:
        second.setGlobalVariableByName("tabf_alpha", ALPHA)
    with pytest.raises(Exception, match=message):
        reporter_queries(restored)
    assert bias2.getBiasUpdateCount(restored) == 0


@pytest.mark.parametrize("damping", [0, 15])
def test_tabf_reporters_and_paired_restart(engine, tmp_path, damping):
    # Thermostat just the distance. Five steps place the checkpoint after an odd
    # number of normal variates, exercising the distribution's cached sample.
    config = ADAPTIVE.replace("extendedTimeConstant 24\n  extendedLangevinDamping 0",
                              f"extendedTimeConstant 24\n  extendedLangevinDamping {damping}")
    reference, first, bias = make_tabf(engine, tmp_path / "continuous", config)
    reference.setVelocities([[0, 0, 0], [0.8, 0, 0]])
    reporter_queries(reference)
    assert bias.getBiasUpdateCount(reference) == 0
    accumulated_distance_gradient = 0.0
    for _ in range(5):
        first.step(1)
        positions, _ = kinematics(reference)
        distance = np.linalg.norm(positions[1]-positions[0])
        extended = bias.getColvarValue(reference, "d")
        spring = 0.00831446261815324*300/0.014**2
        accumulated_distance_gradient += spring*(extended-distance)
    checkpoint = reference.createCheckpoint()
    saved = bias.getBiasState(reference)
    # The CZAR grid samples the same spring gradient in bins of the actual CV.
    counts, = abf_arrays(saved, "z_samples")
    gradient, = abf_arrays(saved, "z_gradient")
    assert np.dot(counts, gradient) == pytest.approx(accumulated_distance_gradient, abs=0.01, rel=0)
    state_file = tmp_path / "paired.colvars.state"
    state_file.write_text(saved)
    reporter_queries(reference)
    assert bias.getBiasState(reference) == saved

    # Continue the original without reporters. Recreate a new Context from both
    # halves of the checkpoint; reporters run at every restored integration boundary.
    first.step(6)
    final = bias.getBiasState(reference)
    x_expected, v_expected = kinematics(reference)
    alpha_expected = first.getAlpha()
    del reference, first

    restored, second, bias2 = make_tabf(engine, tmp_path / "restored", config,
                                        state_path=state_file)
    restored.loadCheckpoint(checkpoint)
    for i in range(6):
        reporter_queries(restored)
        assert bias2.getBiasUpdateCount(restored) == i
        second.step(1)
    reporter_queries(restored)
    actual = bias2.getBiasState(restored)
    assert bias2.getBiasUpdateCount(restored) == 6
    for key, n_biases in [("samples", 2), ("z_samples", 1)]:
        starting, expected, ending = (abf_arrays(state, key) for state in (saved, final, actual))
        assert len(starting) == len(expected) == len(ending) == n_biases
        for before, after, observed in zip(starting, expected, ending):
            assert before.sum() > 0
            assert after.sum() == before.sum()+6
            np.testing.assert_array_equal(observed, after)
    for key, n_biases in [("gradient", 2), ("z_gradient", 1)]:
        baseline, resumed = abf_arrays(final, key), abf_arrays(actual, key)
        assert len(baseline) == len(resumed) == n_biases
        for expected, observed in zip(baseline, resumed):
            np.testing.assert_allclose(observed, expected, atol=2.e-5, rtol=2.e-5)
    hill_step = r"\blastHillStep\s+(\d+)"
    assert re.search(hill_step, final)[1] == re.search(hill_step, actual)[1]
    assert int(re.search(hill_step, actual)[1]) > int(re.search(hill_step, saved)[1])
    x, v = kinematics(restored)
    tol = max(engine.tolerances[0], 5.e-9)
    np.testing.assert_allclose(x, x_expected, rtol=0, atol=tol)
    np.testing.assert_allclose(v, v_expected, rtol=0, atol=max(tol, 1.e-7))
    assert second.getAlpha() == pytest.approx(alpha_expected, abs=tol, rel=0)


def test_tabf_reported_potential_matches_atomic_force(engine, tmp_path):
    context, integrator, bias = make_tabf(engine, tmp_path / "force-gradient", ADAPTIVE)
    context.setVelocities([[0, 0, 0], [0.8, 0, 0]])
    integrator.step(4)
    positions, _ = kinematics(context)
    reported = context.getState(getForces=True, groups={BIAS}).getForces(asNumpy=True)
    force = reported.value_in_unit(unit.kilojoule_per_mole/unit.nanometer)[1, 0]
    history = bias.getBiasState(context)

    # Hold auxiliary coordinates and momenta fixed while varying one atom.
    # Including the coordinate-dependent midpoint estimate of the auxiliary
    # kinetic energy in group 1 would give a spurious atomic force here.
    delta = 0.001
    energies = []
    for sign in (-1, 1):
        trial = positions.copy()
        trial[1, 0] += sign*delta
        context.setPositions(trial)
        state = context.getState(getEnergy=True, groups={BIAS})
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    gradient_force = -(energies[1] - energies[0])/(2*delta)
    assert gradient_force == pytest.approx(force, abs=max(1.e-8, 5*engine.tolerances[2]), rel=0)
    assert bias.getBiasState(context) == history


def test_tabf_unapplied_abf_only_samples(engine, tmp_path):
    config = alpha_config("gromacs", 1.e6) + """
abf {
  colvars a
  applyBias off
  fullSamples 2
  stepZeroData on
}
"""
    context, integrator, bias = make_tabf(engine, tmp_path / "sample-only", config)
    integrator.step(6)
    counts, = abf_arrays(bias.getBiasState(context), "samples")
    assert counts.max() == 6
    energy = context.getState(getEnergy=True, groups={BIAS}).getPotentialEnergy()
    assert energy.value_in_unit(unit.kilojoule_per_mole) == 0.0


def test_tabf_seed_grids_without_pmf_integration(tmp_path):
    engine = Engine("Reference", False, "double")
    config = alpha_config("gromacs", 1.e6) + """
abf {
  colvars a
  integrate off
  stepZeroData on
}
"""
    prefix = tmp_path / "training"
    original, first, bias = make_tabf(engine, prefix, config)
    first.step(4)
    before = bias.getBiasState(original)
    bias.flushOutputs(original)
    del original, first

    seeded_config = config.replace("integrate off",
                                   f"integrate off\n  updateBias off\n  inputPrefix {{{prefix.as_posix()}}}")
    seeded, second, bias = make_tabf(engine, tmp_path / "seeded", seeded_config)
    second.step(1)
    after = bias.getBiasState(seeded)
    for field in ("samples", "gradient"):
        expected, = abf_arrays(before, field)
        actual, = abf_arrays(after, field)
        np.testing.assert_allclose(actual, expected, atol=1.e-7, rtol=1.e-7)


def test_tabf_abf_energy_at_grid_edges(tmp_path):
    engine = Engine("Reference", False, "double")
    prefix = tmp_path / "seed"
    for suffix, value in [(".count", 10), (".grad", 2)]:
        prefix.with_suffix(suffix).write_text(f"# 1\n# 0.7 0.1 2 0\n\n0.75 {value}\n0.85 {value}\n")
    config = alpha_config("gromacs", 1.e6) + f"""
colvar {{
  name d
  width 0.1
  lowerBoundary 0.7
  upperBoundary 0.9
  distance {{
    group1 {{ atomNumbers 1 }}
    group2 {{ atomNumbers 2 }}
  }}
}}
abf {{
  colvars d
  fullSamples 1
  updateBias off
  inputPrefix {{{prefix.as_posix()}}}
}}
"""
    context, integrator, bias = make_tabf(engine, tmp_path / "edges", config)
    for distance in (0.67, 0.72, 0.88, 0.91, 0.94):
        context.setPositions([[0, 0, 0], [distance, 0, 0]])
        state = context.getState(getEnergy=True, getForces=True, groups={BIAS})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        force = state.getForces()[1][0].value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        assert energy == pytest.approx(-2 * (min(0.9, max(0.7, distance)) - 0.7), abs=1.e-12)
        assert force == pytest.approx(2.0 if 0.7 < distance < 0.9 else 0.0, abs=1.e-12)


def test_tabf_detects_changed_live_alpha_before_motion(engine, tmp_path):
    context, integrator, bias = make_tabf(engine, tmp_path / "live", ADAPTIVE)
    integrator.step(4)
    before = kinematics(context)
    alpha = integrator.getAlpha()
    integrator.setGlobalVariableByName("tabf_alpha", 3.5)
    with pytest.raises(Exception, match="TABF alpha differs"):
        integrator.step(1)
    assert context.getStepCount() == bias.getBiasUpdateCount(context) == 4
    for observed, expected in zip(kinematics(context), before):
        np.testing.assert_array_equal(observed, expected)
    integrator.setGlobalVariableByName("tabf_alpha", alpha)
    integrator.step(1)
    assert bias.getBiasUpdateCount(context) == 5
