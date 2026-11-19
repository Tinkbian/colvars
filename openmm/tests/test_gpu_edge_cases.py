"""Analytical regressions also exercised by the CUDA GPU and CPU-fallback modes."""

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from openmm_colvars import ColvarsForce

from test_openmm_colvars import make_context, state_values


@pytest.mark.parametrize("exponent", [0, 1])
def test_signed_distance_polynomial_at_zero(engine, tmp_path, exponent):
    config = f"""units gromacs
colvar {{
  name height
  width 1
  distanceZ {{
    componentCoeff 2
    componentExp {exponent}
    main {{ atomNumbers 1 }}
    ref {{ atomNumbers 2 }}
    axis (0,0,1)
  }}
}}
harmonic {{
  colvars height
  centers 1
  forceConstant 2
}}
"""
    context, integrator, force = make_context(engine, tmp_path, config)
    positions = np.zeros((6, 3))
    positions[0, 0] = 0.3
    # The reciprocal of the tiny positive projection overflows in double precision.
    for z in (0.25, 1.0e-309, 0.0, -0.25):
        positions[0, 2] = z
        context.setPositions(positions)
        energy, forces = state_values(context)
        expected_cv = 2.0 if exponent == 0 else 2.0 * z
        fz = 0.0 if exponent == 0 else 4.0 * (1.0 - expected_cv)
        expected_forces = np.zeros_like(positions)
        expected_forces[:2, 2] = [fz, -fz]
        np.testing.assert_allclose(force.getColvarValue(context, "height"), expected_cv,
                                   rtol=0, atol=engine.tolerances[0])
        np.testing.assert_allclose(energy, (expected_cv - 1.0)**2, rtol=0, atol=engine.tolerances[1])
        np.testing.assert_allclose(forces, expected_forces, rtol=0, atol=engine.tolerances[2])


@pytest.mark.parametrize("component,groups", [
    ("distanceXY", "main { atomNumbers 1 }\n    ref { atomNumbers 2 }\n    axis (0,0,1)"),
    ("distance", "group1 { atomNumbers 1 }\n    group2 { atomNumbers 2 }"),
], ids=["on-axis", "coincident"])
def test_zero_radius_after_nonzero_gradient(engine, tmp_path, component, groups):
    config = f"""units gromacs
colvarsTrajFrequency 0
colvarsRestartFrequency 0
colvar {{
  name radius
  width 1.0
  {component} {{
    {groups}
  }}
}}
harmonic {{
  colvars radius
  centers 1.0
  forceConstant 2.0
}}
"""
    context, integrator, force = make_context(engine, tmp_path, config)
    positions = np.zeros((6, 3))
    for x, fx in ((0.5, 1.0), (0.0, 0.0), (-0.5, -1.0)):
        positions[0] = [x, 0.0, 0.3 if component == "distanceXY" else 0.0]
        context.setPositions(positions)
        energy, forces = state_values(context)
        expected_forces = np.zeros_like(positions)
        expected_forces[:2, 0] = [fx, -fx]
        np.testing.assert_allclose(force.getColvarValue(context, "radius"), abs(x),
                                   rtol=0, atol=engine.tolerances[0])
        np.testing.assert_allclose(energy, (abs(x) - 1.0)**2, rtol=0, atol=engine.tolerances[1])
        np.testing.assert_allclose(forces, expected_forces, rtol=0, atol=engine.tolerances[2])


def test_overlapping_centers_across_reduction_blocks(engine, tmp_path):
    # Different group sizes and overlap distinguish group-array strides,
    # proxy-array strides, and the original particle indices.
    n = 1031
    first = np.arange(0, n, 2)
    second = np.arange(97, n, 7)
    # Include massless particles in both groups, including their overlap.
    masses = (np.arange(n) % 17).astype(float)
    coords = np.random.default_rng(51).uniform(0.1, 0.8, (n, 3))
    def ids(group):
        return " ".join(str(i + 1) for i in group)

    config = f"""units gromacs
colvar {{
  name d
  width 1
  distance {{
    group1 {{ atomNumbers {ids(first)} }}
    group2 {{ atomNumbers {ids(second)} }}
  }}
}}
harmonic {{
  colvars d
  centers 0
  forceConstant 2
}}
"""
    system = mm.System()
    for m in masses:
        system.addParticle(m)
    force = ColvarsForce.fromConfigString(config, str(tmp_path / "overlap"))
    force.setUseGpu(engine.use_gpu)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(engine.platform),
                         engine.properties)
    for shift in (0.0, 0.24):
        coords[first, 0] += shift
        context.setPositions(coords)
        delta = (np.average(coords[second], axis=0, weights=masses[second]) -
                 np.average(coords[first], axis=0, weights=masses[first]))
        expected_forces = np.zeros_like(coords)
        expected_forces[first] += 2 * delta * (masses[first] / masses[first].sum())[:, None]
        expected_forces[second] -= 2 * delta * (masses[second] / masses[second].sum())[:, None]
        state = context.getState(getEnergy=True, getForces=True)
        np.testing.assert_allclose(force.getColvarValue(context, "d"), np.linalg.norm(delta),
                                   atol=engine.tolerances[0], rtol=0)
        np.testing.assert_allclose(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                                   delta @ delta, atol=engine.tolerances[1], rtol=0)
        np.testing.assert_allclose(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
                                   expected_forces, atol=engine.tolerances[2], rtol=0)


@pytest.mark.parametrize("dummy_first", [False, True])
def test_distance_to_dummy_group(engine, tmp_path, dummy_first):
    # Only particle 2 is selected, so the dummy group must not accidentally
    # create an atom slot or alter the force on the unrelated first particle.
    dummy = np.array([0.1, 0.2, 0.3])
    groups = ["atomNumbers 2", "dummyAtom (0.1, 0.2, 0.3)"]
    if dummy_first:
        groups.reverse()
    config = f"""
units gromacs
colvarsTrajFrequency 0
colvarsRestartFrequency 0
colvar {{
  name d
  width 0.5
  distance {{
    group1 {{ {groups[0]} }}
    group2 {{ {groups[1]} }}
  }}
}}
harmonic {{
  colvars d
  centers 0.4
  forceConstant 10.0
}}
"""
    system = mm.System()
    system.addParticle(7.0)
    system.addParticle(12.0)
    force = ColvarsForce.fromConfigString(config, str(tmp_path / "dummy"))
    force.setUseGpu(engine.use_gpu)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName(engine.platform), engine.properties)
    for selected_position in ([0.7, 0.8, 0.9], [0.8, -0.1, 0.7]):
        context.setPositions([[9.0, 8.0, 7.0], selected_position])
        displacement = np.asarray(selected_position) - dummy
        distance = float(np.linalg.norm(displacement))
        spring = 10.0 / 0.5**2
        expected_energy = 0.5 * spring * (distance - 0.4)**2
        expected_force = -spring * (distance - 0.4) * displacement / distance
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        observed = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        np.testing.assert_allclose(force.getColvarValue(context, "d"), distance,
                                   atol=engine.tolerances[0], rtol=0)
        np.testing.assert_allclose(energy, expected_energy,
                                   atol=engine.tolerances[1], rtol=0)
        np.testing.assert_allclose(observed[1], expected_force,
                                   atol=engine.tolerances[2], rtol=0)
        np.testing.assert_array_equal(observed[0], 0.0)
    assert force.getBiasUpdateCount(context) == 1
    del context, integrator


def test_only_dummy_groups_request_no_atoms(engine, tmp_path):
    config = """
units gromacs
colvarsTrajFrequency 0
colvarsRestartFrequency 0
colvar {
  name d
  width 0.5
  distance {
    group1 { dummyAtom (0.1, 0.2, 0.3) }
    group2 { dummyAtom (0.8, -0.1, 0.7) }
  }
}
harmonic {
  colvars d
  centers 0.4
  forceConstant 10.0
}
"""
    system = mm.System()
    system.addParticle(12.0)
    force = ColvarsForce.fromConfigString(config, str(tmp_path / "all_dummy"))
    force.setUseGpu(engine.use_gpu)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName(engine.platform), engine.properties)
    context.setPositions([[9.0, 8.0, 7.0]])
    distance = float(np.linalg.norm([0.7, -0.3, 0.4]))
    expected_energy = 0.5 * (10.0 / 0.5**2) * (distance - 0.4)**2
    state = context.getState(getEnergy=True, getForces=True)
    np.testing.assert_allclose(force.getColvarValue(context, "d"), distance,
                               atol=engine.tolerances[0], rtol=0)
    np.testing.assert_allclose(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                               expected_energy, atol=engine.tolerances[1], rtol=0)
    np.testing.assert_array_equal(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
                                  0.0)
    if engine.platform == "CUDA":
        assert "selected atoms=0" in force.getBackend(context)
    del context, integrator
