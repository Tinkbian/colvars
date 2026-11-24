"""CUDA adaptive histories against deterministic Colvars Reference dynamics."""

import re

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from conftest import Engine
from openmm_colvars import ColvarsForce, ColvarsTABFIntegrator
from test_tabf_integrator import abf_arrays, kinematics


CONFIG = """
units real
colvarsTrajFrequency 3
colvarsRestartFrequency 0
colvar {
  name d
  width 0.05
  lowerBoundary 7.7
  upperBoundary 8.3
  extendedLagrangian on
  extendedFluctuation 0.1
  extendedTimeConstant 28
  extendedLangevinDamping 0
  subtractAppliedForce on
  distance {
    group1 { atomNumbers 1 }
    group2 { atomNumbers 2 }
  }
}
colvar {
  name a
  width 0.01
  lowerBoundary 0.97
  upperBoundary 2.03
  hardLowerBoundary on
  hardUpperBoundary on
  reflectingLowerBoundary on
  reflectingUpperBoundary on
  extendedLagrangian on
  extendedMass 10000
  extendedLangevinDamping 0
  subtractAppliedForce on
  tabfAlpha {}
}
"""
for monitor, first, second in [("m13", 1, 3), ("m23", 2, 3)]:
    CONFIG += f"""
colvar {{
  name {monitor}
  distance {{
    group1 {{ atomNumbers {first} }}
    group2 {{ atomNumbers {second} }}
  }}
}}
"""
for cv, weight in [("d", 0.02), ("a", 0.05)]:
    CONFIG += f"""
abf {{
  name abf_{cv}
  colvars {cv}
  fullSamples 4
  outputEnergy on
}}
metadynamics {{
  name meta_{cv}
  colvars {cv}
  newHillFrequency 2
  gridsUpdateFrequency 6
  hillWidth 3
  hillWeight {weight}
  wellTempered on
  biasTemperature 4000
  writeHillsTrajectory on
  outputEnergy on
}}
"""


def context_for(engine, prefix, restart=None):
    system = mm.System()
    for mass in (12, 12, 14):
        system.addParticle(mass)
    physical = mm.CustomExternalForce("24 + 12*x")
    physical.addParticle(0, [])
    system.addForce(physical)
    system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*row) for row in 3*np.eye(3)))
    bias = ColvarsForce.fromConfigString(CONFIG, str(prefix))
    bias.setUnitSystem("real")
    bias.setTargetTemperature(300)
    bias.setUsesPeriodicBoundaryConditions(True)
    bias.setUseGpu(engine.use_gpu)
    if engine.use_gpu:
        bias.setUseDeviceBias(True)
    bias.setForceGroup(1)
    if restart:
        bias.setRestartInputPrefix(str(restart))
    system.addForce(bias)
    integrator = ColvarsTABFIntegrator(0.002, 300, 0, 1.113)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(engine.platform), engine.properties)
    context.setPositions([[0.2, 0, 0], [1.0021, 0, 0], [0.2, 0.5, 0]])
    context.setVelocities([[0, 0, 0], [1.9, 0, 0], [0, 0, 0]])
    if engine.use_gpu:
        assert "device TABF" in bias.getBackend(context)
    return context, integrator, bias


def bias_state_values(context):
    state = context.getState(getEnergy=True, getForces=True, groups={1})
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))


def compare_endpoints(reference, actual, tolerance):
    cref, iref, bref = reference
    c, integrator, bias = actual
    assert c.getStepCount() == cref.getStepCount()
    original = bref.getBiasState(cref)
    saved = bias.getBiasState(c)
    # Metadynamics prefixes each raw scalar grid with its dimensions.
    grids = [re.sub(r"^grid_parameters\s*\{[^}]*\}\n", "", text, flags=re.M)
             for text in (original, saved)]
    for key, n in (("samples", 2), ("gradient", 2), ("z_samples", 1), ("z_gradient", 1),
                   ("hills_energy", 2), ("hills_energy_gradients", 2)):
        expected, observed = (abf_arrays(text, key) for text in grids)
        assert len(expected) == len(observed) == n
        for lhs, rhs in zip(expected, observed):
            if key.endswith("samples"):
                np.testing.assert_array_equal(lhs, rhs)
            else:
                np.testing.assert_allclose(lhs, rhs, rtol=1.e-6, atol=tolerance)
    for key in ("lastHillStep", "unprojectedHills"):
        assert re.findall(key+r"\s+(\d+)", saved) == re.findall(key+r"\s+(\d+)", original)
    for lhs, rhs in zip(kinematics(cref), kinematics(c)):
        np.testing.assert_allclose(lhs, rhs, rtol=0, atol=tolerance)
    assert integrator.getAlpha() == pytest.approx(iref.getAlpha(), abs=tolerance)
    for lhs, rhs in zip(bias_state_values(cref), bias_state_values(c)):
        np.testing.assert_allclose(lhs, rhs, rtol=1.e-5, atol=tolerance*10)
    assert bias.getBiasState(c) == saved


def test_device_tabf_bias_grids_pending_hills_and_restart(engine, tmp_path):
    if not engine.use_gpu:
        pytest.skip("Two-backend comparison needs the resident CUDA build")
    reference = context_for(Engine("Reference", False, "double"), tmp_path/"reference")
    gpu = context_for(engine, tmp_path/"device")
    tolerance = max(2.e-6, 5*engine.tolerances[2])
    # A checkpoint between projections holds the projected edge hills and
    # two pending analytic hills (steps 8 and 10) for each WTM bias.
    reference[1].step(10)
    gpu[1].step(10)
    compare_endpoints(reference, gpu, tolerance)
    saved = tmp_path/"paired.colvars.state"
    saved.write_text(gpu[2].getBiasState(gpu[0]))
    assert re.findall(r"unprojectedHills\s+(\d+)", saved.read_text()) == ["2", "2"]
    checkpoint = gpu[0].createCheckpoint()
    gpu[2].flushOutputs(gpu[0])
    del gpu
    resumed = context_for(engine, tmp_path/"resumed", saved)
    resumed[0].loadCheckpoint(checkpoint)
    for steps in (2, 7):
        reference[1].step(steps)
        resumed[1].step(steps)
        compare_endpoints(reference, resumed, tolerance)
    resumed[2].flushOutputs(resumed[0])
    reference[2].flushOutputs(reference[0])
    for name in ("a", "d"):
        tail = f".colvars.meta_{name}.hills.traj"
        expected = np.loadtxt(tmp_path/("reference"+tail))
        observed = np.vstack([np.loadtxt(tmp_path/(prefix+tail)) for prefix in ("device", "resumed")])
        np.testing.assert_array_equal(observed[:, 0], expected[:, 0])
        np.testing.assert_allclose(observed[:, 1:], expected[:, 1:], rtol=1.e-5, atol=tolerance)
    expected = np.loadtxt(tmp_path/"reference.colvars.traj")
    observed = np.vstack([np.loadtxt(tmp_path/(prefix+".colvars.traj")) for prefix in ("device", "resumed")])
    np.testing.assert_array_equal(observed[:, 0], expected[:, 0])
    np.testing.assert_allclose(observed[:, 1:], expected[:, 1:], rtol=1.e-5, atol=tolerance)
