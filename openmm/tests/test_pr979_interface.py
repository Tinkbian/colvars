"""Public PR #979 entry points shared by the upstream and resident builds."""
import numpy as np
import openmm as mm
from openmm import unit
import pytest

from openmm_colvars import ColvarsForce, ColvarsTABFIntegrator


def test_force_configuration(tmp_path):
    force = ColvarsForce("start.in", "initial", "lambda0")
    assert not force.getTabfEnabled() and force.getAlchemyEnabled()
    force.setConfigString("colvarsTrajFrequency 0")
    assert force.getConfig() == "colvarsTrajFrequency 0"
    force.setConfigFile("next.in")
    force.setOutputPrefix(str(tmp_path / "out"))
    force.setRestartOutputPrefix(str(tmp_path / "restart"))
    force.setRestartInputPrefix("previous")
    force.setRestartFrequency(20)
    force.setParticleCharge(2, -0.5)
    assert force.getConfig() == "next.in"
    assert force.getRestartOutputPrefix() == str(tmp_path / "restart")
    assert force.getRestartInputPrefix() == "previous"
    assert force.getRestartFrequency() == 20
    assert force.getLambdaParameterName() == "lambda0"
    assert force.getNumParticleCharges() == 3
    assert force.getParticleCharge(0) == 0 and force.getParticleCharge(2) == -0.5


def test_chignolin_pair_forces(engine, tmp_path):
    pairs = ((5, 133), (31, 92), (31, 106))
    config = "units gromacs\ncolvarsTrajFrequency 0\ncolvarsRestartFrequency 0\n"
    for i, (first, second) in enumerate(pairs):
        config += f"""
colvar {{
 name d{i}
 width 1
 distance {{
  group1 {{ atomNumbers {first} }}
  group2 {{ atomNumbers {second} }}
 }}
}}
harmonic {{
 colvars d{i}
 centers 0.4
 forceConstant 8
}}
"""
    force = ColvarsForce.fromConfigString(config, str(tmp_path / "pairs"), "lambda0")
    force.setParticleCharge(4, -0.5)
    force.setAlchemyEnabled(False)
    if hasattr(force, "setUseGpu"):
        force.setUseGpu(engine.use_gpu)
    system = mm.System()
    for _ in range(133):
        system.addParticle(12.011)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(engine.platform),
                         engine.properties)
    positions = np.random.default_rng(979).normal(size=(133, 3))
    context.setPositions(positions)
    state = context.getState(getForces=True, getEnergy=True)
    energy = 0.0
    forces = np.zeros_like(positions)
    for first, second in pairs:
        delta = positions[second-1] - positions[first-1]
        distance = np.linalg.norm(delta)
        energy += 4*(distance-0.4)**2
        pair_force = 8*(distance-0.4)*delta/distance
        forces[first-1] += pair_force
        forces[second-1] -= pair_force
    assert state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        energy, abs=engine.tolerances[1], rel=0)
    np.testing.assert_allclose(state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole/unit.nanometer), forces, atol=engine.tolerances[2], rtol=0)


def test_explicit_tabf_names(engine, tmp_path):
    force = ColvarsForce.fromConfigString("""
colvarsTrajFrequency 1
colvarsRestartFrequency 2
colvar {
 name alpha
 width 0.01
 lowerBoundary 1
 upperBoundary 3
 extendedLagrangian on
 extendedMass 50000
 extendedLangevinDamping 0
 tabfAlpha {}
}
abf {
 colvars alpha
 applyBias off
 fullSamples 100
 outputFreq 2
}
""", str(tmp_path / "alpha"))
    if hasattr(force, "setUseGpu"):
        force.setUseGpu(engine.use_gpu)
    force.setAlchemyEnabled(False)
    force.setTabfEnabled(True)
    force.setTabfAlphaParameterName("alpha_control")
    force.setTabfAlphaDerivativeParameterName("alpha_derivative")
    force.setForceGroup(1)
    system = mm.System()
    system.addParticle(12)
    physical = mm.CustomExternalForce("24")
    physical.addParticle(0, [])
    system.addForce(physical)
    system.addForce(force)
    integrator = ColvarsTABFIntegrator(0.002, 300, 0, 2, 0, 1, "alpha_control", "alpha_derivative")
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(engine.platform),
                         engine.properties)
    context.setPositions([[0, 0, 0]])
    context.setVelocities([[0, 0, 0]])
    integrator.step(1)
    assert integrator.getGlobalVariableByName("alpha_derivative") == pytest.approx(-6)
    integrator.step(3)
    assert 2 < integrator.getGlobalVariableByName("alpha_control") < 3


def test_reference_alchemical_pme():
    import openmm_colvars as cv
    if not hasattr(cv, "AlchemicalPMEForce"):
        pytest.skip("AlchemicalPMEForce is provided by the upstream build")
    system = mm.System()
    force = cv.AlchemicalPMEForce()
    force.setNonbondedMethod(force.PME)
    force.setCutoffDistance(0.9)
    force.setPMEParameters(3.2, 24, 24, 24)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(2.5, 0, 0), mm.Vec3(0, 2.5, 0), mm.Vec3(0, 0, 2.5))
    for sign in (-1, 1):
        system.addParticle(12)
        force.addParticle(sign*0.5, sign*0.75, 0.28, 0.31, 0.2, 0.3)
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    positions = np.array([[0.2, 0.3, 0.4], [0.72, 0.61, 0.55]])
    context.setPositions(positions)
    context.setParameter("lambda", 0.4)
    state = context.getState(getForces=True, getParameterDerivatives=True)
    analytic_derivative = state.getEnergyParameterDerivatives()["lambda"]
    analytic_force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)[0, 0]

    def energy():
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    delta = 1e-4
    context.setParameter("lambda", 0.4+delta)
    high = energy()
    context.setParameter("lambda", 0.4-delta)
    assert analytic_derivative == pytest.approx((high-energy())/(2*delta), rel=2e-5)
    context.setParameter("lambda", 0.4)
    positions[0, 0] += delta
    context.setPositions(positions)
    high = energy()
    positions[0, 0] -= 2*delta
    context.setPositions(positions)
    assert analytic_force == pytest.approx((energy()-high)/(2*delta), rel=2e-5)
