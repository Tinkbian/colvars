"""Validate the resident physical energy reduction, constraints, and RNG continuation."""

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from openmm_colvars import ColvarsForce, ColvarsTABFIntegrator
from test_device_tabf_bias import CONFIG


def pme_context(engine, prefix, restart=None, barostat=False, device=True, probe_energy=False):
    system = mm.System()
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(nonbonded.PME)
    nonbonded.setCutoffDistance(1.0)
    for mass, charge in [(12, 1.0), (12, -0.5), (14, 0.2)]:
        system.addParticle(mass)
        nonbonded.addParticle(charge, 0.3, 0.2)
    system.addForce(nonbonded)
    system.addForce(mm.CMMotionRemover(1))
    if probe_energy:
        offset = mm.CustomExternalForce("log(energyScale)")
        offset.addGlobalParameter("energyScale", 1.0)
        offset.addParticle(0, [])
        system.addForce(offset)
    if barostat:
        system.addForce(mm.MonteCarloBarostat(1.0, 300.0, 1))
    system.addConstraint(0, 1, 0.8)
    system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*row) for row in 3*np.eye(3)))
    bias = ColvarsForce.fromConfigString(CONFIG, str(prefix))
    bias.setUnitSystem("real")
    bias.setTargetTemperature(300)
    bias.setUsesPeriodicBoundaryConditions(True)
    bias.setUseGpu(True)
    bias.setUseDeviceBias(device)
    bias.setForceGroup(1)
    if restart is not None:
        bias.setRestartInputPrefix(str(restart))
    system.addForce(bias)
    integrator = ColvarsTABFIntegrator(0.001, 297, 2.3, 1.113)
    integrator.setRandomNumberSeed(12571)
    properties = dict(engine.properties, UseCpuPme="false")
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"), properties)
    context.setPositions([[0.2, 0, 0], [1.0, 0, 0], [0.2, 0.6, 0]])
    context.setVelocities([[0, 0, 0]]*3)
    return context, integrator, bias


def test_device_tabf_rejects_unscaled_barostat(engine, tmp_path):
    if not engine.use_gpu:
        pytest.skip("Device TABF requires CUDA")
    with pytest.raises(Exception, match="barostat"):
        pme_context(engine, tmp_path/"npt", barostat=True)


def test_failed_device_step_requires_new_context(engine, tmp_path):
    if not engine.use_gpu:
        pytest.skip("Device TABF requires CUDA")
    context, integrator, bias = pme_context(engine, tmp_path/"bad-energy", probe_energy=True)
    integrator.step(1)
    context.setParameter("energyScale", -1)
    with pytest.raises(Exception, match="Nonfinite|Invalid CUDA TABF"):
        integrator.step(1)
    failed_step = context.getStepCount()
    context.setParameter("energyScale", 1)
    with pytest.raises(Exception, match="new Context"):
        integrator.step(1)
    assert context.getStepCount() == failed_step


def test_device_pme_thermostat_matches_openmm_cuda(engine, tmp_path):
    if not engine.use_gpu:
        pytest.skip("Device TABF requires CUDA")
    outcomes = []
    for device in (False, True):
        context, integrator, bias = pme_context(engine, tmp_path/str(device), device=device)
        integrator.step(13)
        state = context.getState(getPositions=True, getVelocities=True)
        outcomes.append((state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                         state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond),
                         integrator.getAlpha(), integrator.getAlphaDerivative()))
        del context, integrator, bias
    # Single-precision OpenMM rounds intermediate coordinates before
    # reconstructing constrained velocities from their displacement.
    single = engine.precision == "single"
    tolerances = (5.e-6, 5.e-4, 5.e-6, 1.e-3) if single else (2.e-7, 1.e-5, 2.e-6, 1.e-4)
    for host, device, atol in zip(*outcomes, tolerances):
        np.testing.assert_allclose(host, device, rtol=0, atol=atol)


def test_device_physical_pme_and_checkpoint(engine, tmp_path):
    if not engine.use_gpu:
        pytest.skip("Device TABF requires CUDA")
    context, integrator, bias = pme_context(engine, tmp_path/"first")
    for step in range(1, 4):
        alpha = integrator.getAlpha()
        integrator.step(1)
        state = context.getState(getPositions=True, getVelocities=True, getEnergy=True, groups={0})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        derivative = -energy/alpha**2
        assert integrator.getAlphaDerivative() == pytest.approx(derivative, rel=2.e-6, abs=2.e-6)
        assert context.getParameter("colvars_tabf_dE_dalpha") == integrator.getAlphaDerivative()
        x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        assert np.linalg.norm(x[1]-x[0]) == pytest.approx(0.8, abs=1.e-5)
        assert abs(np.dot(x[1]-x[0], v[1]-v[0])) < 2.e-5
        assert state.getStepCount() == step
    state_file = tmp_path/"checkpoint.colvars.state"
    state_file.write_text(bias.getBiasState(context))
    checkpoint = context.createCheckpoint()
    integrator.step(5)
    expected = context.getState(getPositions=True, getVelocities=True)
    expected_alpha = integrator.getAlpha()
    del integrator, context, bias

    context, integrator, bias = pme_context(engine, tmp_path/"continued", state_file)
    context.loadCheckpoint(checkpoint)
    integrator.step(5)
    actual = context.getState(getPositions=True, getVelocities=True)
    for observed, reference in [(actual.getPositions(asNumpy=True), expected.getPositions(asNumpy=True)),
                                (actual.getVelocities(asNumpy=True), expected.getVelocities(asNumpy=True))]:
        np.testing.assert_allclose(observed.value_in_unit(reference.unit), reference.value_in_unit(reference.unit),
                                   rtol=0, atol=2.e-6)
    assert integrator.getAlpha() == pytest.approx(expected_alpha, rel=0, abs=2.e-6)
