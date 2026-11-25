"""Joint distance/alpha ABF and WTM with an expanding metadynamics grid."""

import re

import numpy as np
import openmm as mm
import pytest

from conftest import Engine
from openmm_colvars import ColvarsForce, ColvarsTABFIntegrator
from test_device_tabf_bias import bias_state_values
from test_tabf_integrator import abf_arrays, kinematics


CONFIG = """
units real
colvarsTrajFrequency 1
colvarsRestartFrequency 0
colvar {
  name d
  width 0.25
  lowerBoundary 8.0
  upperBoundary 9.0
  expandBoundaries on
  lowerWall 8.0
  upperWall 9.0
  lowerWallConstant 0.1
  upperWallConstant 0.1
  extendedLagrangian on
  extendedFluctuation 0.25
  extendedTimeConstant 10
  extendedLangevinDamping 0
  subtractAppliedForce on
  distance {
    group1 { atomNumbers 1 }
    group2 { atomNumbers 2 }
  }
}
colvar {
  name alpha
  width 0.06
  lowerBoundary 0.8
  upperBoundary 1.4
  hardLowerBoundary on
  hardUpperBoundary on
  reflectingLowerBoundary on
  reflectingUpperBoundary on
  extendedLagrangian on
  extendedMass 220
  extendedLangevinDamping 0
  outputTotalForce on
  subtractAppliedForce on
  tabfAlpha {}
}
abf {
  name joint_abf
  colvars d alpha
  CZARestimator off
  fullSamples 200
  stepZeroData on
  outputEnergy on
  outputFreq 21
  historyFreq 21
}
metadynamics {
  name joint_wtm
  colvars d alpha
  newHillFrequency 3
  gridsUpdateFrequency 5
  gaussianSigmas 0.5 0.045
  hillWeight 0.06
  wellTempered on
  biasTemperature 900
  writeHillsTrajectory on
  outputEnergy on
  keepFreeEnergyFiles on
  outputFreq 22
}
"""


def context_for(engine, prefix, device=False, restart=None, config=CONFIG):
    system = mm.System()
    for mass in (12, 14):
        system.addParticle(mass)
    physical = mm.CustomExternalForce("-8.368")
    physical.addParticle(0, [])
    system.addForce(physical)
    bias = ColvarsForce.fromConfigString(config, str(prefix))
    bias.setUnitSystem("real")
    bias.setTargetTemperature(300)
    bias.setUsesPeriodicBoundaryConditions(False)
    bias.setUseGpu(engine.platform == "CUDA")
    bias.setUseDeviceBias(device)
    bias.setForceGroup(1)
    if restart:
        bias.setRestartInputPrefix(str(restart))
    system.addForce(bias)
    integrator = ColvarsTABFIntegrator(0.0005, 300, 0, 1.147)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(engine.platform), engine.properties)
    context.setPositions([[0.2, 0, 0], [1.014, 0, 0]])
    context.setVelocities([[0, 0, 0], [6.7, 0, 0]])
    if device:
        assert "device TABF" in bias.getBackend(context)
    return context, integrator, bias


def trajectory(prefix):
    path = prefix.with_name(prefix.name+".colvars.traj")
    names = path.read_text().splitlines()[0].lstrip("# ").split()
    data = np.loadtxt(path, ndmin=2)
    return dict(zip(names, data.T))


def state_grids(state):
    meta_params = re.findall(r"^grid_parameters\s*\{([^}]*)\}", state, re.M)
    raw = re.sub(r"^grid_parameters\s*\{[^}]*\}\s*\n", "", state, flags=re.M)
    grids = {}
    for key in ("samples", "gradient", "hills_energy", "hills_energy_gradients"):
        grids[key], = abf_arrays(raw, key)
    assert not re.search(r"^z_(samples|gradient)\s*$", state, re.M)
    return grids, meta_params


def check_metadynamics(prefix, grids, meta_params, tr):
    lower, = re.findall(r"lower_boundaries\s+([^\n]+)", meta_params[0])
    sizes, = re.findall(r"sizes\s+([^\n]+)", meta_params[0])
    lower = np.fromstring(lower, sep=" ")
    n_d, n_alpha = np.fromstring(sizes, sep=" ", dtype=int)
    widths = np.array([0.25, 0.06])
    assert lower[0] < 8 and lower[0]+n_d*widths[0] > 9
    assert n_alpha == 10 and lower[1] == pytest.approx(0.8)
    axes = [lower[i]+widths[i]*(np.arange(n)+0.5) for i, n in enumerate((n_d, n_alpha))]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    hills = np.loadtxt(prefix.with_name(prefix.name+".colvars.joint_wtm.hills.traj"), ndmin=2)
    np.testing.assert_array_equal(hills[:, 0], np.arange(3, 38, 3))

    def gaussian(x, center, sigma, weight):
        dist2 = np.sum(((x-center)/sigma)**2, axis=-1)
        return np.where(dist2 <= 23, weight*np.exp(-0.5*dist2), 0.0)

    # Existing projected hills are evaluated at the bin center, and pending
    # hills are evaluated analytically at the current extended coordinate.
    expected_energy = np.zeros((n_d, n_alpha))
    expected_gradient = np.zeros_like(grid)
    active_lower, active_upper = 8.0, 9.0
    pending = []
    for step, d, alpha in zip(tr["step"].astype(int), tr["r_d"], tr["alpha"]):
        center = np.array([d, alpha])
        # gaussianSigmas sets a one-cell expansion margin. A newly created
        # row has zero historical grid energy until future projections.
        bin_lower = 8.0+widths[0]*np.floor((d-8.0)/widths[0])
        active_lower = min(active_lower, bin_lower-widths[0])
        active_upper = max(active_upper, bin_lower+2*widths[0])
        if step % 3 == 0:
            hill = hills[step//3-1]
            np.testing.assert_allclose(hill[1:3], center, rtol=0, atol=1.e-10)
            np.testing.assert_allclose(hill[3:5], [0.5, 0.045], atol=1.e-12, rtol=0)
            cell = tuple(np.floor((center-lower)/widths).astype(int))
            potential = expected_energy[cell] + sum(gaussian(center, h[1:3], h[3:5], h[5]) for h in pending)
            assert hill[5] == pytest.approx(0.06*np.exp(-potential/(900*0.00831446261815324/4.184)),
                                            rel=2.e-6, abs=2.e-8)
            pending.append(hill)
        if step % 5 == 0:
            active = (grid[:, :, 0] > active_lower) & (grid[:, :, 0] < active_upper)
            for h in pending:
                contribution = active*gaussian(grid, h[1:3], h[3:5], h[5])
                expected_energy += contribution
                expected_gradient -= contribution[:, :, None]*(grid-h[1:3])/h[3:5]**2
            pending.clear()
    assert len(pending) == 1
    np.testing.assert_allclose(grids["hills_energy"], expected_energy.ravel(), rtol=2.e-6, atol=2.e-7)
    np.testing.assert_allclose(grids["hills_energy_gradients"], expected_gradient.ravel(), rtol=2.e-6, atol=2.e-6)


def check_analytical_grids(prefix, state):
    # Reconstruct every cell independently from the reported endpoints. The
    # raw serialized gradient is the conditional mean in each occupied cell.
    tr = trajectory(prefix)
    grids, params = state_grids(state)
    d, ext, alpha = tr["d"], tr["r_d"], tr["alpha"]
    k = 0.00831446261815324*300/4.184/0.25**2
    np.testing.assert_allclose(tr["ft_alpha"], -2/alpha**2, rtol=2.e-6, atol=2.e-6)
    assert np.any(d > 9.0)
    assert np.all((alpha >= 0.8) & (alpha <= 1.4))
    counts = np.zeros((4, 10), dtype=int)
    sums = np.zeros((4, 10, 2))
    for i, position in enumerate(ext):
        x = int(np.floor((position-8.0)/0.25))
        y = int(np.floor((alpha[i]-0.8)/0.06))
        if 0 <= x < 4 and 0 <= y < 10:
            counts[x, y] += 1
            sums[x, y] += [k*(ext[i]-d[i]), -tr["ft_alpha"][i]]
    assert 0 < counts.sum() < len(d)
    expected = sums/np.maximum(counts, 1)[:, :, None]
    np.testing.assert_array_equal(grids["samples"], counts.ravel())
    np.testing.assert_allclose(grids["gradient"], expected.ravel(), rtol=2.e-5, atol=5.e-5)
    # WTM has expanded beyond the shared CV's initial ABF window. The
    # reflecting alpha bounds keep the second axis at its original size.
    check_metadynamics(prefix, grids, params, tr)


def test_joint_tabf_trajectory_and_grid_oracle(engine, tmp_path):
    prefix = tmp_path/"joint"
    context, integrator, bias = context_for(engine, prefix, device=engine.use_gpu)
    integrator.step(37)
    bias.flushOutputs(context)
    check_analytical_grids(prefix, bias.getBiasState(context))
    assert bias.getBiasUpdateCount(context) == 37
    assert np.isfinite(bias_state_values(context)[0])


def test_joint_device_rejects_czar(engine, tmp_path):
    if not engine.use_gpu:
        pytest.skip("Checks the strict joint device layout")
    with pytest.raises(Exception, match="CZAR"):
        context_for(engine, tmp_path/"czar", device=True, config=CONFIG.replace("CZARestimator off", "CZARestimator on"))


def snapshot(run, prefix):
    context, integrator, bias = run
    bias.flushOutputs(context)
    # Multidimensional ABF integrates its gridded PMF on output. The next
    # energy query must see that just-computed scalar surface.
    before = bias.getBiasState(context)
    energy, forces = bias_state_values(context)
    for _ in range(2):
        again, f_again = bias_state_values(context)
        assert again == pytest.approx(energy, abs=1.e-9)
        np.testing.assert_array_equal(f_again, forces)
    assert bias.getBiasState(context) == before
    grids, params = state_grids(before)
    output = {}
    for suffix in ("count", "grad", "pmf"):
        path = prefix.with_name(f"{prefix.name}.joint_abf.{suffix}")
        output[suffix] = np.loadtxt(path, ndmin=2)
    assert not prefix.with_name(f"{prefix.name}.joint_abf.zcount").exists()
    return (grids, params, output, energy, forces, kinematics(context), integrator.getAlpha(),
            re.findall(r"(?:lastHillStep|unprojectedHills)\s+(\d+)", before))


def compare(expected, observed, atol):
    grids, params, output, energy, force, xv, alpha, hills = expected
    other_grids, other_params, other_output, other_energy, other_force, other_xv, other_alpha, other_hills = observed
    assert other_params == params
    assert other_hills == hills
    for key in grids:
        if key.endswith("samples"):
            np.testing.assert_array_equal(other_grids[key], grids[key])
        else:
            np.testing.assert_allclose(other_grids[key], grids[key], rtol=1.e-5, atol=atol)
    for key in output:
        np.testing.assert_allclose(other_output[key], output[key], rtol=1.e-5, atol=atol)
    assert other_energy == pytest.approx(energy, abs=atol*10)
    assert other_alpha == pytest.approx(alpha, abs=atol)
    np.testing.assert_allclose(other_force, force, rtol=1.e-5, atol=atol*10)
    for ref, obs in zip(xv, other_xv):
        np.testing.assert_allclose(obs, ref, rtol=0, atol=atol)


def test_joint_tabf_expansion_and_paired_restart(engine, tmp_path):
    if not engine.use_gpu:
        pytest.skip("Compares device TABF with both CUDA hybrid and Reference")
    config = CONFIG.replace("colvarsTrajFrequency 1", "colvarsTrajFrequency 7")
    config = config.replace("stepZeroData on", "stepZeroData off")
    # Activate ABF in this short run so both adaptive forces influence motion
    # across the checkpoint; the production 200-sample ramp is tested separately.
    config = config.replace("fullSamples 200", "fullSamples 4")
    reference = context_for(Engine("Reference", False, "double"), tmp_path/"ref", config=config)
    reference[1].step(19)
    before_abf_integration = bias_state_values(reference[0])[0]
    first_reference = snapshot(reference, tmp_path/"ref")
    assert first_reference[0]["samples"].max() >= 4
    assert abs(before_abf_integration-first_reference[3]) > 0.1
    ref_file = tmp_path/"reference.state"
    ref_file.write_text(reference[2].getBiasState(reference[0]))
    ref_checkpoint = reference[0].createCheckpoint()
    reference[1].step(18)
    final_reference = snapshot(reference, tmp_path/"ref")
    del reference
    # Ordinary 2D ABF stores counts and gradients. The integrated PMF used
    # for reporting is refreshed at its next output after a restart.
    ref_tail = tmp_path/"ref-restart"
    reference = context_for(Engine("Reference", False, "double"), ref_tail, restart=ref_file, config=config)
    reference[0].loadCheckpoint(ref_checkpoint)
    reference[1].step(18)
    snapshot(reference, ref_tail)
    del reference
    atol = max(2.e-6, 2*engine.tolerances[2])
    for device, name in [(False, "hybrid"), (True, "device")]:
        prefix = tmp_path/name
        run = context_for(engine, prefix, device=device, config=config)
        run[1].step(19)
        compare(first_reference, snapshot(run, prefix), atol)
        saved = run[2].getBiasState(run[0])
        assert re.search(r"unprojectedHills\s+1\b", saved)
        state_file = tmp_path/(name+".state")
        state_file.write_text(saved)
        checkpoint = run[0].createCheckpoint()
        del run
        continued = tmp_path/(name+"-restart")
        run = context_for(engine, continued, device=device, restart=state_file, config=config)
        run[0].loadCheckpoint(checkpoint)
        run[1].step(18)
        compare(final_reference, snapshot(run, continued), atol)
        del run
        for suffix in (".joint_abf.hist.grad", ".joint_abf.hist.count", ".joint_abf.hist.pmf",
                       ".joint_wtm.22.pmf", ".joint_wtm.37.pmf"):
            expected = np.loadtxt(tmp_path/("ref"+suffix), ndmin=2)
            np.testing.assert_allclose(np.loadtxt(continued.with_name(continued.name+suffix), ndmin=2),
                                       expected, rtol=1.e-5, atol=atol)
        for suffix in (".colvars.traj", ".colvars.joint_wtm.hills.traj"):
            reference_rows = np.loadtxt(tmp_path/("ref"+suffix), ndmin=2)
            expected = np.vstack([reference_rows[reference_rows[:, 0] <= 19],
                                  np.loadtxt(ref_tail.with_name(ref_tail.name+suffix), ndmin=2)])
            joined = np.vstack([np.loadtxt(path.with_name(path.name+suffix), ndmin=2)
                                for path in (prefix, continued)])
            np.testing.assert_array_equal(joined[:, 0], expected[:, 0])
            np.testing.assert_allclose(joined[:, 1:], expected[:, 1:], rtol=2.e-5, atol=atol*5)
