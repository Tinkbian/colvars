"""Check the joint ABF sample ramp for both components of one occupied bin."""

import numpy as np

from test_device_tabf_joint import CONFIG, context_for, trajectory
from test_tabf_integrator import abf_arrays


def test_joint_abf_reversed_axes_and_full_samples(engine, tmp_path):
    config = CONFIG.split("metadynamics {")[0]
    config = config.replace("expandBoundaries on", "expandBoundaries off")
    config = config.replace("colvars d alpha", "colvars alpha d")
    config = config.replace("extendedMass 220", "extendedMass 5000000")
    config = config.replace("subtractAppliedForce on", "subtractAppliedForce on\n  outputAppliedForce on")
    config = config.replace("extendedFluctuation 0.25", "extendedFluctuation 0.25\n  outputTotalForce on")
    prefix = tmp_path/"ramp"
    context, integrator, bias = context_for(engine, prefix, device=engine.use_gpu, config=config)
    context.setPositions([[0.2, 0, 0], [1.044, 0, 0]])
    context.setVelocities([[0, 0, 0], [0.011, 0, 0]])
    integrator.step(205)
    bias.flushOutputs(context)
    tr = trajectory(prefix)
    samples, = abf_arrays(bias.getBiasState(context), "samples")
    samples = samples.reshape(10, 4)
    assert samples[5, 1] == 205 and samples.sum() == 205
    # The negative mean system force is ramped from zero at sample 100
    # to its full magnitude at sample 200, independently on each axis.
    n = np.arange(1, 206)
    ramp = np.clip((n-100)/100, 0., 1.)
    for axis in ["alpha", "d"]:
        observed = tr["fa_"+axis]
        expected = -np.cumsum(tr["ft_"+axis])/n*ramp
        np.testing.assert_allclose(observed, expected, atol=2.e-7, rtol=2.e-6)
        assert observed[99] == 0
        assert abs(observed[199]) > 1.e-8
