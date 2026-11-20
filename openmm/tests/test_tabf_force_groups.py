"""OpenMM integration entry points, force-group validation, and COM removal."""

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from openmm_colvars import ColvarsForce, ColvarsTABFIntegrator
from test_tabf_integrator import alpha_config


def grouped_system(engine, prefix):
    system = mm.System()
    system.addParticle(12)
    system.addParticle(12)
    bias = ColvarsForce.fromConfigString(alpha_config("gromacs", 500) + """
colvar {
  name d
  distance {
    group1 { atomNumbers 1 }
    group2 { atomNumbers 2 }
  }
}
""", str(prefix))
    bias.setUseGpu(engine.use_gpu)
    bias.setForceGroup(7)
    system.addForce(bias)
    return system


def test_tabf_context_update_with_legacy_constructor(engine, tmp_path):
    system = grouped_system(engine, tmp_path / "com")
    com = mm.CMMotionRemover(1)
    com.setForceGroup(5)
    system.addForce(com)
    integrator = ColvarsTABFIntegrator(1*unit.femtosecond, 300*unit.kelvin,
                                       0/unit.picosecond, 2.0, 5, 7, "alpha", "derivative")
    integrator.setStepSize(2*unit.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(engine.platform), engine.properties)
    positions = [[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]]
    context.setPositions(positions)
    context.setVelocities([[1, 0, 0], [1, 0, 0]])
    integrator.step(2)
    state = context.getState(getPositions=True, getVelocities=True)
    np.testing.assert_allclose(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                               positions, atol=engine.tolerances[0], rtol=0)
    np.testing.assert_allclose(state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond),
                               0, atol=engine.tolerances[0], rtol=0)
    assert integrator.getGlobalVariableByName("alpha") == 2.0
    assert integrator.getGlobalVariableByName("derivative") == 0.0


@pytest.mark.parametrize("case,match", [
    ("outside", "all physical forces"),
    ("collision", "reserved TABF"),
])
def test_tabf_detects_unintended_force_group_or_parameter(engine, tmp_path, case, match):
    system = grouped_system(engine, tmp_path / case)
    name = "colvars_tabf_dE_dalpha" if case == "collision" else "offset"
    physical = mm.CustomExternalForce(name)
    physical.addGlobalParameter(name, 0)
    physical.addParticle(0, [])
    physical.setForceGroup(5 if case == "collision" else 3)
    system.addForce(physical)
    integrator = ColvarsTABFIntegrator(0.002, 300, 1, 1, 5, 7)
    with pytest.raises(Exception, match=match):
        mm.Context(system, integrator, mm.Platform.getPlatformByName(engine.platform), engine.properties)
