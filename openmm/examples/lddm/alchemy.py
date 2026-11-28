"""Single growing partition with NAMD's linear PME energy scaling."""

import copy
import openmm as mm
from openmm_fep import prepare_fep_system


def prepare_lddm_system(system, config):
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            # cuPotential 0.3.5 accepts this common automatic mesh policy.
            force.setEwaldErrorTolerance(5e-4)
    prepared = prepare_fep_system(system, config)
    ligand = set(config.alch_atoms.grow)
    if not ligand or config.alch_atoms.vanish or config.alch_decouple:
        raise ValueError("LDDM requires one growing, annihilated partition")
    original, = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    if original.getNonbondedMethod() != mm.NonbondedForce.PME:
        raise ValueError("LDDM requires PME")
    # In the main PME grid q_ligand scales by l, so its self-interaction
    # scales by l^2. NAMD scales the complete partition energy linearly.
    # This second PME grid includes images, reciprocal, self and exclusions.
    # The main force's 1-4 parameter offsets already scale linearly.
    internal = copy.deepcopy(original)
    for atom in range(internal.getNumParticles()):
        q, sigma, epsilon = internal.getParticleParameters(atom)
        internal.setParticleParameters(atom, q if atom in ligand else 0, sigma, 0)
    for i in range(internal.getNumExceptions()):
        a, b, _, sigma, _ = internal.getExceptionParameters(i)
        internal.setExceptionParameters(i, a, b, 0, sigma, 0)
    correction = mm.CustomCVForce("l*(1-l)*internal; l=fep_grow_electrostatics")
    correction.addGlobalParameter("fep_grow_electrostatics", 1)
    correction.addCollectiveVariable("internal", internal)
    correction.setForceGroup(prepared.force_group_electrostatics)
    remove = []
    for i, force in enumerate(prepared.system.getForces()):
        if isinstance(force, mm.CustomNonbondedForce):
            if "lambda_correction" in force.getEnergyFunction():
                remove.append(i)
            elif force.getForceGroup() == prepared.force_group_vdw:
                # NAMD's potential switch is cubic in squared distance.
                # Switching is a function of the physical pair distance.
                force.setUseSwitchingFunction(False)
                expr = force.getEnergyFunction()
                force.setEnergyFunction(
                    f"({expr})*sw;sw=select(step(r-0.9),"
                    "(1-r^2)^2*(1+2*r^2-3*0.9^2)/(1-0.9^2)^3,1)"
                )
    for i in reversed(remove):
        prepared.system.removeForce(i)
    prepared.system.addForce(correction)
    return prepared
