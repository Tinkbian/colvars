"""Run one LDDM preparation stage or one bidirectionally sampled lambda state."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import openmm as mm
from openmm import app, unit as u
from openmm_colvars import ColvarsForce

from openmm_fep import FEPAtoms, FEPConfig
from openmm_fep.openmm import set_context_lambda
from alchemy import prepare_lddm_system
from protocol import CVS, completed_trajectory, cv_config, fit_restraints


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "bfee", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--mode", choices=("cpu", "gpu", "cupotential"), required=True)
    p.add_argument("--stage", choices=("eq", "reeq", "solvent-eq", "bound", "solvent"), required=True)
    p.add_argument("--state", type=Path, help="OpenMM State XML from the preceding stage")
    p.add_argument("--params", type=Path, help="six restraint centers and fitted force constants")
    p.add_argument("--intervals", type=int)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--steps", type=int, help="default 5 ns EQ or 500000 steps per FEP state")
    p.add_argument("--discard", type=int, default=100000)
    p.add_argument("--seed", type=int, default=39173)
    args = p.parse_args()
    window = args.stage in ("bound", "solvent")
    if args.intervals is None:
        args.intervals = 10 if args.stage == "bound" else 5
    if args.steps is None:
        args.steps = 500000 if window else 2500000
    if args.steps <= 0 or args.intervals <= 0 or not 0 <= args.index <= args.intervals:
        p.error("invalid steps, intervals or window index")
    if window and (args.steps % 50 or args.discard % 50 or not 0 <= args.discard < args.steps):
        p.error("sampled windows require multiples of 50 and a positive production interval")
    if (window or args.stage == "reeq") and args.state is None:
        p.error("this stage requires --state")
    if args.stage in ("bound", "reeq") and args.params is None:
        p.error("this stage requires --params")
    return args


def input_names(inputs, solvent):
    if not solvent:
        return "complex.pdb", "complex.parm7", "fep.pdb"
    if (inputs/"ligandOnly.parm7").is_file():
        return "ligandOnly.pdb", "ligandOnly.parm7", "fep_ligandOnly.pdb"
    return "ligo.pdb", "ligo_hmr.prmtop", "ligo.pdb"


def base_system(inputs, solvent):
    pdb_name, top_name, _ = input_names(inputs, solvent)
    pdb_path, top_path = inputs/pdb_name, inputs/top_name
    pdb = app.PDBFile(str(pdb_path))
    box = pdb.topology.getPeriodicBoxVectors()
    top = app.AmberPrmtopFile(str(top_path), periodicBoxVectors=box)
    if box is None or top.topology.getNumAtoms() != len(pdb.positions):
        raise ValueError("inconsistent Amber topology, coordinates or box")
    for a, b in zip(top.topology.atoms(), pdb.topology.atoms()):
        if (a.name, a.residue.name) != (b.name, b.residue.name):
            raise ValueError(f"coordinate/topology order differs at atom {a.index+1}")
    system = top.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*u.nanometer,
                              constraints=app.HBonds, rigidWater=True, removeCMMotion=False)
    nb, = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(0.9*u.nanometer)
    nb.setEwaldErrorTolerance(1e-5)
    nb.setUseDispersionCorrection(False)
    return pdb, system


def install_cupotential(system):
    import nv_openmm_plugin as nv
    idx, = [i for i, f in enumerate(system.getForces()) if isinstance(f, mm.NonbondedForce)]
    source = system.getForce(idx)
    replacement = nv.NonbondedForce(source)
    replacement.setForceGroup(source.getForceGroup())
    system.addForce(replacement)
    system.removeForce(idx)
    return type(replacement).__module__ + "." + type(replacement).__name__


def save_state(ctx, path):
    state = ctx.getState(getPositions=True, getVelocities=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(mm.XmlSerializer.serialize(state))
    tmp.replace(path)


def sample_window(args, sim, prepared, bias, params):
    here = 1 - args.index / args.intervals
    neighbours = [j for j in (args.index+1, args.index-1) if 0 <= j <= args.intervals]
    ctx = sim.context
    groups = {prepared.force_group_electrostatics, prepared.force_group_vdw}
    def energy():
        return ctx.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(u.kilocalorie_per_mole)
    sim.step(args.discard)
    with (args.out/"samples.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "from", "to", "physical_kcal", "restraint_kcal", "total_kcal"])
        for step in range(args.discard+50, args.steps+1, 50):
            sim.step(50)
            base = energy()
            restraint = 0.
            if bias:
                ctx.getState(getEnergy=True, groups={bias.getForceGroup()})
                for cv, center, k in zip(CVS, params["centers"], params["k"]):
                    delta = bias.getColvarValue(ctx, cv) - center
                    if cv == "polarPhi":
                        delta = (delta + 180) % 360 - 180
                    restraint += 0.5*k*delta**2
            # NAMD SimParameters::setupIDWS chooses alchOutFreq when nonzero.
            # Odd blocks project to lambda2 and even blocks to lambdaIDWS.
            other = neighbours[0] if len(neighbours) == 1 or (step//50) % 2 else neighbours[1]
            try:
                set_context_lambda(ctx, prepared.config, 1-other/args.intervals)
                physical = energy() - base
                bias_work = (other-args.index)/args.intervals*restraint
                if not math.isfinite(physical+bias_work):
                    raise ValueError("nonfinite neighbour energy")
                writer.writerow([step, args.index, other, physical, bias_work, physical+bias_work])
            finally:
                set_context_lambda(ctx, prepared.config, here)


def main():
    args = cli()
    args.inputs = args.inputs.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    window = args.stage in ("bound", "solvent")
    solvent = args.stage.startswith("solvent")
    pdb, system = base_system(args.inputs, solvent)
    params = json.loads(args.params.read_text()) if args.params else None
    prepared = None
    if window:
        flags = args.inputs / input_names(args.inputs, solvent)[2]
        atoms = FEPAtoms.from_alch_file(flags)
        if not atoms.grow or atoms.vanish:
            raise ValueError("BFEE LDDM expects a ligand flagged +1")
        cfg = FEPConfig(atoms, (1, 0), (args.steps,), alch_vdw_lambda_end=0.7)
        prepared = prepare_lddm_system(system, cfg)
        system = prepared.system
    if args.mode == "cupotential" and not window:
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setEwaldErrorTolerance(5e-4)
    bias = None
    if not solvent:
        constants = params
        if args.stage == "reeq":
            constants = dict(params, k=[0.1]*5 + [10.0])
        fraction = args.index/args.intervals if window else 1.
        config = cv_config(args.bfee, args.inputs, args.stage, constants, fraction)
        config_file = args.out/"colvars.in"
        config_file.write_text(config)
        bias = ColvarsForce(str(config_file.resolve()), str((args.out/"cv").resolve()))
        bias.setUnitSystem("real")
        bias.setUseGpu(args.mode != "cpu")
        bias.setTargetTemperature(300)
        bias.setForceGroup(30)
        system.addForce(bias)
    pressure = mm.MonteCarloBarostat(1.01325*u.bar, 300*u.kelvin, 25)
    pressure.setRandomNumberSeed(args.seed)
    system.addForce(pressure)
    integrator = mm.LangevinMiddleIntegrator(300*u.kelvin, 1/u.picosecond, 0.002*u.picoseconds)
    integrator.setRandomNumberSeed(args.seed)
    integrator.setConstraintTolerance(1e-5)
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "mixed"}
    initial_positions = pdb.positions
    native_minimum = not args.state and args.mode == "cupotential"
    if native_minimum:
        # OpenMM's minimizer may evaluate high-force configurations on Reference.
        # cuPotential is CUDA-only. Minimize the matching native force first.
        if bias:
            bias.setOutputPrefix(str((args.out/"minimize").resolve()))
        minimizer = app.Simulation(pdb.topology, system, mm.VerletIntegrator(0.002), platform, properties)
        minimizer.context.setPositions(initial_positions)
        minimizer.minimizeEnergy(maxIterations=500)
        initial_positions = minimizer.context.getState(getPositions=True).getPositions()
        del minimizer
        if bias:
            bias.setOutputPrefix(str((args.out/"cv").resolve()))
    nb_type = install_cupotential(system) if args.mode == "cupotential" else "openmm.NonbondedForce"
    sim = app.Simulation(pdb.topology, system, integrator, platform, properties)
    ctx = sim.context
    if args.state:
        sim.loadState(str(args.state))
    else:
        ctx.setPositions(initial_positions)
    if window:
        set_context_lambda(ctx, prepared.config, 1-args.index/args.intervals)
    if not args.state:
        if not native_minimum:
            sim.minimizeEnergy(maxIterations=500)
        ctx.setVelocitiesToTemperature(300*u.kelvin, args.seed)
        ctx.applyVelocityConstraints(1e-5)
    backend = bias.getBackend(ctx) if bias else "unrestrained solvent leg"
    print("Backend", backend, nb_type, flush=True)
    print("Device", platform.getPropertyValue(ctx, "DeviceName"), flush=True)
    start = time.monotonic()
    if window:
        sample_window(args, sim, prepared, bias, params)
    else:
        for done in range(0, args.steps, 10000):
            sim.step(min(10000, args.steps-done))
            print("steps", min(done+10000, args.steps), flush=True)
    potential = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilocalorie_per_mole)
    elapsed = time.monotonic()-start
    if not math.isfinite(potential):
        raise ValueError("nonfinite final potential energy")
    if bias:
        bias.flushOutputs(ctx)
    save_state(ctx, args.out/"state.xml")
    if args.stage == "eq" and args.steps >= 2500000:
        fitted = fit_restraints(completed_trajectory(args.out, args.steps), args.inputs)
        (args.out/"restraints.json").write_text(json.dumps(fitted, indent=2))
    result = dict(stage=args.stage, index=args.index, mode=args.mode, backend=backend,
                  nonbonded=nb_type, steps=args.steps, elapsed_s=elapsed,
                  ns_day=args.steps*0.000002*86400/elapsed, potential_kcal=potential)
    (args.out/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
