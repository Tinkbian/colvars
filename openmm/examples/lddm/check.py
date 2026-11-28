"""Same-coordinate energy and force comparisons before timing an LDDM build."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from openmm_fep import FEPConfig, FEPAtoms
from openmm_fep.openmm import set_context_lambda
from run import (ColvarsForce, base_system, cv_config, input_names, install_cupotential,
                 mm, prepare_lddm_system, u)


def evaluate(args, mode, solvent):
    pdb, system = base_system(args.inputs, solvent)
    flag = args.inputs / input_names(args.inputs, solvent)[2]
    cfg = FEPConfig(FEPAtoms.from_alch_file(flag), (1, 0), (1,), alch_vdw_lambda_end=0.7)
    prepared = prepare_lddm_system(system, cfg)
    if mode == "cupotential":
        install_cupotential(prepared.system)
    bias = None
    if not solvent:
        values = {"centers": [1, 2, 3, 55, 85, 11], "k": [.1]*5+[10]}
        text = cv_config(args.bfee, args.inputs, "bound", values, 0.4)
        bias = ColvarsForce.fromConfigString(text, str(args.out/mode))
        bias.setUnitSystem("real")
        bias.setUseGpu(mode != "cpu")
        bias.setForceGroup(30)
        prepared.system.addForce(bias)
    ctx = mm.Context(prepared.system, mm.VerletIntegrator(0.002),
                     mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed", "DeviceIndex": "0"})
    if args.namd_coor:
        raw = args.namd_coor.read_bytes()
        xyz = np.frombuffer(raw, dtype="<f8", offset=4).reshape(-1, 3)
        if len(xyz) != len(pdb.positions) or int.from_bytes(raw[:4], "little") != len(xyz):
            raise ValueError("wrong NAMD coordinate file for topology")
        ctx.setPositions(xyz*u.angstrom)
    else:
        ctx.setPositions(pdb.positions)
    print(mode, bias.getBackend(ctx) if bias else "solvent", flush=True)
    energies, forces = [], []
    for lam in (1, .75, .5, .25, 0):
        set_context_lambda(ctx, cfg, lam)
        state = ctx.getState(getEnergy=True, getForces=True)
        energies.append(state.getPotentialEnergy().value_in_unit(u.kilocalorie_per_mole))
        forces.append(state.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole/u.nanometer))
    return np.array(energies), np.array(forces)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "out", "bfee"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--leg", choices=("bound", "solvent"), required=True)
    p.add_argument("--namd-coor", type=Path)
    args = p.parse_args()
    args.inputs = args.inputs.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    reference = None
    rows = []
    for mode in ("cpu", "gpu", "cupotential"):
        energies, forces = evaluate(args, mode, args.leg=="solvent")
        gc.collect()
        if not np.isfinite(energies).all() or not np.isfinite(forces).all():
            raise ValueError("non-finite observation")
        if reference is None:
            reference = energies, forces
        delta = np.diff(energies) - np.diff(reference[0])
        relative = np.linalg.norm(forces-reference[1])/np.linalg.norm(reference[1])
        row = dict(mode=mode, kcal=energies.tolist(), neighbour_error_kcal=float(max(abs(delta))),
                   relative_force_error=float(relative), max_force_error=float(np.max(abs(forces-reference[1]))))
        rows.append(row)
        print(row, flush=True)
        (args.out/"energies.json").write_text(json.dumps(rows, indent=2))
        if max(abs(delta)) > 0.05 or relative > 0.001:
            raise ValueError("backends disagree on the same coordinates")


if __name__ == "__main__":
    main()
