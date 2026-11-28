"""Run shared equilibration, then compare three LDDM implementations on one GPU."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from analyze import summarize


def solvent_inputs(args, root):
    if args.solvent_inputs:
        return args.solvent_inputs.resolve()
    inputs = args.inputs.resolve()
    strip_script = inputs/"002.3.1_removeProtein.cpptraj"
    if (inputs/"ligandOnly.parm7").exists() or not strip_script.exists():
        return inputs
    # BFEE3 can provide the solvent coordinates plus a cpptraj strip recipe
    # instead of shipping a second Amber topology.
    prepared = root/"solvent-inputs"
    if (prepared/"ligandOnly.parm7").is_file():
        return prepared
    cpptraj = args.cpptraj or shutil.which("cpptraj")
    if not cpptraj:
        raise ValueError("Supply --cpptraj or --solvent-inputs to prepare the BFEE3 solvent topology")
    masks = re.findall(r"^\s*strip\s+([^\s]+)\s+parmout\s+\S+\s*$", strip_script.read_text(), re.M|re.I)
    if len(masks) != 1:
        raise ValueError(f"Expected one strip mask in {strip_script}")
    prepared.mkdir(exist_ok=True)
    for name in ("complex.parm7", "complex.pdb", "ligandOnly.pdb", "fep_ligandOnly.pdb"):
        dest = prepared/name
        if not dest.exists():
            shutil.copyfile(inputs/name, dest)
    recipe = ("parm complex.parm7\ntrajin complex.pdb\n"
              f"strip {masks[0]} parmout building.parm7\n"
              "trajout cpptraj-check.pdb\ngo\n")
    (prepared/"strip.cpptraj").write_text(recipe)
    with (prepared/"cpptraj.log").open("w") as log:
        subprocess.run([str(cpptraj), "-i", "strip.cpptraj"], cwd=prepared, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    (prepared/"building.parm7").replace(prepared/"ligandOnly.parm7")
    return prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("inputs", "bfee", "out"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--solvent-inputs", type=Path, help="directory containing the prepared solvent topology, coordinates and flags")
    p.add_argument("--cpptraj", help="cpptraj executable for a BFEE3 solvent topology strip recipe")
    p.add_argument("--mode", choices=("all", "cpu", "gpu", "cupotential"), default="all",
                   help="run a single backend throughout, or compare all three from shared preparation")
    p.add_argument("--window-steps", type=int, default=20000)
    p.add_argument("--discard", type=int, default=5000)
    p.add_argument("--bound-intervals", type=int, default=10)
    p.add_argument("--solvent-intervals", type=int, default=5)
    args = p.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    signature = dict(inputs=str(args.inputs.resolve()), bfee=str(args.bfee.resolve()),
                     window_steps=args.window_steps, discard=args.discard,
                     bound_intervals=args.bound_intervals, solvent_intervals=args.solvent_intervals)
    modes = ("cpu", "gpu", "cupotential") if args.mode == "all" else (args.mode,)
    prep_mode = "gpu" if args.mode == "all" else args.mode
    if args.mode != "all":
        signature["mode"] = args.mode
    solvent = solvent_inputs(args, root)
    signature["solvent_inputs"] = str(solvent)
    manifest = root/"workflow.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        p.error("this output directory was prepared with different inputs or settings")
    manifest.write_text(json.dumps(signature, indent=2))
    def run(stage, mode, directory, *, state=None, params=None, index=0, count=1):
        window = stage in ("solvent", "bound")
        steps = args.window_steps if window else 2500000
        result = directory/"result.json"
        if result.exists():
            data = json.loads(result.read_text())
            if (data["steps"], data["mode"], data["stage"], data["index"]) != (steps, mode, stage, index):
                raise ValueError(f"incompatible result in {directory}")
            if not (directory/"state.xml").is_file():
                raise ValueError(f"incomplete state in {directory}")
            return
        directory.parent.mkdir(parents=True, exist_ok=True)
        input_dir = str(solvent) if stage.startswith("solvent") else signature["inputs"]
        cmd = [sys.executable, "-u", str(Path(__file__).with_name("run.py")), "--inputs", input_dir,
               "--bfee", signature["bfee"], "--stage", stage, "--mode", mode, "--out", str(directory),
               "--steps", str(steps), "--intervals", str(count), "--index", str(index),
               "--seed", str(39173+index+100*("eq", "reeq", "solvent-eq", "bound", "solvent").index(stage))]
        if window:
            cmd += ["--discard", str(args.discard)]
        for key, value in (("--state", state), ("--params", params)):
            if value:
                cmd += [key, str(value)]
        print(f"Running {directory}", flush=True)
        with directory.with_suffix(".log").open("x") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)

    params = root/"eq/restraints.json"
    run("eq", prep_mode, root/"eq")
    run("solvent-eq", prep_mode, root/"solvent-eq")
    run("reeq", prep_mode, root/"reeq", state=root/"eq/state.xml", params=params)
    intervals = (args.bound_intervals, args.solvent_intervals)
    for leg, count in zip(("bound", "solvent"), intervals):
        start = root/("reeq" if leg == "bound" else "solvent-eq")/"state.xml"
        for index in range(count+1):
            for mode in modes:
                dest = root/mode/f"{leg}-{index:02d}"
                state = dest.parent/f"{leg}-{index-1:02d}"/"state.xml" if index else start
                run(leg, mode, dest, state=state, params=params if leg=="bound" else None,
                    index=index, count=count)
    result = summarize(root, intervals, modes)
    (root/"summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
