"""Build the two-leg BFEE3 LDDM protocol from prepared Amber input files."""

import importlib.util
import re

import numpy as np


CVS = ("eulerTheta", "eulerPhi", "eulerPsi", "polarTheta", "polarPhi", "r")
BOUNDS = ((-90, 90), (-180, 180), (-180, 180), (0, 180), (-180, 180))
KT = 0.0019872041 * 300  # kcal/mol


def templates(bfee):
    path = bfee / "BFEE2/templates_namd/configTemplate.py"
    spec = importlib.util.spec_from_file_location("bfee_namd_templates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configTemplate()


def groups(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.strip().startswith("["):
            group = result.setdefault(line.strip(" []"), [])
        elif line.strip():
            group.extend(int(i) - 1 for i in line.split())
    return result


def geometry(inputs):
    xyz = np.loadtxt(inputs / "complex.xyz", skiprows=2, usecols=(1, 2, 3))
    ids = groups(inputs / "complex.ndx")
    centers = {name: xyz[atoms].mean(axis=0) for name, atoms in ids.items()}
    distance = round(float(np.linalg.norm(centers["ligand"] - centers["reference"])), 1)
    return centers["protein"], distance


def cv_config(bfee, inputs, stage, params=None, fraction=0):
    template = templates(bfee)
    protein, distance = geometry(inputs)
    ref = str(inputs / "complex.xyz")
    text = template.cvHeadTemplate(inputs / "complex.ndx")
    text = re.sub(r"colvarsTrajFrequency\s+5000", "colvarsTrajFrequency 50", text)
    monitor = stage == "eq"
    if monitor:
        text += template.cvRMSDTemplate(True, 0, 10, ref, extendedLagrangian=False)
    for name, (low, high) in zip(CVS, BOUNDS):
        text += template.cvAngleTemplate(monitor, low, high, name, ref, True, False)
    text += template.cvRTemplate(monitor, max(0, distance - 5), distance + 22, False)
    if params:
        for name, center, k in zip(CVS, params["centers"], params["k"]):
            # The generator's harmonic template formats k to one decimal. LDDM's
            # subsequent per-window file generator preserves full fit precision.
            block = template.cvHarmonicTemplate(name, 1, center)
            block = re.sub(r"(forceConstant\s+)1\.0", lambda m: m[1] + str(k*fraction), block)
            block = re.sub(r"(centers\s+)[^\s]+", lambda m: m[1] + str(center), block)
            text += block
    return text + template.cvProteinTemplate(protein, ref)


def fit_restraints(trajectory, inputs):
    """BFEE3 parseDat and updateCenters, with finite-data checks for a short run."""
    _, distance = geometry(inputs)
    bounds = (*BOUNDS, (max(0, distance - 5), distance + 22))
    # eq trajectory starts with step, RMSD, then the six scalar CVs. The pin
    # includes a vector component later in the line, so read only scalar columns.
    data = np.loadtxt(trajectory, usecols=range(2, 8), ndmin=2)
    out = {"centers": [], "k": [], "frames": len(data)}
    for i, (low, high) in enumerate(bounds):
        width = 0.1 if CVS[i] == "r" else 1
        edges = np.linspace(low, high, round((high-low)/width)+1)
        hist, _ = np.histogram(data[:, i], edges)
        peak = int(hist.argmax())
        subset = np.array([peak-2, peak, peak+2])
        if min(subset) < 0 or max(subset) >= len(hist) or min(hist[subset]) == 0:
            raise ValueError(f"Insufficient histogram support for {CVS[i]}")
        centers = (edges[:-1]+edges[1:])/2
        k = float(np.polyfit(centers[subset], -np.log(hist[subset])/KT, 2)[0])
        if not np.isfinite(k) or k <= 0:
            raise ValueError(f"Invalid BFEE fit for {CVS[i]}: {k}")
        out["centers"].append(float(centers[peak]))
        out["k"].append(k)
    return out


def completed_trajectory(directory, expected_step):
    # The OpenMM minimizer can allocate a temporary Reference Context. Colvars
    # then rotates the original open trajectory to .BAK on Linux.
    for path in (directory/"cv.colvars.traj", directory/"cv.colvars.traj.BAK"):
        if path.exists():
            steps = np.loadtxt(path, usecols=0, ndmin=1)
            if len(steps) and steps[-1] == expected_step and np.all(np.diff(steps) == 50):
                return path
    raise ValueError("missing complete equilibration CV trajectory")


def standard_state(params):
    theta, _, _, polar, _, r = params["centers"]
    angular, radial = np.array(params["k"][:5]), params["k"][5]
    jac = r*r*np.cos(np.radians(theta))*np.sin(np.radians(polar))
    factor = np.sqrt(np.prod(angular*(180/np.pi)**2)*radial)
    return KT*np.log(8*np.pi**2*1661*factor/(jac*(2*np.pi*KT)**3))
