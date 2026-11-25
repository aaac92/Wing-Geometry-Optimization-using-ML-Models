"""
Checks of the AVL wing geometries, made after the rec wing of datasets 17 and 17.5 was found with twice its span
(its tip was placed at the full span and the surface was mirrored).

  reference_check   an .avl file: the area and span of the wing AVL solves (mirrored if it has YDUPLICATE, projected on
                    the XY plane) against the Sref and Bref of its header
  check_wings       reference_check of every wing configuration, without AVL (one file each, in a temp folder)
  zero_lift_check   NACA 0012 at zero lift, with AVL: there is no induced drag, so CD must be the section drag of the
                    CDCL polar times true area / Sref. Finds a wrong span and a wrong section polar

Both raise GeometryCheckError. check_wings runs before stage 3 of Hossain_DB_Gen and before every batch of simulations,
zero_lift_check before stage 4.

Unit test (this file's __main__ block)
---------
    python src/simulation_support/geometry_checks.py

Compares CD (NACA 0012) and E (five designs) of every configuration with reference values of the corrected wings
(AVL 3.40; AVL 3.32 must agree within 0.5 %), and checks that a doubled span and the legacy winglet polar are refused.
"""
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # run as a script (unit test): make the src packages importable
    sys.path.insert(0, str(SRC_DIR))

from simulation_support.key_point_processing import key_points_result, read_key_points_file  # noqa: E402

REFERENCE_TOLERANCE = 0.01   # area and span against Sref and Bref
ZERO_LIFT_TOLERANCE = 0.01   # CD at zero lift against the section drag
ZERO_LIFT_CL = 1e-3          # |CL| below this counts as zero lift


class GeometryCheckError(RuntimeError):
    """A wing geometry that does not match its reference values or its expected zero-lift drag."""


def parse_avl(path) -> dict:
    """Sref, Cref, Bref, whether the surface is mirrored, and the sections (x, y, z, chord) of a one-surface .avl file."""
    lines = [line.split("!")[0].strip() for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[1:]]
    lines = [line for line in lines if line and not line.startswith("#")]
    sref, cref, bref = (float(v) for v in lines[2].split()[:3])   # after the Mach number and the symmetry flags
    sections = [tuple(float(v) for v in lines[i + 1].split()[:4]) for i, line in enumerate(lines) if line == "SECTION"]
    return {"Sref": sref, "Cref": cref, "Bref": bref, "mirrored": "YDUPLICATE" in lines, "sections": sections}


def wing_areas(geometry: dict) -> dict:
    """Projected area, true (developed) area and span of the wing AVL solves."""
    s = np.array(geometry["sections"])
    dy, dz, chord = np.diff(s[:, 1]), np.diff(s[:, 2]), (s[1:, 3] + s[:-1, 3]) / 2
    sides = 2 if geometry["mirrored"] else 1
    return {"projected": float(sides * np.sum(chord * dy)), "true": float(sides * np.sum(chord * np.hypot(dy, dz))),
            "span": float(sides * np.ptp(s[:, 1]))}


def reference_check(path) -> dict:
    """Area and span of an .avl file against its Sref and Bref. Returns the values; raises GeometryCheckError."""
    geometry = parse_avl(path)
    areas = wing_areas(geometry)
    for name, value, reference, unit in (("an area of", areas["projected"], geometry["Sref"], "m2"),
                                         ("a span of", areas["span"], geometry["Bref"], "m")):
        if abs(value / reference - 1) > REFERENCE_TOLERANCE:
            raise GeometryCheckError(f"{Path(path).name}: the wing has {name} {value:.5f} {unit} but its header says "
                                     f"{reference:.5f} ({value / reference - 1:+.1%}). Its coefficients would be wrong.")
    return {**geometry, **areas}


def write_wing(creator, identifier: str, name: str, dat_path, kp, naca_kp_file, out_dir) -> Path:
    """Writes the .avl file of one airfoil and wing configuration; returns its path."""
    maker = creator(key_points_data=[{'file_path': f"{name}.txt", **key_points_result(kp)}], airfoil_abs_paths=[str(dat_path)],
                    naca_key_points_data=str(naca_kp_file), output_dir=str(out_dir), verbose=False, pb=False)
    if maker.create_geo_files():
        raise GeometryCheckError(f"{identifier}: the geometry file could not be written: {maker.errors[0]['error']}")
    return Path(out_dir) / f"{name}_{identifier}_geo.avl"


def check_wings(creators: dict, naca_kp_file) -> None:
    """reference_check of every wing configuration ({identifier: creator class}). Needs no airfoil, XFOIL or AVL."""
    kp = read_key_points_file(naca_kp_file)
    with tempfile.TemporaryDirectory() as folder:
        for identifier, creator in creators.items():
            reference_check(write_wing(creator, identifier, "check", Path(folder) / "check.dat", kp, naca_kp_file, folder))


def section_drag_at_zero_lift(kp) -> float:
    """CD of AVL's CDCL polar (two parabolas joined at the minimum-drag point) at cl = 0."""
    cl_neg, cd_neg, cl_min, cd_min, cl_pos, cd_pos = kp
    cl_end, cd_end = (cl_pos, cd_pos) if cl_min < 0 else (cl_neg, cd_neg)
    return cd_min + (cd_end - cd_min) * (cl_min / (cl_end - cl_min)) ** 2


def run_wing(settings, creator, identifier: str, name: str, cst, kp, run_dir) -> dict:
    """One AVL run of one wing from given CST increments and key points (no XFOIL): CL, CD, e and the wing's areas."""
    from geometric_support import Airfoil_Generator
    from simulation_support.airfoil_simulation import read_force_file, read_span_efficiency
    from simulation_support.avl_simulation import AVL_Run_Case
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    generator = Airfoil_Generator(base_airfoil_csv=settings.base_airfoil, n_cst=settings.n_cst, verbose=False)
    x, yu, yl = generator.generate(cst_coeffs=np.asarray(cst, dtype=float))
    generator._create_dat_file(output_dir=run_dir, n_cst=settings.n_cst, x=x, yu=yu, yl=yl, file_name=name)
    geo = write_wing(creator, identifier, name, run_dir / f"{name}.dat", kp, settings.naca_kp_file, run_dir)
    case = run_dir / Path(settings.case_file).name
    shutil.copy(settings.case_file, case)
    force = run_dir / f"force_{name}_{identifier}.txt"
    force.unlink(missing_ok=True)
    AVL_Run_Case(geo_files=[str(geo)], case_file=str(case), avl_exe_path=settings.avl_exe, output_dir=run_dir).run()
    cl, cd = read_force_file(force)
    if not (math.isfinite(cl) and math.isfinite(cd)):
        raise GeometryCheckError(f"{identifier}: AVL wrote no forces for the check.")
    geometry = parse_avl(geo)
    return {"CL": cl, "CD": cd, "e": read_span_efficiency(force), **geometry, **wing_areas(geometry)}


def zero_lift_check(settings, creators: dict, run_dir, kp=None) -> dict:
    """
    NACA 0012 (every CST increment 0) at alpha 0 in every configuration. kp: key points of the wing sections (default:
    the ones of the winglet file). Returns {identifier: (CD, expected CD)}; raises GeometryCheckError.
    """
    kp = list(kp) if kp is not None else read_key_points_file(settings.naca_kp_file)
    results = {}
    for identifier, creator in creators.items():
        run = run_wing(settings, creator, identifier, "zl", [0.0] * (2 * settings.n_cst), kp, run_dir)
        if abs(run["CL"]) > ZERO_LIFT_CL:
            raise GeometryCheckError(f"{identifier}: CL = {run['CL']:.5f} for a symmetric airfoil at alpha 0.")
        expected = section_drag_at_zero_lift(kp) * run["true"] / run["Sref"]
        if abs(run["CD"] / expected - 1) > ZERO_LIFT_TOLERANCE:
            raise GeometryCheckError(f"{identifier}: CD = {run['CD']:.5f} at zero lift, expected {expected:.5f} "
                                     f"({run['CD'] / expected - 1:+.1%}): wrong wing area or wrong section polar.")
        results[identifier] = (run["CD"], expected)
    return results


# =============================================================================================== #
# === Unit test: python src/simulation_support/geometry_checks.py === #
# =============================================================================================== #

# CST increments and zero_out key points of five designs, and their E on the corrected wings
DESIGNS = {
    "Best of dataset 17.5 (s12a757572c)": {
        "cst": [0.0, 0.125, 0.0417, 0.125, 0.125, 0.125, 0.0417, 0.0833, 0.0833, 0.125],
        "kp": [-0.7766623318094329, 0.03332971679036528, 0.40387339191871696, 0.0085825776570303, 1.4921920679893361, 0.04751552452471319],
        "E": {"rec": 20.22, "flat": 21.81, "60DEG": 21.02}},
    "Marbas-C1 (s5b1f793f71)": {
        "cst": [0.03756614774465561, 0.15336018800735474, 0.09993980824947357, 0.12224414199590683, 0.25822874903678894, -0.01945602335035801, 0.08783352375030518, -0.04088135436177254, 0.03393779695034027, 0.11773297190666199],
        "kp": [-0.5427703557909666, 0.032581628997267416, 0.40918925297546765, 0.007838099765328154, 1.3712921246573828, 0.04608627650176996],
        "E": {"rec": 22.08, "flat": 24.25, "60DEG": 23.36}},
    "Consenso-B2 (s2c77575240)": {
        "cst": [0.020432965829968452, 0.13516604900360107, 0.07660336047410965, 0.1550910472869873, 0.2041371464729309, 0.027693839743733406, 0.09602285921573639, 0.026176897808909416, 0.07350659370422363, 0.15974119305610657],
        "kp": [-0.6398644093444668, 0.03857266475655763, 0.4961796585777747, 0.0081811887623644, 1.4157892661786864, 0.05267998543831709],
        "E": {"rec": 21.35, "flat": 23.96, "60DEG": 23.1}},
    "Marbas-A1 (s8f20f950b8)": {
        "cst": [0.004700043238699436, 0.13248838484287262, 0.07051019370555878, 0.1511683464050293, 0.19142235815525055, 0.007156571839004755, 0.09912434965372086, 0.051395244896411896, 0.06735028326511383, 0.1955050677061081],
        "kp": [-0.6924992423094878, 0.031239409261537787, 0.5312825405668349, 0.00831484607229882, 1.3739809953683277, 0.0569483999559921],
        "E": {"rec": 21.37, "flat": 23.97, "60DEG": 23.09}},
    "Aleatorio-C2 (s9b942877ab)": {
        "cst": [0.0001443471119273454, 0.12940029799938202, 0.04279837757349014, 0.1810569018125534, 0.27203279733657837, 0.002018689876422286, 0.1615927368402481, 0.08742392063140869, 0.12858407199382782, 0.2573127746582031],
        "kp": [-0.6848180518334956, 0.041929223535598366, 0.5941730157182268, 0.00991985803082838, 1.4331897576103998, 0.07444028683698231],
        "E": {"rec": 19.75, "flat": 23.28, "60DEG": 22.42}},
}
NACA_0012_CD = {"rec": 0.00868, "flat": 0.00865, "60DEG": 0.00974}
LEGACY_KP = [-1.0392755591564935, 0.029713337593332056, 0.010281803652329558, 0.0012786983441526178, 1.0379894500713305,
             0.02552940093777245]
ACCEPTANCE_TOLERANCE = 0.005


def unit_test() -> bool:
    from dataclasses import replace
    from simulation_support import GEO_CREATORS_BY_IDENTIFIER, protocol_settings
    from simulation_support.key_point_processing import write_key_points_file
    from simulation_support.sweep_protocol_check import default_temp_root
    creators, settings = GEO_CREATORS_BY_IDENTIFIER, protocol_settings("zero_out")
    kp = read_key_points_file(settings.naca_kp_file)
    run_dir = default_temp_root() / "geo_check"
    shutil.rmtree(run_dir, ignore_errors=True)
    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def check_refused(name, function):
        try:
            function()
            check(name, False, "it was accepted")
        except GeometryCheckError as e:
            check(name, True, f"refused | {e}")

    print(f"Wing geometry checks | AVL: {settings.avl_exe} | winglet key points: {Path(settings.naca_kp_file).name}")
    try:
        check_wings(creators, settings.naca_kp_file)
        check("area and span", True, "match Sref and Bref in " + ", ".join(creators))
        for ident, (cd, expected) in zero_lift_check(settings, creators, run_dir).items():
            check(f"NACA 0012, zero lift, {ident}", abs(cd / NACA_0012_CD[ident] - 1) <= ACCEPTANCE_TOLERANCE,
                  f"CD {cd:.5f} (section drag {expected:.5f}, reference {NACA_0012_CD[ident]:.5f})")
        for name, design in DESIGNS.items():
            for ident, reference in design["E"].items():
                run = run_wing(settings, creators[ident], ident, "ds", design["cst"], design["kp"], run_dir)
                e = run["CL"] / run["CD"]
                check(f"{name}, {ident}", abs(e / reference - 1) <= ACCEPTANCE_TOLERANCE,
                      f"E {e:.2f} (reference {reference:.2f}, {e / reference - 1:+.2%}) | CL {run['CL']:.4f} "
                      f"CD {run['CD']:.5f} e {run['e']:.3f}")
        geo = write_wing(creators["rec"], "rec", "bad", run_dir / "bad.dat", kp, settings.naca_kp_file, run_dir)
        text = geo.read_text(encoding="utf-8")
        geo.write_text(text.replace("  0.0  0.33  0.0  0.121  0.0", "  0.0  0.66  0.0  0.121  0.0"), encoding="utf-8")
        check_refused("doubled rec span", lambda: reference_check(geo))
        legacy = run_dir / "legacy_kp.txt"
        write_key_points_file(key_points_result(LEGACY_KP), legacy, {"name": "legacy", "mach": "0.093", "re": "250000", "ncrit": "9"})
        check_refused("legacy winglet polar", lambda: zero_lift_check(replace(settings, naca_kp_file=str(legacy)),
                                                                      {"flat": creators["flat"]}, run_dir, kp=kp))
    except GeometryCheckError as e:
        check("stopped", False, str(e))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    failed = results.count(False)
    print("\n" + (f"ALL {len(results)} CHECKS PASSED" if not failed else f"{failed} OF {len(results)} CHECKS FAILED"))
    return not failed


if __name__ == "__main__":
    sys.exit(0 if unit_test() else 1)
