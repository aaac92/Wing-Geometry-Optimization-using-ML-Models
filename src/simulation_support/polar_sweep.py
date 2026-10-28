"""
Alpha-sweep strategies shared by both Xfoil_Polar_Creator implementations (Windows and Linux).

'sequential' is the original behavior: one XFOIL process walks the alpha sequence in the given order,
so every angle starts from the boundary layer solution of the previous one.

'zero_out' runs two independent XFOIL processes that both start at 0 deg: one walks up to the highest
angle and the other walks down to the lowest. Neither branch starts from a deep-stall solution. The two
branch polars are merged into the requested polar file, sorted by alpha.
"""
from pathlib import Path

SWEEP_MODES = ("sequential", "zero_out")
COMPLETION_FLAG = "XFOIL process completed successfully"


def zero_out_branches(alfa_sequence):
    """Splits an alpha sequence into an ascending branch and a descending branch, both starting at 0 deg."""
    angles = sorted(set(float(a) for a in alfa_sequence))
    positive = [a for a in angles if a >= 0]
    negative = sorted((a for a in angles if a < 0), reverse=True)
    if negative:
        negative = [0.0] + negative
    return [branch for branch in (positive, negative) if branch]


def _split_polar(path):
    """Returns (header lines up to the dashed separator, {alpha: data line}) of an XFOIL polar file."""
    lines = Path(path).read_text().splitlines()
    separator = next((i for i, line in enumerate(lines) if line.strip().startswith("------")), None)
    if separator is None:
        return lines, {}
    rows = {}
    for line in lines[separator + 1:]:
        if COMPLETION_FLAG in line or not line.strip():
            continue
        try:
            alpha = float(line.split()[0])
        except (ValueError, IndexError):
            continue
        rows.setdefault(alpha, line)
    return lines[:separator + 1], rows


def merge_polar_files(branch_paths, output_path):
    """
    Merges branch polar files into one polar sorted by alpha. The header comes from the first branch that
    has one; for an angle present in several branches, the first branch's row is kept.
    """
    header, merged = None, {}
    for path in branch_paths:
        if not Path(path).exists():
            continue
        branch_header, rows = _split_polar(path)
        if header is None and rows:
            header = branch_header
        for alpha, line in rows.items():
            merged.setdefault(alpha, line)
    if header is None:
        return False
    body = [merged[a] for a in sorted(merged)]
    Path(output_path).write_text("\n".join(header + body) + "\n")
    return True


def run_zero_out_sweep(runner, airfoil_file_path, polar_file_path, alfa_sequence, visc, mach, iter):
    """
    Runs every zero_out branch through runner.run_simulation (sequential mode) into its own branch file,
    merges them into polar_file_path and appends the completion flag. Returns True if any branch produced data.
    """
    polar_file_path = Path(polar_file_path)
    branch_paths = []
    for i, branch in enumerate(zero_out_branches(alfa_sequence)):
        branch_path = polar_file_path.with_name(f"{polar_file_path.stem}_b{i}{polar_file_path.suffix}")
        if branch_path.exists():
            branch_path.unlink()
        runner.run_simulation(airfoil_file_path, str(branch_path), branch, visc=visc, mach=mach, iter=iter)
        branch_paths.append(branch_path)
    merged = merge_polar_files(branch_paths, polar_file_path)
    for branch_path in branch_paths:
        if branch_path.exists():
            branch_path.unlink()
    if merged:
        runner.completion_flag(str(polar_file_path))
    return merged
