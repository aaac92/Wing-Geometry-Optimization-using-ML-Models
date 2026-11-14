"""
Plots one airfoil the way the filter plots draw it (filter_plotter.py), without the filters.

Two modes, both through Airfoil_Generator, so the plot shows the airfoil as the pipeline sees it (CST fit, 150 points):
    file  an Airfoiltools.com CSV (e.g. data/support/naca2412_unit.csv), fitted with n_cst CST coefficients per surface
    cst   a base airfoil CSV plus 2 x n_cst CST increments (lower surface first), as the dataset generation applies them

Usage
-----
    python src/utilities/airfoil_plotter.py --mode file --file naca2412_unit.csv
    python src/utilities/airfoil_plotter.py --mode cst --base naca0012_unit.csv --n-cst 5 --cst -0.0625 0.0417 0 0 0.125 0.0625 0.0833 0.0417 0 0.125
"""
import argparse
import sys
from pathlib import Path

utilities_dir = Path(__file__).resolve().parent
src_dir = utilities_dir.parent
project_root = src_dir.parent
support_files_dir = project_root / "data" / "support"
sys.path.insert(0, str(src_dir))

from geometric_support import Airfoil_Generator  # noqa: E402
from geometric_support.custom_graphs import plot_airfoil  # noqa: E402

# Same palette as the airfoil in the filter plots (TF_COLORS / C_COLORS of filter_plotter.py)
AF_COLORS = {
    'af_airfoil': '#484C68',
    'af_fill': '#FCF0DA',
}


def resolve_csv(arg: str) -> Path:
    """An airfoil CSV given as written, or inside data/support."""
    path = Path(arg)
    if not path.exists() and (support_files_dir / path).exists():
        path = support_files_dir / path
    if not path.exists():
        sys.exit(f"[ERROR] Airfoil file not found: {arg}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot one airfoil like the filter plots, from a file or from CST increments.")
    parser.add_argument('--mode', required=True, choices=['file', 'cst'],
                        help="file: plot an airfoil CSV; cst: plot a base airfoil modified by CST increments.")
    parser.add_argument('--file', help="Airfoil CSV for --mode file (path, or a file name in data/support).")
    parser.add_argument('--base', default="naca0012_unit.csv", help="Base airfoil CSV for --mode cst (default: naca0012_unit.csv).")
    parser.add_argument('--n-cst', type=int, default=5, help="CST coefficients per surface (default: 5).")
    parser.add_argument('--cst', type=float, nargs='+', help="2 x n_cst CST increments for --mode cst, lower surface first.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == 'file':
        if not args.file:
            sys.exit("[ERROR] --mode file needs --file.")
        airfoil_path = resolve_csv(args.file)
        cst = [0.0] * (2 * args.n_cst)   # no increments: the airfoil of the file itself
    else:
        if not args.cst or len(args.cst) != 2 * args.n_cst:
            sys.exit(f"[ERROR] --mode cst needs --cst with {2 * args.n_cst} values (2 x n_cst, lower surface first).")
        airfoil_path = resolve_csv(args.base)
        cst = args.cst

    airfoil = Airfoil_Generator(base_airfoil_csv=str(airfoil_path), n_cst=args.n_cst, x_len=150, verbose=False)
    x, zu, zl = airfoil.generate(cst_coeffs=cst)

    name = airfoil.airfoil_base_name or airfoil_path.stem
    title = name if args.mode == 'file' else f"{name} modificado"
    print(f"{title} | {airfoil_path.name} | n_cst {args.n_cst}" + (f" | CST {cst}" if args.mode == 'cst' else ""))
    plot_airfoil(x, zu, zl, title=title, colors=AF_COLORS)
