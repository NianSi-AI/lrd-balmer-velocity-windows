"""Build the velocity-window results and figures in an empty build directory."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="Also typeset the manuscript with latexmk")
    args = parser.parse_args()
    build = ROOT / "build"
    if build.exists() and any(build.iterdir()):
        parser.error("Use a fresh checkout or move the existing build directory aside.")
    tables = build / "tables"
    tables.mkdir(parents=True)
    for name in ("protocol.json", "sensitivity_protocol.json"):
        shutil.copyfile(ROOT / "data" / name, tables / name)
    env = {**os.environ, "MPLBACKEND":"Agg", "MPLCONFIGDIR":str(build/"mpl-cache"),
           "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "VECLIB_MAXIMUM_THREADS":"1"}
    for script, arguments in [
        ("phase2_cumulative_moments", ["inject", tables]),
        ("phase2_cumulative_moments", ["measure", tables]),
        ("phase2_wing_sensitivity", ["inject", tables]),
        ("phase2_wing_sensitivity", ["measure", tables]),
        ("phase2_moment_results", [tables]),
        ("edge_comparison", [tables]),
        ("balmer_windows", []),
        ("paper_numbers", []),
        ("figures", []),
    ]:
        subprocess.run([sys.executable, ROOT/"src"/(script+".py"), *arguments], check=True, env=env)
    for name in ("main.tex", "references.bib", "Orcid-ID.png"):
        shutil.copyfile(ROOT / "manuscript" / name, build / name)
    if args.pdf:
        subprocess.run(["latexmk","-pdf","-halt-on-error","-interaction=nonstopmode","main.tex"],
                       cwd=build, check=True, env=env)


if __name__ == "__main__":
    main()
