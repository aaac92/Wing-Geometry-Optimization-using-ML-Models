"""
File-name length limits of the simulation binaries.

XFOIL and AVL read file names into fixed-length Fortran strings and silently cut whatever does not fit:
the file is then read from, or written to, a different (truncated) name and the caller never finds it.
Measured with the binaries in bin/ (2026-10-29):
    XFOIL  LOAD, PACC                 64 characters
    AVL    FT (force file)            80 characters  (LOAD and AFIL accepted at least 130)
Both programs are therefore started inside the folder they write to and given short relative names.
"""
import os
from pathlib import Path

XFOIL_MAX_NAME = 64
AVL_MAX_NAME = 80


def short_name(path, work_dir, limit, program="XFOIL"):
    """
    Shortest way to name `path` for a program running inside `work_dir`: relative to it or absolute.
    Raises ValueError if even the shortest form is longer than `limit` characters.
    """
    absolute = os.path.abspath(path)
    options = [absolute]
    try:
        options.append(os.path.relpath(absolute, os.path.abspath(work_dir)))
    except ValueError:  # Windows: different drives have no relative path
        pass
    name = min(options, key=len)
    if len(name) > limit:
        raise ValueError(f"{program} reads at most {limit} characters of a file name, and '{Path(path).name}' needs "
                         f"{len(name)} even from its working folder ({name}). Use a shorter file or folder name.")
    return name
