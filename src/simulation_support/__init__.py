import sys

from .key_point_processing import Key_Point_Extraction
from .geo_file_creation import (
    RecWingGeo_Creation,
    SkyeyeGeo_Creation,
    HossainWingRec_Creation,
    HossainWing60DEG_Creation,
    HossainWingFlat_Creation,
)

from .polar_creation import Xfoil_Polar_Creator
from .alpha_sweep import SWEEP_MODES, DEFAULT_SWEEP_MODE, SWEEP_REVISIONS

if sys.platform == "win32":
    from .avl_simulation import AVL_Run_Case
else:
    from .avl_simulation_linux import AVL_Run_Case