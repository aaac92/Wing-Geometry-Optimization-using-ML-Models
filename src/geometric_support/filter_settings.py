"""
Geometric filter settings shared by every script that filters generated airfoils: the dataset generation
(Hossain_DB_Gen.py), its unit tests and studies, the filter plots and the inference reports.

DATASET_FILTERS holds one entry per filter: an 'enabled' switch plus its thresholds (a threshold set to None
switches only that check off). The entries run in their order and the first failed filter rejects the airfoil.

    Disable a filter:   filter_settings(camber={"enabled": False})
    Change a threshold: filter_settings(thickness={"threshold_max_t": 0.20})
    Add a filter:       add its entry to DATASET_FILTERS and its check to FILTER_CHECKS, a function
                        (Validate_Airfoil, settings entry) -> ("valid" | "invalid", log line or None)

filter_parameters() gives the same settings in the older flat form (geo_filt_parm) that the plot functions in
custom_graphs.py read.
"""
import copy

from cst_modeling.foil import check_validity

from .geometric_filters import Validate_Airfoil

# Thresholds of the dataset 17 / 17.5 generation
DATASET_FILTERS = {
    "thickness": {"enabled": True, "threshold_min_t": 0.03, "threshold_max_t": 0.25,
                  "threshold_t_x_min": 0.18, "threshold_t_x_max": 0.45},
    "sharpness": {"enabled": True, "activate_curvature_cont": True, "threshold_rLE": 0.01},
    "monotonic_thickness": {"enabled": True},
    "camber": {"enabled": True, "threshold_max_camber": 0.12, "threshold_min_camber": -0.02},
}

FILTER_CHECKS = {
    "thickness": lambda va, s: va.thickness_filters(th_t_min=s.get("threshold_min_t"), th_t_max=s.get("threshold_max_t"),
                                                    th_t_x_max=s.get("threshold_t_x_max"), th_t_x_min=s.get("threshold_t_x_min")),
    "sharpness": lambda va, s: va.sharpness_filter(curv_cont=bool(s.get("activate_curvature_cont")), th_rLE=s.get("threshold_rLE")),
    "monotonic_thickness": lambda va, s: va.monotonic_thickness_filter(),
    "camber": lambda va, s: va.camber_filters(th_c_max=s.get("threshold_max_camber"), th_c_min=s.get("threshold_min_camber")),
}


def filter_settings(base: dict = None, **changes) -> dict:
    """A copy of `base` (default DATASET_FILTERS) with some entries changed, e.g. filter_settings(camber={"enabled": False})."""
    settings = copy.deepcopy(DATASET_FILTERS if base is None else base)
    for name, update in changes.items():
        if name not in settings:
            raise KeyError(f"Unknown filter '{name}'. Known: {list(settings)}")
        settings[name].update(update)
    return settings


def enabled_filters(settings: dict = None) -> list:
    """Names of the enabled filters, in run order."""
    settings = DATASET_FILTERS if settings is None else settings
    return [name for name, entry in settings.items() if entry.get("enabled", True)]


def filter_parameters(settings: dict = None) -> dict:
    """The settings in the flat geo_filt_parm form; a disabled filter gives None thresholds and False switches."""
    settings = DATASET_FILTERS if settings is None else settings
    flat = {}
    for name, entry in settings.items():
        on = entry.get("enabled", True)
        for key, value in entry.items():
            if key != "enabled":
                flat[key] = value if on else (False if isinstance(value, bool) else None)
    flat["monotonic_activation"] = bool(settings.get("monotonic_thickness", {}).get("enabled", False))
    return flat


def describe_filters(settings: dict = None) -> list:
    """One readable line per filter (for logs and reports)."""
    settings = DATASET_FILTERS if settings is None else settings
    lines = []
    for name, entry in settings.items():
        values = ", ".join(f"{k} = {v}" for k, v in entry.items() if k != "enabled")
        lines.append(f"{name}: {'on' if entry.get('enabled', True) else 'OFF'}" + (f" ({values})" if values else ""))
    return lines


def apply_filters(validator: Validate_Airfoil, settings: dict = None, stop_at_first: bool = True):
    """
    Runs the enabled filters on one airfoil. Returns (status, log line, results):
      status   "valid" or "invalid" (the first failed filter decides)
      log line the failed filter's line for the invalid-airfoil log (None when valid)
      results  [{"filter", "passed", "reason"}] for every filter that ran (all of them with stop_at_first=False)
    """
    settings = DATASET_FILTERS if settings is None else settings
    status, first_line, results = "valid", None, []
    for name in enabled_filters(settings):
        if name not in FILTER_CHECKS:
            raise KeyError(f"Filter '{name}' has settings but no check in FILTER_CHECKS.")
        outcome, line = FILTER_CHECKS[name](validator, settings[name])
        passed = outcome == "valid"
        reason = line.split("Reason:", 1)[1].strip() if (line and "Reason:" in line) else (line.strip() if line else None)
        results.append({"filter": name, "passed": passed, "reason": None if passed else reason})
        if not passed and status == "valid":
            status, first_line = "invalid", line
            if stop_at_first:
                break
    return status, first_line, results


def check_airfoil(x, yu, yl, cst, n_cst: int, name: str = "airfoil", settings: dict = None, stop_at_first: bool = False) -> dict:
    """
    Full geometric check of one airfoil: check_validity (crossing or too thin surfaces) and then every enabled
    filter. Returns {"valid_geometry", "passed", "filters": [...], "reason"}; "passed" needs both.
    """
    if not check_validity(x, yu, yl, verbose=False):
        return {"valid_geometry": False, "passed": False, "filters": [], "reason": "check_validity failed"}
    try:
        validator = Validate_Airfoil(x=x, yu=yu, yl=yl, cst_coeffs=list(cst), n_cst=n_cst, name=name, verbose=False)
    except ValueError as e:
        return {"valid_geometry": True, "passed": False, "filters": [], "reason": str(e).split("Reason:", 1)[-1].strip()}
    status, _line, results = apply_filters(validator, settings, stop_at_first=stop_at_first)
    failed = [r for r in results if not r["passed"]]
    return {"valid_geometry": True, "passed": status == "valid", "filters": results,
            "reason": "; ".join(f"{r['filter']}: {r['reason']}" for r in failed) or None}
