def __getattr__(name):
    if name == "Progress_GUI":
        from .GUI import Progress_GUI
        return Progress_GUI
    if name in ("run_with_progress", "display_available"):
        from . import progress
        return getattr(progress, name)
    if name in ("recovery_json", "recovery_Fabius"):
        from .recovery import recovery_json, recovery_Fabius
        return {"recovery_json": recovery_json, "recovery_Fabius": recovery_Fabius}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")