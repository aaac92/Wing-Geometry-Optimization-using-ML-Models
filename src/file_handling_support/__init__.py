def __getattr__(name):
    if name in ("get_all_files_from", "compression_Alma"):
        from .file_handling import get_all_files_from, compression_Alma
        return {"get_all_files_from": get_all_files_from, "compression_Alma": compression_Alma}[name]
    if name == "Training_Data_Generator":
        from .trining_csv_creator import Training_Data_Generator
        return Training_Data_Generator
    if name in ("Dataset_Rectifier", "grid_levels_from_ranges", "display_available"):
        from . import database_correction
        return getattr(database_correction, name)
    if name == "Simulation_Settings":  # moved to simulation_support.airfoil_simulation; kept here for older imports
        from simulation_support.airfoil_simulation import Simulation_Settings
        return Simulation_Settings
    if name in ("Database_Corrector", "AccelerationConfig"):
        from .database_correction import Database_Corrector, AccelerationConfig
        return {"Database_Corrector": Database_Corrector, "AccelerationConfig": AccelerationConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")