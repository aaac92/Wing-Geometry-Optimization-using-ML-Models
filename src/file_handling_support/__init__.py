def __getattr__(name):
    if name in ("get_all_files_from", "compression_Alma"):
        from .file_handling import get_all_files_from, compression_Alma
        return {"get_all_files_from": get_all_files_from, "compression_Alma": compression_Alma}[name]
    if name == "Training_Data_Generator":
        from .trining_csv_creator import Training_Data_Generator
        return Training_Data_Generator
    if name in ("Database_Corrector", "AccelerationConfig"):
        from .database_correction import Database_Corrector, AccelerationConfig
        return {"Database_Corrector": Database_Corrector, "AccelerationConfig": AccelerationConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")