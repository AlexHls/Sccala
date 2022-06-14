import os

import sccala.asynphot.config as cfg

if not os.path.exists(cfg.get_vega_path()):
    print(
        "-------------------------------------------\n"
        "WARNING - NO VEGA REFERENCE SPECTRUM FOUND!\n"
        "-------------------------------------------\n"
        "You need to configure a reference spectrum:\n"
    )
    file = str(input("Path to Vega reference spectrum: "))
    cfg.set_vega_spec(file)
