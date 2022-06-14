import os
import warnings

### Check if project has been initialized
### Look for '.sccala_proj' directory in current working directory
### In this directory a 'sccala_paths' file needs to exist
### If project settings directory is not found, throw a warning

# Get current working directory
cwd = os.getcwd()

if not os.path.exists(os.path.join(cwd, ".sccala_proj")):
    warnings.warn(
        (
            "No '.sccala_proj' directory found in current working directory!\n"
            + "Please initialize project before running anything else!\n"
        ),
        ImportWarning,
    )
elif not os.path.exists(os.path.join(cwd, ".sccala_proj", ".sccala_paths")):
    warnings.warn(
        (
            "Found '.sccala_proj' directory, but cannot find '.sccala_paths' file.\n"
            + "Make sure project has been initialized correctly.\n"
        ),
        ImportWarning,
    )
else:
    pass
