import os


def main():
    print(
        "******************************************************************\n"
        "This script will setup the necessary paths for the Sccala Toolkit.\n"
        "You can always manually change the paths by editing\n"
        "the '.sccala_paths' file in the Sccala directory.\n"
        "\n"
        "The following paths need to be set:\n"
        "Data: Folder where all subfolders containing the SN spectra and photometry can be found.\n"
        "Diagnostic: Folder where all diagnostic plots will be stored.\n"
        "Results: Folder where all results will be stored.\n"
        "\n"
        "For more information and an example, see the README.md file.\n"
        "\n"
        "IMPORTANT: All paths should be specified as absolute paths!\n"
        "*****************************************************************"
    )

    # Create project settings directory
    if not os.path.exists(".sccala_proj"):
        os.mkdir(".sccala_proj")

    if os.path.exists(os.path.join(".sccala_proj", ".sccala_paths")):
        if not input("Found existing '.sccala_paths' file. Continue? y/[n]") == "y":
            return

    data_path = os.path.abspath(str(input("Data path [Data]: ") or "Data"))
    diag_path = os.path.abspath(
        str(
            input("Diagnostic path [.sccala_proj/diagnostic]: ")
            or ".sccala_proj/diagnostic"
        )
    )
    res_path = os.path.abspath(
        str(input("Results path [.sccala_proj/results]: ") or ".sccala_proj/results")
    )

    # Write paths file
    with open(".sccala_proj/.sccala_paths", "w") as f:
        f.write(
            "data_path %s\n"
            "diag_path %s\n"
            "res_path %s" % (data_path, diag_path, res_path)
        )

    # Create directories if they do not already exist
    if not os.path.exists(data_path):
        print("Data directory does not yet exist, creating it...")
        os.makedirs(data_path)
    if not os.path.exists(diag_path):
        print("Diagnostic directory does not yet exist, creating it...")
        os.makedirs(diag_path)
    if not os.path.exists(res_path):
        print("Results directory does not yet exist, creating it...")
        os.makedirs(res_path)

    print(
        "******************************\n"
        "Finished initializing project!\n"
        "******************************"
    )
