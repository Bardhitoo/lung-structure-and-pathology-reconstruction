import os

from utils.utils import extract_meshes_from_scan, load_meshes, visualize_meshes
from utils.config import config


# TODO: Create a threshold slider widget
#   Rescale and position the nodules in the appropriate position
def main():
    if config["PROCESS_FROM_FILE"]:
        patients = [file for file in os.listdir(config["INPUT_FOLDER"]) if file.endswith(".mhd")]
        patients.sort()
        patient_of_interest = patients[1]

        mesh_airways, mesh_lungs_fill, mesh_nodules, mesh_skeleton = extract_meshes_from_scan(patient_of_interest)
    else:
        mesh_airways, mesh_lungs_fill, mesh_nodules, mesh_skeleton = load_meshes()

    processed_meshes = [mesh_skeleton, mesh_lungs_fill, mesh_airways, mesh_nodules]
    visualize_meshes(processed_meshes)


if "__main__" == __name__:
    main()
