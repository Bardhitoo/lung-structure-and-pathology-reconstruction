import os
import numpy as np
import pyvista as pv

from colorama import Fore
from utils.LungScanner import LungScanner
from utils.Mesh import Mesh

# Some constants
INPUT_FOLDER = "./data"
OUTPUT_FOLDER = "./meshes"
PROCESS_FROM_FILE = False


def main():
    if PROCESS_FROM_FILE:
        patients = [file for file in os.listdir(INPUT_FOLDER) if file.endswith(".mhd")]
        patients.sort()

        scan = LungScanner(os.path.join(INPUT_FOLDER, patients[0]))
        scan.read_scan()
        scan.resample(new_spacing=np.array([1, 1, 1]))

        segmented_lungs = scan.segment_lung_mask(False)
        segmented_lungs_fill = scan.segment_lung_mask(True)

        print(Fore.GREEN + 'PLOT: 3D Plotting ribcage... May take a little bit')
        mesh_skeleton = Mesh.from_scan(scan.get_scan(), threshold=400)
        mesh_skeleton.save("skeleton.vtk")

        print(Fore.GREEN + 'PLOT: 3D plotting lungs... May take a little bit')
        mesh_lungs_fill = Mesh.from_scan(segmented_lungs_fill, threshold=0)
        mesh_lungs_fill.save("lungs.vtk")

        print(Fore.GREEN + 'PLOT: 3D plotting lung airways... May take a little bit')
        mesh_airways = Mesh.from_scan(segmented_lungs_fill - segmented_lungs, threshold=0)
        mesh_airways.save("airways.vtk")
    else:
        print(Fore.GREEN + 'LOAD: 3D meshes loading...')
        mesh_skeleton = Mesh.load(os.path.join(OUTPUT_FOLDER, "skeleton.vtk"))
        mesh_lungs_fill = Mesh.load(os.path.join(OUTPUT_FOLDER, "lungs.vtk"))
        mesh_airways = Mesh.load(os.path.join(OUTPUT_FOLDER, "airways.vtk"))

    # shape="3|1" means 3 plots on the left and 1 on the right,
    # shape="4/2" means 4 plots on top of 2 at bottom.
    plotter = pv.Plotter(shape='3|1', window_size=(1000, 1200))

    plotter.subplot(0)
    plotter.add_text("Skeleton")
    plotter.add_mesh(mesh_skeleton, show_edges=False, color=(.98, 1., .96))

    plotter.subplot(1)
    plotter.add_text("Lungs")
    plotter.add_mesh(mesh_lungs_fill, show_edges=False, color=(1., 0., 0.4))

    plotter.subplot(2)
    plotter.add_text("Airways")
    plotter.add_mesh(mesh_airways, show_edges=False, color=(1., 0.5, 0.5))

    plotter.subplot(3)
    plotter.add_text("")
    plotter.add_mesh(mesh_lungs_fill, show_edges=False, color=(1., 0., 0.4))
    plotter.add_mesh(mesh_skeleton, show_edges=False, color=(.98, 1., .96))
    plotter.add_mesh(mesh_airways, show_edges=False, color=(1., 0.5, 0.5))

    # link all the views
    plotter.link_views()

    # plotter.camera_position = [(15, 5, 0), (0, 0, 0), (0, 1, 0)]
    plotter.show()


if "__main__" == __name__:
    main()
