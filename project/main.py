import os
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style
import pyvista as pv
from LungSegmentor import LungSegmenter
from utils import *

# Some constants
INPUT_FOLDER = "C:\\Users\\bardh\\Desktop\\2021\\RIT_CS\\sem_2\\independent_study\\lung_cancer_proj\\data" \
               "\\subset_012345\\"


def main():
    patients = [file for file in os.listdir(INPUT_FOLDER) if file.endswith(".mhd")]
    patients.sort()

    # scan = LungSegmenter(INPUT_FOLDER + patients[0], "./meshes")
    # scan.load_scan()
    # scan.resample(new_spacing=np.array([1, 1, 1]))
    # segmented_lungs = scan.segment_lung_mask(False)
    # segmented_lungs_fill = scan.segment_lung_mask(True)
    #
    # print(Fore.GREEN + 'PLOT: 3D Plotting ribcage... May take a little bit')
    # mesh_skeleton = scan.generate_mesh(pix_resampled, 400, face_color=(.98, 1., .96), alpha=0.2)
    #
    # print(Fore.GREEN + 'PLOT: 3D plotting lungs... May take a little bit')
    # mesh_lungs_fill = generate_mesh(segmented_lungs_fill, 0, face_color=(0.65, 0., 0.25))
    #
    # print(Fore.GREEN + 'PLOT: 3D plotting lung airways... May take a little bit')
    # mesh_airways = generate_mesh(segmented_lungs_fill - segmented_lungs, 0, face_color=(0.45, 0.45, 1.), alpha=0.8)

    # ============================================
    # first_patient = load_scan(INPUT_FOLDER + patients[2])
    # first_patient_pixels = get_pixels_hu(first_patient)
    #
    # # hounsfield_histogram(first_patient_pixels)
    #
    # # print(Fore.GREEN + 'RESCALE: Interpolating the values to isomorphic resolution... May take a little bit' + Style.RESET_ALL)
    # # pix_resampled, spacing = resample(first_patient_pixels, first_patient, np.array([1, 1, 1]))
    #
    # print(Fore.YELLOW + "\tShape before resampling\t", first_patient_pixels.shape)
    # # print(Fore.YELLOW + "\tShape after resampling\t", pix_resampled.shape)
    # print(Style.RESET_ALL)
    #
    # print(Fore.GREEN + 'PROCESS: 3D processing lungs... May take a little bit' + Style.RESET_ALL)
    # segmented_lungs = segment_lung_mask(first_patient_pixels, False)
    # segmented_lungs_fill = segment_lung_mask(first_patient_pixels, True)
    #
    # print(Fore.GREEN + 'MESH: 3D reconstructing ribcage... May take a little bit')
    # mesh_skeleton = generate_mesh(first_patient_pixels, 300)
    # save_mesh(mesh_skeleton, 'skeleton.vtk')
    #
    # print(Fore.GREEN + 'MESH: 3D reconstructing lungs... May take a little bit')
    # mesh_lungs_fill = generate_mesh(segmented_lungs_fill, 0, face_color=(0.65, 0., 0.25))
    # save_mesh(mesh_lungs_fill, 'lungs_fill.vtk')
    #
    # print(Fore.GREEN + 'MESH: 3D reconstructing lung airways... May take a little bit')
    # mesh_airways = generate_mesh(segmented_lungs_fill - segmented_lungs, 0, face_color=(0.45, 0.45, 1.), alpha=0.8)
    # save_mesh(mesh_airways, 'airways.vtk')
    # print(Style.RESET_ALL)


    # both = pv.Plotter()
    # both.add_mesh(mesh_skeleton, color=(.98, 1., .96))
    # plotter.add_mesh(mesh_airways, color=(0.65, 0., 0.25))
    # both.add_mesh(mesh_lungs_fill, color=(1., 0., 0.4))
    # plotter.show()
    mesh_skeleton = pv.read("./meshes/skeleton.vtk")
    mesh_lungs_fill = pv.read("./meshes/lungs_fill.vtk")
    mesh_airways = pv.read("./meshes/airways.vtk")

    # shape="3|1" means 3 plots on the left and 1 on the right,
    # shape="4/2" means 4 plots on top of 2 at bottom.
    plotter = pv.Plotter(shape='3|1', window_size=(1000, 1200))

    plotter.subplot(0)
    plotter.add_text("Skeleton")
    plotter.add_mesh(mesh_skeleton, show_edges=False, color=(.98, 1., .96))

    # load and plot the uniform data example on the right-hand side
    plotter.subplot(1)
    plotter.add_text("Lungs")
    plotter.add_mesh(mesh_lungs_fill, show_edges=False, color=(1., 0., 0.4))

    plotter.subplot(2)
    plotter.add_text("Airways")
    plotter.add_mesh(mesh_airways, show_edges=False, color=(1., 0.5, 0.5))

    plotter.subplot(3)
    plotter.add_text("")
    plotter.add_mesh(mesh_lungs_fill.merge(mesh_skeleton), show_edges=False, color=(1., 1., .95))
    plotter.link_views()  # link all the views
    # plotter.camera_position = [(15, 5, 0), (0, 0, 0), (0, 1, 0)]
    plotter.show()


if "__main__" == __name__:
    main()
