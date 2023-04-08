import os
import numpy as np
import pandas as pd
import pyvista as pv
import scipy
from colorama import Fore

import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.LungScanner import LungScanner
from utils.Mesh import Mesh
from utils.config import config


def nodule_3d_viz(nodule_mask, title, edgecolor='0.2', cmap='cool', step=1, figsize=(5, 5), backend='matplotlib'):
    mask = nodule_mask  # boolean_mask(pad=[(1,1), (1,1), (1,1)])
    rij = 0.625  # self.scan.pixel_spacing
    rk = 1  # self.scan.slice_thickness

    if backend == 'matplotlib':
        verts, faces, _, _ = marching_cubes(mask.astype(np.float), 0.5,
                                            spacing=(rij, rij, rk),
                                            step_size=step)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        t = np.linspace(0, 1, faces.shape[0])
        mesh = Poly3DCollection(verts[faces],
                                edgecolor=edgecolor,
                                facecolors=plt.cm.cmap_d[cmap](t))
        ax.add_collection3d(mesh)

        ceil = max(mask.shape) * rij
        ceil = int(np.round(ceil))

        ax.set_xlim(0, ceil)
        ax.set_xlabel('length (mm)')

        ax.set_ylim(0, ceil)
        ax.set_ylabel('length (mm)')

        ax.set_zlim(0, ceil)
        ax.set_zlabel('length (mm)')

        plt.title(title)
        plt.tight_layout()
        plt.show()
    elif backend == 'mayavi':
        try:
            from mayavi import mlab
            sf = mlab.pipeline.scalar_field(mask.astype(np.float))
            sf.spacing = [rij, rij, rk]
            mlab.pipeline.iso_surface(sf, contours=[0.5])
            mlab.show()
        except ImportError:
            print("Mayavi could not be imported. Is it installed?")


def remove_irrelevant_preds(nodule: pd.Series, nodule_mask: np.array):
    D, H, W = nodule_mask.shape
    z, y, x, d = nodule.coordZ, nodule.coordY, nodule.coordX, nodule.diameter_mm

    z_start = max(0, int(np.floor(z - d / 2.)))
    y_start = max(0, int(np.floor(y - d / 2.)))
    x_start = max(0, int(np.floor(x - d / 2.)))
    z_end = min(D, int(np.ceil(z + d / 2.)))
    y_end = min(H, int(np.ceil(y + d / 2.)))
    x_end = min(W, int(np.ceil(x + d / 2.)))

    nodule_idxs = np.unique(nodule_mask[z_start:z_end, y_start:y_end, x_start:x_end])
    for nodule_idx in nodule_idxs:
        nodule_mask[nodule_mask == nodule_idx] = 0  # Make the nodule same color as background

    return nodule_mask


def generate_contours(uid_of_interest):
    load_dir = r"C:\Users\bardh\Desktop\2021\RIT_CS\sem_2\independent_study\lung_cancer_proj\NoduleNet\results\cross_val_test\res\200"
    rpns = pd.read_csv(load_dir + "\\FROC\\submission_rpn.csv")
    THRESHOLD = 0.95

    pred = np.load(os.path.join(load_dir, uid_of_interest + ".npy"))
    pred_updated = pred.copy()  # To store and compare changes

    # get the row of the highest prediction
    uid_nodules = rpns[(rpns.seriesuid == uid_of_interest)]

    # Anything smaller than threshold, not taken into account
    uid_irrelevant_nodules = uid_nodules[uid_nodules.probability < THRESHOLD]

    # Post-processing of detected nodules
    for idx, nodule in uid_irrelevant_nodules.iterrows():
        remove_irrelevant_preds(nodule, pred_updated)

    return pred_updated


def extract_meshes_from_scan(patient_of_interest):
    scan = LungScanner(os.path.join(config["INPUT_FOLDER"], patient_of_interest))
    scan.read_scan()
    scan.resample(new_spacing=np.array([1, 1, 1]))

    segmented_lungs = scan.segment_lung_mask(False)
    segmented_lungs_fill = scan.segment_lung_mask(True)

    print(Fore.GREEN + 'PLOT: 3D plotting ribcage... May take a little bit')
    mesh_skeleton = Mesh.from_scan(scan.get_scan(), name="Skeleton", color=(.98, 1., .96), threshold=400)
    mesh_skeleton.save("skeleton.obj")

    print(Fore.GREEN + 'PLOT: 3D plotting lungs... May take a little bit')
    mesh_lungs_fill = Mesh.from_scan(segmented_lungs_fill, name="Lungs", color=(1., 0., 0.4), threshold=0)
    mesh_lungs_fill.save("lungs.obj")

    print(Fore.GREEN + 'PLOT: 3D plotting lung airways... May take a little bit')
    mesh_airways = Mesh.from_scan(segmented_lungs_fill - segmented_lungs, name="Airways", color=(1., 0.5, 0.5),
                                  threshold=0)
    mesh_airways.save("airways.obj")

    print(Fore.GREEN + 'PLOT: 3D plotting lung nodules... May take a little bit' + Fore.RESET)
    nodules_mask = generate_contours(patient_of_interest[:-4])
    resampled_nodule_mask = scipy.ndimage.interpolation.zoom(nodules_mask, scan.real_resize_factor, mode="nearest")  # TODO: Find a better way to do this
    # resampled_nodule_mask = resampled_nodule_mask.transpose(2, 1, 0)
    mesh_nodules = Mesh.from_scan(resampled_nodule_mask, name="Nodules", color=(0.18, 0.09, 0.2), threshold=0)
    mesh_nodules.save("nodules.obj")

    return mesh_airways, mesh_lungs_fill, mesh_nodules, mesh_skeleton


def load_meshes():
    print(Fore.GREEN + 'LOAD: 3D meshes loading...' + Fore.RESET)
    mesh_nodules = Mesh.load(os.path.join(config["OUTPUT_FOLDER"], "nodules.obj"))
    mesh_skeleton = Mesh.load(os.path.join(config["OUTPUT_FOLDER"], "skeleton.obj"))
    mesh_lungs_fill = Mesh.load(os.path.join(config["OUTPUT_FOLDER"], "lungs.obj"))
    mesh_airways = Mesh.load(os.path.join(config["OUTPUT_FOLDER"], "airways.obj"))

    return mesh_airways, mesh_lungs_fill, mesh_nodules, mesh_skeleton


def visualize_meshes(processed_meshes):
    """
    Visualizes meshes with an interactive interface
    """
    def set_opacity(actor, value):
        actor.prop.opacity = value

    # shape="3|1" means 3 plots on the left and 1 on the right,
    # shape="4/2" means 4 plots on top of 2 at bottom.
    plotter = pv.Plotter(shape='4|1', window_size=(1000, 1200))

    for idx, mesh in enumerate(processed_meshes):
        plotter.subplot(idx)
        plotter.add_text(mesh.name)
        plotter.add_mesh(mesh.polydata, show_edges=False, color=mesh.color)

    plotter.subplot(4)
    for idx, mesh in enumerate(processed_meshes):
        if mesh.name == "Lungs":
            lung_actor = plotter.add_mesh(mesh.polydata, show_edges=False, color=mesh.color, opacity=0.0)
            plotter.add_slider_widget(callback=lambda value, actor=lung_actor: set_opacity(actor, value), value=0.0,
                                      rng=[0, 1], title=f'{mesh.name} Opacity', pointa=(0.025, 0.1), pointb=(0.31, 0.1))
            continue
        elif mesh.name == "Skeleton":
            lung_actor = plotter.add_mesh(mesh.polydata, show_edges=False, color=mesh.color, opacity=1.0)
            plotter.add_slider_widget(callback=lambda value, actor=lung_actor: set_opacity(actor, value), value=0.0,
                                      rng=[0, 1], title=f'{mesh.name} Opacity', pointa=(0.35, 0.1),
                                      pointb=(0.64, 0.1), )
            continue
        elif mesh.name == "Airways":
            lung_actor = plotter.add_mesh(mesh.polydata, show_edges=False, color=mesh.color, opacity=1.0)
            plotter.add_slider_widget(callback=lambda value, actor=lung_actor: set_opacity(actor, value), value=0.0,
                                      rng=[0, 1], title=f'{mesh.name} Opacity', pointa=(0.67, 0.1),
                                      pointb=(0.98, 0.1), )
            continue
        plotter.add_mesh(mesh.polydata, show_edges=False, color=mesh.color)

    # link all the views
    plotter.link_views()

    # plotter.camera_position = [(15, 5, 0), (0, 0, 0), (0, 1, 0)]
    plotter.show()


