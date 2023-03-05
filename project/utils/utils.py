import os
import numpy as np

import pyvista as pv
from skimage import measure


def save_mesh(mesh, filename, output_folder='./meshes'):
    mesh.save(os.path.join(output_folder, filename))


def generate_mesh(image, threshold=-300, mask=None, face_color=(0.65, 0., 0.25), ax=None, new_figure=False, alpha=0.70):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    verts, faces, normals, values = measure.marching_cubes(p, level=threshold, mask=mask)

    vfaces = np.column_stack((np.ones(len(faces), ) * 3, faces)).astype(int)

    mesh = pv.PolyData(verts, vfaces)
    mesh['Normals'] = normals
    mesh['values'] = values

    return mesh


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def background_labels_around_circle(labels):
    d, w, h = labels.shape

    # parameter that needs tuning
    stroke_thickness = 2
    # get radius
    radius1 = (w / 2) - 1
    radius2 = radius1 - stroke_thickness
    center = (w / 2)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    # create a mask to measure the distance of the points from the center
    dist = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)

    # create mask with two different radius
    mask1 = dist <= radius1
    mask2 = dist <= radius2

    # logical xor between those two masks with different radius
    # to get only a thin stroked circle with the specific pixels
    mask = np.logical_xor(mask1, mask2)

    # identify irrelevant pixels from each slice
    mask_in_3D = np.ones(labels.shape) * mask
    irrelevant_labels = mask_in_3D * labels

    return irrelevant_labels
