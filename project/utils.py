import os
import numpy as np

import pyvista as pv
import scipy.ndimage
import SimpleITK as sitk
from skimage import measure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if not os.path.exists("./meshes"):
    os.mkdir("./meshes")


# Load the scans in given folder path
def load_scan(path):
    # Read the .mhd file
    image = sitk.ReadImage(path)

    return image


def get_pixels_hu(image_obj):
    # Get the pixel array from the SimpleITK image object
    image = sitk.GetArrayFromImage(image_obj)

    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0, same as air
    image[image < -2000] = 0

    return np.array(image, dtype=np.int16)


def resample(image: np.ndarray, scan: sitk.Image, new_spacing: np.ndarray = np.array([1, 1, 1])):
    # Determine current pixel spacing
    # Note: Spacing: X-axis, Y-axis, Z-axis
    spacing = np.array(scan.GetSpacing())

    # Note: Rearranging: Z-axis, Y-axis, X-axis to accommodate for image.shape (z_axis, y_axis, x_axis)
    spacing = spacing[np.array([2, 1, 0])]

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


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
    # mesh.plot(scalars='values')
    # a = 2
    # if new_figure:
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #
    # # Fancy indexing: `verts[faces]` to generate a collection of triangles
    # mesh = Poly3DCollection(verts[faces], alpha=alpha)
    # mesh.set_facecolor(face_color)
    # ax.add_collection3d(mesh)
    #
    # ax.set_xlim(0, p.shape[0])
    # ax.set_ylim(0, p.shape[1])
    # ax.set_zlim(0, p.shape[2])
    #
    # # plt.show()
    # return mesh, ax


def hounsfield_histogram(first_patient_pixels):
    plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()
    # Show some slice in the middle
    plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
    plt.show()


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


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    # connected components of binary image
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    irrelevant_labels = background_labels_around_circle(labels)

    for idx, label in enumerate(labels):
        for irrelevant_label in irrelevant_labels[idx]:
            # Make the selected labels background
            binary_image[idx][label == irrelevant_label] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image
