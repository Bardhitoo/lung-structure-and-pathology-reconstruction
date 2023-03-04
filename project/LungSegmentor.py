import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import SimpleITK as sitk
from colorama import Fore, Style
from skimage import measure
import scipy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class LungSegmenter:
    def __init__(self, scan_path, output_folder="./meshes"):
        self.scan_path = scan_path
        self.output_folder = output_folder
        self.image = None
        self.pixel_spacing = None
        self.resampled_image = None
        self.meshes = dict()

    def load_scan(self):
        self.image = sitk.ReadImage(self.scan_path)

        # Determine current pixel spacing
        # Note: Spacing: X-axis, Y-axis, Z-axis
        self.pixel_spacing = np.array(self.image.GetSpacing())[np.array([2, 1, 0])]
        self._get_pixels_hu()

    def _get_pixels_hu(self):
        # Get the pixel array from the SimpleITK image object
        image = sitk.GetArrayFromImage(self.image)

        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0, same as air
        image[image < -2000] = 0
        self.image = np.array(image, dtype=np.int16)

    def resample(self, new_spacing=np.array([1, 1, 1])):
        print(Fore.GREEN + 'RESCALE: Interpolating the values to isomorphic resolution... May take a little bit' + Style.RESET_ALL)
        spacing = self.pixel_spacing

        resize_factor = spacing / new_spacing
        new_real_shape = self.image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / self.image.shape
        # new_spacing = spacing / real_resize_factor

        self.resampled_image = scipy.ndimage.interpolation.zoom(self.image, real_resize_factor, mode="nearest")

        print(Fore.YELLOW + "\tShape before resampling\t", self.pixel_spacing.shape)
        print(Fore.YELLOW + "\tShape after resampling\t", new_shape)
        print(Style.RESET_ALL)

    def generate_mesh(self, image=None, threshold=-300, face_color=(0.65, 0., 0.25), ax=None, new_figure=False, alpha=0.70):
        # Position the scan upright,
        # so the head of the patient would be at the top facing the camera
        if not image:
            image = self.resampled_image

        p = image.transpose(2, 1, 0)
        verts, faces, normals, values = measure.marching_cubes(p, threshold)

        vfaces = np.column_stack((np.ones(len(faces), ) * 3, faces)).astype(int)

        mesh = pv.PolyData(verts, vfaces)
        mesh['Normals'] = normals
        mesh['values'] = values

        return mesh


    def save_mesh(self, key, filename):
        self.meshes[key].save(os.path.join(self.output_folder, filename))

    def hounsfield_histogram(self):
        plt.hist(self.image.flatten(), bins=80, color="c")
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
        plt.imshow(self.image[80], cmap=plt.cm.gray)
        plt.show()

    def segment_lung_mask(self, fill_lung_structures=True):
        print(Fore.GREEN + 'PROCESS: 3D processing lungs... May take a little bit' + Style.RESET_ALL)

        # not actually binary, but 1 and 2.
        # 0 is treated as background, which we do not want
        binary_image = np.array(self.image > -320, dtype=np.int8) + 1
        # connected components of binary image
        labels = measure.label(binary_image)

        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air
        #   around the person in half
        irrelevant_labels = self.background_labels_around_circle(labels)

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
                l_max = self.largest_label_volume(labeling, bg=0)

                if l_max is not None:  # This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1  # Make the image actual binary
        binary_image = 1 - binary_image  # Invert it, lungs are now 1

        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = self.largest_label_volume(labels, bg=0)
        if l_max is not None:  # There are air pockets
            binary_image[labels != l_max] = 0

        return binary_image

    @staticmethod
    def largest_label_volume(im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)
        counts = counts[vals != bg]
        vals = vals[vals != bg]
        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    @staticmethod
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
