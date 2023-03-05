import os
import numpy as np
import pyvista as pv
from skimage import measure

if not os.path.exists("../meshes"):
    os.mkdir("../meshes")


class Mesh:
    def __init__(self, vertices, faces, normals, values):
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.values = values

        vfaces = np.column_stack((np.ones(len(faces), ) * 3, faces)).astype(int)
        self.polydata = pv.PolyData(vertices, vfaces)
        self.polydata['Normals'] = normals
        self.polydata['values'] = values

    def save(self, filename, output_folder='./meshes'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self.polydata.save(os.path.join(output_folder, filename))

    @staticmethod
    def load(mesh_path):
        return pv.read(mesh_path)

    @classmethod
    def from_scan(cls, image, threshold=-300, mask=None):
        # Position the scan upright,
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2, 1, 0)
        verts, faces, normals, values = measure.marching_cubes(p, level=threshold, mask=mask)

        return cls(verts, faces, normals, values)
