import os
import numpy as np
import pyvista as pv
from skimage import measure
import pickle

OUTPUT_FOLDER = "./meshes"

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)


class Mesh:
    def __init__(self, vertices, faces, normals, values, name="scan", color=(1,1,1)):
        self.name = name
        self.color = color

        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.values = values

        vfaces = np.column_stack((np.ones(len(faces), ) * 3, faces)).astype(int)
        self.polydata = pv.PolyData(vertices, vfaces)
        self.polydata['Normals'] = normals
        self.polydata['values'] = values

    def save(self, filename, output_folder=OUTPUT_FOLDER):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(f"{os.path.join(output_folder, filename)}", "wb") as handler:
            pickle.dump(self, handler)

    @staticmethod
    def load(mesh_path):
        with open(mesh_path, "rb") as file:
            mesh = pickle.load(file)

        return mesh

    @classmethod
    def from_scan(cls, scan, name="scan", color=(1,1,1), threshold=-300, mask=None):
        # Position the scan upright,
        # so the head of the patient would be at the top facing the camera
        p = scan.transpose(2, 1, 0)
        verts, faces, normals, values = measure.marching_cubes(p, level=threshold, mask=mask)

        return cls(verts, faces, normals, values, name, color)
