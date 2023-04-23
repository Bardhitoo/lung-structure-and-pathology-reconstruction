import pyvista as pv
import numpy as np
from utils.Slicer import Slicer

# create an example numpy volume
volume = np.load("ndarrays/scan.npy")
volume.transpose((2, 1, 0))

# convert the volume to a PyVista data object
mesh = pv.wrap(volume)

pl = pv.Plotter(shape=(2, 2))
slicer = Slicer(pl, mesh)

pl.subplot(0, 0)
# pl.enable_parallel_projection()
pl.add_slider_widget(slicer.slice_z, mesh.bounds[0:2], )

pl.subplot(0, 1)
pl.add_slider_widget(slicer.slice_y, mesh.bounds[4:6], )

pl.subplot(1, 0)
pl.add_slider_widget(slicer.slice_x, mesh.bounds[2:4], )

pl.show()