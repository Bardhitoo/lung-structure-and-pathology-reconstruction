import pyvista as pv
import numpy as np
from utils.Slicer import Slicer

# create an example numpy volume
volume = np.load("ndarrays/scan.npy")
volume.transpose((2, 1, 0))

# convert the volume to a PyVista data object
mesh = pv.wrap(volume)

pl = pv.Plotter(shape=(3, 1))
slicer = Slicer(pl, mesh)

pl.subplot(0, 0)
# pl.enable_parallel_projection()
pl.add_slider_widget(slicer.slice_z, mesh.bounds[0:2], )

pl.subplot(1, 0)
pl.add_slider_widget(slicer.slice_y, mesh.bounds[2:4], )

pl.subplot(2, 0)
pl.add_slider_widget(slicer.slice_x, mesh.bounds[4:6], )
pl.show()