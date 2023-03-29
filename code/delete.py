import numpy as np
import pyvista as pv

# Define a function to update the plot based on the slider value
def update_slice(idx, grid, plotter):
    # Clear the plot
    plotter.clear()

    # Add the sliced grid to the plot
    slice_grid = grid.slice_along_axis(axis="x", n=1, tolerance=0, center=(idx, 0, 0))
    plotter.add_mesh(slice_grid)

    # Render the plot
    plotter.show()

# Generate a random 3D numpy array
data = np.load("numpy_arrays/segmented_lungs_fill.npy")
# Create a PyVista grid from the numpy array
grid = pv.wrap(data)

# Create a PyVista plotter
plotter = pv.Plotter()

# Add the grid to the plotter
plotter.add_mesh_slice(grid)

# Add a slider to the plot
# plotter.add_slider_widget(
#     rng=(0, data.shape[0]-1),
#     value=0,
#     title="X Slice Index",
#     callback=lambda idx: update_slice(idx, grid, plotter),
# )

# Show the plot
plotter.show()