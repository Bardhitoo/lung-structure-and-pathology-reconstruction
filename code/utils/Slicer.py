class Slicer:
    def __init__(self, plotter, mesh):
        self.mesh = mesh
        self.plotter = plotter

    def slice_x(self, location):
        slc = self.mesh.slice(normal=(0, 0, 1), origin=(self.mesh.center[0], self.mesh.center[1], location))

        self.plotter.subplot(6)
        self.plotter.add_mesh(slc, name='slice_x', pickable=False)
        self.plotter.camera_position = 'xy'
        self.plotter.camera.roll = 90
        self.plotter.add_text("X-Axis", font_size=14, position="lower_left")


        # self.plotter.camera.reset_clipping_range()
        # self.plotter.subplot(1, 1)
        # self.plotter.add_mesh(slc, name='slice_x')

    def slice_y(self, location):
        slc = self.mesh.slice(normal=(0, 1, 0), origin=(self.mesh.center[0], location, self.mesh.center[2]))

        self.plotter.subplot(5)
        self.plotter.add_mesh(slc, name='slice_y', pickable=False)
        self.plotter.camera_position = 'xz'
        self.plotter.camera.roll = 90
        self.plotter.add_text("Y-Axis", font_size=14, position="lower_left")

        # self.plotter.subplot(1, 1)
        # self.plotter.add_mesh(slc, name='slice_y')

    def slice_z(self, location):
        slc = self.mesh.slice(normal=(1, 0, 0), origin=(location, *self.mesh.center[1:]))

        self.plotter.subplot(4)
        self.plotter.add_mesh(slc, name='slice_z', pickable=False)
        self.plotter.camera_position = 'yz'
        self.plotter.camera.roll = 0
        self.plotter.add_text("Z-Axis", font_size=14, position="lower_left")
        # self.plotter.camera.reset_clipping_range()

        # self.plotter.subplot(1, 1)
        # self.plotter.add_mesh(slc, name='slice_z')
