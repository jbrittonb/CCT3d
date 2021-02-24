import pyvista as pv
import numpy as np

"compute volume V, surface area A, the ratio A/V, and the ratio A^3/V^2 for various sized prisms"


# simple function to check for symmetric matrix
# 	returns True for symmetric matrix
def check_symmetric(a, rtol=1e-05, atol=1e-08):
	return np.allclose(a, a.T, rtol=rtol, atol=atol)    


# array of widths, depths, and heights
maxSize = 7
W = np.linspace(1,maxSize,maxSize) 	# width
D = W 								# depth
H = W 								# height

DD, HH, WW = np.meshgrid(D, H, W) 

# volume
V = DD*HH*WW 

# surface area
A = 2*(WW*DD) + 2*(WW*HH) + 2*(DD*HH)

# surface area to volume
AtoV = A/V

# nondimensional surface area to volume A^3/V^2
ndAtoV = A**3/(V**2)

# create a PyVista mesh to store this data
# Create the spatial reference
data = pv.UniformGrid()

# Set the grid dimensions: shape + 1 because we want to inject our values on
#   the CELL data
data.dimensions = np.array(V.shape) + 1
data.cell_arrays["V"] = V.flatten(order="F")  
data.cell_arrays["A"] = A.flatten(order="F")  
data.cell_arrays["A/V"] = AtoV.flatten(order="F")  
data.cell_arrays["A3/V2"] = ndAtoV.flatten(order="F")  


# plot
data.set_active_scalars("V")
data.plot(show_edges=True, show_grid=True) 

data.set_active_scalars("A")
data.plot(show_edges=True, show_grid=True) 

data.set_active_scalars("A/V")
data.plot(show_edges=True, show_grid=True) 

data.set_active_scalars("A3/V2")
data.plot(show_edges=True, show_grid=True) 


# contours
data.set_active_scalars("V")
ptdata = data.cell_data_to_point_data()
VdataContours = data.cell_data_to_point_data().contour(5, scalars=data.active_scalars_info.name)
VdataContours.plot(show_grid=True)

data.set_active_scalars("A")
ptdata = data.cell_data_to_point_data()
AdataContours = data.cell_data_to_point_data().contour(5, scalars=data.active_scalars_info.name)
AdataContours.plot(show_grid=True)

data.set_active_scalars("A/V")
ptdata = data.cell_data_to_point_data()
AVdataContours = data.cell_data_to_point_data().contour(5, scalars=data.active_scalars_info.name)
AVdataContours.plot(show_grid=True)

data.set_active_scalars("A3/V2")
ptdata = data.cell_data_to_point_data()
ndAVdataContours = data.cell_data_to_point_data().contour(5, scalars=data.active_scalars_info.name)
ndAVdataContours.plot(show_grid=True)

# write out cell-based data
# pv.save_meshio("AV.xdmf", data) 

# write out point-based data
# pv.save_meshio("AVpoint.xdmf", ptdata) 


