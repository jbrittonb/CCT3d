#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initiated on Thu Aug 27 15:47 2020

@author: jbrittonb

Script to generate a particular "simulation file" that specifies
one or more simulations to be run by cct3D (which is built on Fenics)

The "simulation file" is a CSV (or spreadsheet) table that is used by the
cct3D function simulate_cuboid(filename). Each row in the CSV file specifies one
simulation; the function simulate_cuboid(filename) iterates over the rows to conduct
a "batch" of simulations.

Scope and basic situation of these particular simulations:
	This test is limited to rectangular cuboids that span a range of
	surface areas A and volumes V (and range of ratio A/V).

	A typical mass concrete situation is taken to be a rectangular cuboid
	sitting on the ground; the bottom is in contact with earth and the remaining
	5 sides are surrounded by air or water.  For now solar radiation is ignored,
	so that only a convective boundary condition appears on these 5 sides.

	Accordingly in these simulations, these cuboids are taken to be insulated
	on the bottom with convection on the remaning sides.  For computational
	efficiency, we take advantage of the symmetry of this situation and solve
	only one 'quadrant' of the cuboid:
		- take the dimensions to be
			
			width W in the x-direction
			depth D in the y-direction
			height H in the z-direction
		
		- then only need to simulate the 1/4 cuboid of size
			(1/2 W)  X  (1/2 D)  X  (H)

		- note that the user defines W, D, and H "full size",
		  and then specifies which of W, D, and H are to be halved 
		  due to symmetry; the program then only simulates the
		  parts that are required to be simulated;
		  the output data from the simulation are for this "halved" geometry

	The cuboid's face numbering and correspoinding boundary conditions are:

		- YZ plane at x=0 		normal points in -x direction 	face 0, convective BC
		- YZ plane at x=W/2 	normal points in +x direction   face 1, symmetry (adiabatic) BC
		
		- XZ plane at y=0 		normal points in -y direction	face 2, convective BC
		- XZ plane at y=D/2 	normal points in +y direction	face 3, symmetry (adiabatic) BC
		
		- XY plane at z=0 		normal points in -z direction	face 4, adiabatic (symmetry) BC
		- XY place at z=H 		normal points in +z direction 	face 5, convective BC


Assumptions:
	1. variations in boundary conditions have negligible effect on
	   any relation between T_contour and V_inContour
	2. variations in initial temperatures likewise have negligible
	   effect on any relation between T_contour and V_inContour
	3. different concrete mixes also have negligible effect on
	   any relation between T_contour and V_inContour

By these assumptions, the scenarios to simulate are only different sizes
of rectangular cuboids


Use Fenics to solve the 3D heat diffusion equation with Schindler/Riding 
"egen" term using finite elements on unstructured mesh


Boundary conditions are limited to:
    - "convective" (scare quotes to allow for incorporating linearized radiation 
      via sol-air temperature)
    - symmetry (adiabatic) conditions: normal temperature gradients are zero
"""

import numpy as np
import math
import json
import cct3D as cct
import pandas as pd


# ---------------------------------
# "BASE" FILENAME
baseFileName = 'nomogramSimInputs03'


# ---------------------------------
# CONCRETE TO USE
# this is the name of the concrete in the JSON file
# "concreteLibrary.json"
concrete = "Revised_AA+_Baseline"


# ---------------------------------
# SCENARIOS: INITIAL AND BOUNDARY CONDTIONS
# Ufmwk = 0.181       # U-value (= convection coefficient) of formwork in W/(m2 K)

# Identify scenarios using the pair (ambient temperature, initial temperature)
#   in degrees Fahrenheit; use Celsius/Kelvin for computations however
# conversion from Fahrenheit to Celsius
def FtoC(T_F):
	return (T_F - 32)*(5/9)

# Scenarios:        Tamb    Ti			   Tamb      Ti
scenarios = {	str(40)+str(40)+'F': (FtoC(40), FtoC(40)), 
				str(40)+str(50)+'F': (FtoC(40), FtoC(50)),
				 
				str(50)+str(50)+'F': (FtoC(50), FtoC(50)),
				str(50)+str(60)+'F': (FtoC(50), FtoC(60)),
				str(50)+str(70)+'F': (FtoC(50), FtoC(70)),
				 
				str(60)+str(50)+'F': (FtoC(60), FtoC(50)),
				str(60)+str(60)+'F': (FtoC(60), FtoC(60)),
				str(60)+str(70)+'F': (FtoC(60), FtoC(70)),
				 
				str(70)+str(50)+'F': (FtoC(70), FtoC(50)),
				str(70)+str(60)+'F': (FtoC(70), FtoC(60)),
				str(70)+str(70)+'F': (FtoC(70), FtoC(70)),
				str(70)+str(80)+'F': (FtoC(70), FtoC(80)),
				
				str(80)+str(50)+'F': (FtoC(80), FtoC(50)),
				str(80)+str(60)+'F': (FtoC(80), FtoC(60)),
				str(80)+str(70)+'F': (FtoC(80), FtoC(70)),
				str(80)+str(80)+'F': (FtoC(80), FtoC(80)),
				str(80)+str(90)+'F': (FtoC(80), FtoC(90))
	  		}

hconv = 23          # convection coefficient in W/(m2 K)


# ---------------------------------
# PARAMETERS OF THE CUBOIDS TO SIMULATE
# maximum and minimum widths W, depths D, and heights H
#  are maxSize and minSize, respectively 
#  	- smallest volume is a cube with sides minSize in length
#  	- largest volume is a cube with sides maxSize in length
# 	- sizeStep is the increment between minSize and maxSize
#  	- in between are rectangular cuboids (but not cubes) 
# 	  with sides minSize <= side <= maxSize
# the cuboids will be "instantiated" later as arrays of widths, depths, volumes, etc.
DEFINECUBOIDS = False
if DEFINECUBOIDS:
    minSizeFt = 5
    maxSizeFt = 6
    sizeStepFt = 1
    minSize = minSizeFt*0.3048			# meters; 1 m ~= 3 ft.
    maxSize = maxSizeFt*0.3048			# meters; 2.5 m ~= 6.5 ft.
    sizeStep = sizeStepFt*0.3048		# meters
    W, D, H, A, V, AtoV, ndAtoV = cct.define_cuboids(minSize, maxSize, sizeStep)
else:
    Wft = np.array([3, 4,   9, 5,    4, 4]) 
    Dft = np.array([9, 3.4, 9, 2.75, 2, 9]) 
    Hft = np.array([9, 6,   3, 7,    6, 15]) 
    W = Wft*0.3048			# meters; 1 m ~= 3 ft.
    D = Dft*0.3048			# meters; 2.5 m ~= 6.5 ft.
    H = Hft*0.3048		# meters
    W, D, H, A, V, AtoV, ndAtoV = cct.establish_cuboids(W, D, H)    

# plot these cuboids for visual confirmation
# for i in range(V.size):
# 	cuboid_definition = [ (0,0,0), (0,D[i],0), (W[i],0,0), (0,0,H[i]) ]
# 	cct.plot_cuboid(cuboid_definition, maxSize)



# ---------------------------------
# COMPUTATIONAL PARAMETERS
# where the coordinate origin is placed:
#   'B': origin at the "bottom" of the domain, such that
#             0 <= z <= H
#        -0.5*D <= y <= 0.5*D
#        -0.5*W <= x <= 0.5*W
#   'C': origin at the "center" of the domain, such that
#        -0.5*H <= z <= 0.5*H
#        -0.5*D <= y <= 0.5*D
#        -0.5*W <= x <= 0.5*W
#   'S': origin suitable for a "symmetry" quadrant of the domain, such that
#        0 <= z <= H
#        0 <= y <= D (if using symmetry, 0.5*D)
#        0 <= x <= W (if using symmetry, 0.5*W)
coordOrigin = 'S'

# given the assumptions in the header comments and the subsequent
# 	use of symmetry in the problem, need to apportion the
# 	boundary conditions to the various faces
# below, it's {face number: {'tag for bounary condition type': U-value at that face}}
boundary_conditions = {0: {'Convective' : hconv}, 
                       1: {'Symmetry'	: 0}, 
                       2: {'Convective' : hconv}, 
                       3: {'Symmetry'	: 0}, 
                       4: {'Symmetry'  	: 0}, 
                       5: {'Convective'	: hconv}} 
# these boundary conditions are written out to the following JSON file
faceSpecFilename = 'faceSpec01.json'

# tag which of W, D, and H need to be halved for symmetry reasons
halfW = True 
halfD = True
halfH = False

# finite element parameters
nx = 8                  # resolution of mesh cells/meter, x-direction
ny = 8                  # resolution of mesh cells/meter, y-direction
nz = 8                  # resolution of mesh cells/meter, z-direction

FElement = 'P'
FEdegree = 2            # degree of the finite element

# simulation duration and timestep
tfinal_h = 96          	# simulation duration, hours
dt_h = 0.25             # time step size, hours

# snapshots?
#  	if snapshot= True, periodic snapshots will be taken: data at a timestep 
# 	will be written out in VTK format
snapshots = False 		
snapshot_every_x_timestep = 4 		# take a snapshot every this many time steps










# ===============================================
# WRITE OUT THE SIMULATION FILE
# create filenames to be used to store the data of each individual simulation
filenames = []
# need to make new arrays for W, D, H, A, V, AtoV, ndAtoV, Ti_C, and Tamb_C
WW = []
DD = []
HH = []
AA = []
VV = []
A2V = []
ndA2V = []
TiC = []
TambC = []
# loop over cuboids
for i in range(V.size):
	# loop over scenarios
	for j in scenarios:
		filenames.append('cct3d_ab_' + str( np.int(np.round(W[i]/0.3048)) ) + '-' + str( np.int(np.round(D[i]/0.3048)) ) + '-' + str( np.int(np.round(H[i]/0.3048)) ) + 'ft_normAir_' + j +'_' + baseFileName)
		WW.append(W[i])
		DD.append(D[i])
		HH.append(H[i])
		AA.append(A[i])
		VV.append(V[i])
		A2V.append(AtoV[i])
		ndA2V.append(ndAtoV[i])
		TambC.append(scenarios[j][0])
		TiC.append(scenarios[j][1])

# the way the boundary conditions are apportioned to the spaces are written out to a JSON file
#   note that the keys are written as strings; will need to convert to int in subsequent use
with open(faceSpecFilename, 'w') as json_file:
  json.dump(boundary_conditions, json_file)


# set up the table using pandas
sim_parameters = {'ID'	         	: filenames,
				  'concrete'		: concrete,
				  'Width_m' 		: WW,
				  'Depth_m' 		: DD,
				  'Height_m'		: HH,
				  'SurfArea_m2'		: AA,
				  'Volume_m3'		: VV,
				  'AtoV_1/m'		: A2V,
				  'ndAtoV'			: ndA2V,
				  'Tamb_C'			: TambC,
				  'Ti_C'			: TiC,
				  'hconv_W/(m2 K)'	: hconv,
				  'finalTime_h' 	: tfinal_h,
				  'timestep_h'		: dt_h,
				  'snapshots'		: snapshots,
				  'snapshot_every_x_timestep': snapshot_every_x_timestep,
				  'nx_cells/m'		: nx,
				  'ny_cells/m'		: ny,
				  'nz_cells/m'		: nz,
				  'FEtype'			: FElement,
				  'FEdegree'		: FEdegree,
				  'faceSpec'		: faceSpecFilename,
				  'coordOrigin'		: coordOrigin,
				  'halfWidth'		: halfW,
				  'halfDepth'		: halfD,
				  'halfHeight'		: halfH
}

# create dataframe
spdf = pd.DataFrame(sim_parameters) 

# write out to excel file
simFileName = baseFileName+'.xlsx'
with pd.ExcelWriter(simFileName) as writer:
    spdf.to_excel(writer, sheet_name='Summary')