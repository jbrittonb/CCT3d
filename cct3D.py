from fenics import *
import numpy as np
import math
import pandas as pd 
import json

from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection




# ==============================================================================================
# Utilities
# ----------------------------------------------------------------------------------------------
# helper function to create the mesh based on the coordOrigin variable
def GenMesh(Hx, Hy, Hz, nx, ny, nz, coordOrigin):
    if coordOrigin == "B":
        return BoxMesh(Point(-0.5*Hx, -0.5*Hy, 0), Point(0.5*Hx, 0.5*Hy, Hz), nx, ny, nz)
    elif coordOrigin == "C":
        return BoxMesh(Point(-0.5*Hx, -0.5*Hy, -0.5*Hz), Point(0.5*Hx, 0.5*Hy, 0.5*Hz), nx, ny, nz)
    elif coordOrigin == "S":
        return BoxMesh(Point(0, 0, 0), Point(Hx, Hy, Hz), nx, ny, nz)
    else:
        print("coordOrigin not defined or is out of spec")
        quit()




# compute the temperature gradient
def Tgrad(T, FElement):
    "Return grad(T) projected into same space as T"
    TFS = T.function_space()
    mesh = TFS.mesh()
    degree = TFS.ufl_element().degree()
    VFS = VectorFunctionSpace(mesh, FElement, degree)
    grad_T = project(grad(T), VFS)
    return grad_T

# compute the *negative* of the temperature gradient
#  so that it points in the direction of heat flux
def nTgrad(T, FElement):
    "Return grad(T) projected into same space as T"
    TFS = T.function_space()
    mesh = TFS.mesh()
    degree = TFS.ufl_element().degree()
    VFS = VectorFunctionSpace(mesh, FElement, degree)
    ngrad_T = project(-grad(T), VFS)
    return ngrad_T

# compute magnitude of the temperature gradient
def TgradMag(T, FElement):
    "Return magnitue grad(T) projected into same space as T"
    TFS = T.function_space()
    mesh = TFS.mesh()
    degree = TFS.ufl_element().degree()
    VFS = VectorFunctionSpace(mesh, FElement, degree)
    grad_T = project(grad(T), VFS)
    Tx, Ty, Tz = grad_T.split(deepcopy=True)
    Tx_nodalVals = Tx.vector().vec().array
    Ty_nodalVals = Ty.vector().vec().array
    Tz_nodalVals = Tz.vector().vec().array
    gradMag_T = np.sqrt(Tx_nodalVals**2 + Ty_nodalVals**2 + Tz_nodalVals**2)
    
    return gradMag_T

# compute heat flux
def heatFlux(T, k, FElement):
    "Return -k*grad(T) projected into same space as T"
    TFS = T.function_space()
    mesh = TFS.mesh()
    degree = TFS.ufl_element().degree()
    VFS = VectorFunctionSpace(mesh, FElement, degree)
    heat_flux = project(-k*grad(T), VFS)
    return heat_flux







# ==============================================================================================
# Runs simulations based on the inputs in an external file;
#   - This file is a CSV or spreadsheet, each row containing all information needed
#     to simulate one simulation
#   - Iterates over all rows
#   - Note that this spreadsheet uses length in meters, temperature in Celsius
# ----------------------------------------------------------------------------------------------
def simulate_cuboid(filename):
    # read in the file
    simInputsDF = pd.read_excel(filename, sheet_name='Summary')
    print(' ')
    print('====================================================================')
    print('Running main simulation inputfile: ' + filename)
    print('--------------------------------------------------------------------')
    print(' ')
    print(' ')

    # lists for holding some data for each row of the input file
    maxMaxT_lst = []
    maxMaxGradT_lst = []
    simulatedOn = []

    # simulate based on inputs in each row
    for i in range(0, simInputsDF.index.max()+1):
        
        # ---------------------------------------------------------------
        # get inputs
        thisSimFilename = simInputsDF['ID'][i]

        print('======================================================')
        print('Loading inputs for simulation ' + str(i) + ': for file ' + thisSimFilename)

        # retreieve concrete data from the JSON file concreteLibrary.json
        concrete = simInputsDF['concrete'][i]
        with open('concreteLibrary.json') as f:
            concreteData = json.load(f)

        Hcem    = concreteData[concrete]['Hcem_J/g'] 
        Hu      = concreteData[concrete]['Hu_J/g'] 
        Ea      = concreteData[concrete]['Ea_J/mol'] 
        alphau  = concreteData[concrete]['alphau'] 
        tau_h   = concreteData[concrete]['tau_h']  
        beta    = concreteData[concrete]['beta'] 
        Cc      = concreteData[concrete]['mC_kg'] # note mC_kg is kg in 1m3 concrete so is numerically = to cement concentration Cc
        cv      = concreteData[concrete]['cvAvg_J/(kg K)']
        rho     = concreteData[concrete]['rho_kg/m3']
        k       = concreteData[concrete]['kAvg_W/(m k)']

        # concrete geometry
        W = simInputsDF["Width_m"][i] 
        D = simInputsDF["Depth_m"][i] 
        H = simInputsDF["Height_m"][i]

        # initial and ambient temperatures
        Ti_C = simInputsDF["Ti_C"][i]
        Tamb_C = simInputsDF["Tamb_C"][i]

        # duration of simulation, and timestep
        tfinal_h = simInputsDF["finalTime_h"][i]
        dt_h = simInputsDF["timestep_h"][i]

        # take snapshots?
        snapshots = simInputsDF["snapshots"][i]
        if snapshots:
            snapshot_every_x_timestep = simInputsDF["snapshot_every_x_timestep"][i]

        # resolution of finite elements
        nx = simInputsDF["nx_cells/m"][i]
        ny = simInputsDF["ny_cells/m"][i]
        nz = simInputsDF["nz_cells/m"][i]

        # finitie element type and degree
        FElement = simInputsDF["FEtype"][i]
        FEdegree = simInputsDF["FEdegree"][i]

        # load in how and where the boundary conditions are apportioned
        #   to the faces of the cuboid, i.e. a "faceSpec"; 
        #   this is in the JSON file given in the faceSpec field
        faceSpecFile = simInputsDF["faceSpec"][i]
        with open(faceSpecFile) as g:
            boundary_conditions = json.load(g)

        # some details on coordinate origin and if using symmetry arguments, how to
        #   select the subdomain to simulate
        coordOrigin = simInputsDF["coordOrigin"][i]
        halfW = simInputsDF["halfWidth"][i]
        halfD = simInputsDF["halfDepth"][i]
        halfH = simInputsDF["halfHeight"][i]

        print('...inputs loaded')



        # ---------------------------------------------------------------
        # some preprocessing
        Tr   = 294.25           # reference temperature, Kelvin
        Rgas = 8.314            # gas constant, J/(mol K)

        # convenience constants
        totCumEgen = (Hu*1000)*Cc*alphau # J/m3
        C1 = k/(rho*cv)         # thermal diffusivity, m2/s; not called alpha here
        C2 = 1/(rho*cv)         # thermal capacity per unit volume, (m3 K)/J

        # boundary condition converstions to K
        Ti    = Ti_C + 273.15   # initial temperature in K
        Tamb  = Tamb_C + 273.15 # ambient temperature in K

        # time converstions
        tfinal_s = tfinal_h*3600# final time, seconds
        dt_s = dt_h*3600        # time step size, seconds
        nt = tfinal_h/dt_h      # number of time steps

        # define the minimum and maximum axis coordinate points to be simulated
        if halfW:
            Lx = W/2
        else:
            Lx = W

        if halfD:
            Ly = D/2
        else:
            Ly = D

        if halfH:
            Lz = H/2
        else:
            Lz = H

        if coordOrigin == "B":
            minX = -0.5*Lx
            maxX =  0.5*Lx
            minY = -0.5*Ly
            maxY =  0.5*Ly
            minZ =  0
            maxZ =  Lz
        elif coordOrigin == "C":
            minX = -0.5*Lx
            maxX =  0.5*Lx
            minY = -0.5*Ly
            maxY =  0.5*Ly
            minZ = -0.5*Lz
            maxZ =  0.5*Lz   
        elif coordOrigin == "S":
            minX = 0
            maxX = Lx
            minY = 0
            maxY = Ly
            minZ = 0
            maxZ = Lz        
        else:
            print("coordOrigin not defined or is out of spec")
            quit()


        # set up files to read out time series data
        thisSimDataFilename = 'output/' + thisSimFilename + '_Tsolution.xdmf'
        XDMF_Tfile = XDMFFile(thisSimDataFilename)
        
        # thisSimGradDataFilename = 'output/' + thisSimFilename + '_TGradSolution.xdmf'
        # XDMF_gradFile = XDMFFile(thisSimGradDataFilename)

        # set up snapshot filenames, if desired
        if snapshots:
            snapshotBaseFilename = thisSimFilename + '_snapshot_'


        # set up lists for holding data like maximum temperature at a timestep, etc.
        maxT_lst = []
        maxGradT_lst = []
        time_h_lst = []




        
        # ---------------------------------------------------------------
        # problem preparation

        # for defining boundary subdomains
        tol = 1e-14
        class BoundaryXmin(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], minX, tol)
        class BoundaryXmax(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], maxX, tol)
        class BoundaryYmin(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], minY, tol)
        class BoundaryYmax(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], maxY, tol)
        class BoundaryZmin(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], minZ, tol)
        class BoundaryZmax(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], maxZ, tol)

        # create mesh and define function space
        ncellsx = int(np.ceil(nx*Lx))
        ncellsy = int(np.ceil(ny*Ly))
        ncellsz = int(np.ceil(nz*Lz))
        mesh = GenMesh(Lx, Ly, Lz, ncellsx, ncellsy, ncellsz, coordOrigin)
        V = FunctionSpace(mesh, FElement, int(FEdegree))

        # variables of variational problem
        T = TrialFunction(V)         # for nonlinear use Function() instead of TrialFunction()
        v = TestFunction(V)

        # define functions for the intermediate steps (i.e. the substeps within each timestep)
        Ta = TrialFunction(V)
        Tb = TrialFunction(V)

        # iniital conditions
        Tn = interpolate(Constant(Ti), V)   # represents the solution at the previous (and initial) timestep
        Tref = interpolate(Constant(Tr), V) # just in case the reference temperature needs to be 'the same' as Ta, etc.
        ten = interpolate(Constant(0), V)   # equivalent time (te) at previous timestep


        # container for gradients, fluxes, etc.
        GVFS = VectorFunctionSpace(mesh, FElement, 2)
        gradient = interpolate(Constant((0,0,0)), GVFS)

        gradientMag = interpolate(Constant(0), V) 

        
        # assign boundary conditions
        # mark the boundaries; recall face numbering scheme:
        #   0: minX
        #   1: maxX
        #   2: minY
        #   3: maxY
        #   4: minZ
        #   5: maxZ
        # boundary_markers = FacetFunction('size_t', mesh)
        # boundary_markers.set_all(9999)
        boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 9999)

        bxmin = BoundaryXmin()
        bxmax = BoundaryXmax()
        bymin = BoundaryYmin()
        bymax = BoundaryYmax()
        bzmin = BoundaryZmin()
        bzmax = BoundaryZmax()

        bxmin.mark(boundary_markers, 0)
        bxmax.mark(boundary_markers, 1)
        bymin.mark(boundary_markers, 2)
        bymax.mark(boundary_markers, 3)
        bzmin.mark(boundary_markers, 4)
        bzmax.mark(boundary_markers, 5)

        # redefine boundary integration measure
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        # collect convective boundary conditions
        integrals_Conv = []
        for j in boundary_conditions:
            if 'Convective' in boundary_conditions[j]:
                hconv = boundary_conditions[j]['Convective']
                jint = int(j)
                integrals_Conv.append(dt_s*(C2*hconv)*(Tb-Tamb)*v*ds(jint))

        # collect symmetry boundary conditions
        integrals_Symm = []
        for j in boundary_conditions:
            if 'Symmetry' in boundary_conditions[j]:
                Csymm = boundary_conditions[j]['Symmetry']
                jint = int(j)
                integrals_Symm.append(dt_s*Csymm*v*ds(jint))




        # ---------------------------------------------------------------
        # define variational problem
        #   this is done using a Strang splitting method,
        #   specifically RDR, for reaction-diffusion-reaction
        # 
        #   the basic idea is that at each timestep:
        #       - reaction: advance temperature "adiabatically": no heat transfer, but use Schindler egen for 1/2 timestep
        #       - diffusion: using this new temperature field, simulate a purely diffusive problem (i.e. only heat transfer) for a full timestep
        #       - reaction: with the newly diffused temperature field, advance temperature adiabatically for a second 1/2 timestep
        
        # first step is a simple Euler advance of (adiabatic) temperature over
        #  half a time step, which arises due to the Strang splitting.  
        #  The function Tadv does this; in the time loop it's returned as Ta
        def Tadv(Ea, Rgas, tau_h, beta, totCumEgen, C2, ten, delt, Tref, T):
            "advance (adiabatic) temperature by Euler method"
            V = T.function_space()
            Tmesh = V.mesh()
            degree = V.ufl_element().degree()
            W = FunctionSpace(Tmesh, 'P', degree)
            teq = project(ten + (delt/2)*exp((Ea/Rgas)*((1/Tref) - (1/T))), W)
            Tnext = project(T + ( (delt/2)*C2*totCumEgen*exp(-1*(tau_h*3600/teq)**beta)*((tau_h*3600/teq)**beta)*(beta/teq)*exp((Ea/Rgas)*((1/Tref) - (1/T))) ), W)
            return teq, Tnext

        # second step is a pure diffusion step; the Robin boundary condition is inherently included here 
        # F2 = Tb*v*dx + dt_s*C1*dot(grad(Tb), grad(v))*dx + dt_s*(C2*hconv)*(Tb-Tamb)*v*ds - Ta*v*dx
        F2 = Tb*v*dx + dt_s*C1*dot(grad(Tb), grad(v))*dx + sum(integrals_Conv) + sum(integrals_Symm) - Tn*v*dx
        a2 = lhs(F2)
        L2 = rhs(F2)

        # third step is essentially the same as the first step, only using the
        #  temperature (Tb) from the second step


        # Assemble matrix for second step
        #  doing this now slightly speeds execution
        A2 = assemble(a2)




        # ---------------------------------------------------------------
        # SOLVING
        # error/information logging
        CRITICAL  = 50      # errors that may lead to data corruption and suchlike
        ERROR     = 40      # things that go boom
        WARNING   = 30      # things that may go boom later
        INFO      = 20      # information of general interest
        PROGRESS  = 16      # what's happening (broadly)
        TRACE     = 13      # what's happening (in detail)
        DBG       = 10      # sundry
        set_log_level(WARNING)

        t = 0
        ni = 0
        T = Function(V)
        Tb = Function(V)

        print("Beginning time loop")
        print(" -------- ")
        from timeit import default_timer as timer
        start = timer()
        # .......................................................
        while t < tfinal_s:

            # Update current time and iteration number
            t += dt_s    
            ni += 1

            # first step: adiabatic advance over 1/2 timestep
            te, Ta = Tadv(Ea, Rgas, tau_h, beta, totCumEgen, C2, ten, dt_s, Tref, Tn)
            ten.assign(te)
            Tn.assign(Ta)

            # second step: diffusion without any {$\dot e_gen$} over full timestep
            # solve(a2 == L2, Tb)
            b2 = assemble(L2)
            solve(A2, Tb.vector(), b2)
            Tn.assign(Tb)

            # third step: adiabatic advance over 1/2 timestep
            te, T = Tadv(Ea, Rgas, tau_h, beta, totCumEgen, C2, ten, dt_s, Tref, Tn)
            ten.assign(te)
            
            percentDone = (ni/nt)*100
            if ni % 20 == 0:
                print(" .... ")
                print("Time loop {:3.1f}% done".format(percentDone))
                print(" .... ")
            if ni % 5 == 0:
                print(" time = {:3.3f} hours;   maximum temperature = {:3.3f} deg. C".format(t/3600, T.vector().max()-273.15))

            # write out snapshot if it's time to
            if snapshots:
                if (t/dt_s % snapshot_every_x_timestep == 0):
                    currentTime = t/3600
                    currentSnapshotFilename = 'output/' + snapshotBaseFilename + str(currentTime) + 'h.pvd'
                    VTKfile = File(currentSnapshotFilename)
                    VTKfile << (T, currentTime)
                    print("    --> snapshot written, time: {:3.1f} hours <--".format(currentTime))

            # Update previous solution
            Tn.assign(T)


            # ..........................
            # compute some secondary quantities

            # compute the temperature gradient
            Tgradient = Tgrad(Tn, FElement)
            gradient.assign(Tgradient)

            # compute the negative of the temperature gradient
            # nTgradient = nTgrad(Tn, FElement)
            # gradient.assign(nTgradient)

            # compute the heat flux
            # hf = heatFlux(Tn, k, FElement)
            # gradient.assign(hf)

            # compute the magnitude of the temperature gradient
            #  here, gradientMag is a NumPy array with values of
            #  gradient magnitude at the nodes, i.e. "nodal values"
            #  of the magnitude of the gradient
            gradientMag = TgradMag(Tn, FElement)



            # ..........................
            # some other secondary data
            time_h_lst.append(t/3600)
            
            # convert location and temperature data to numpy arrays
            # xyz = V.tabulate_dof_coordinates() # array with 3 columns: x, y, z
            Tarray = T.vector().get_local()
            
            # find and collect the maximums (at each timestep)
            maxT = Tarray.max()
            maxT_lst.append(maxT)
            maxGradT_lst.append(np.max(gradientMag))

            # find location of maximum temperature
            # indxMaxT = np.argwhere(Tarray==Tarray.max())[0][0]
            # x_maxT = xyz[indxMaxT][0]
            # y_maxT = xyz[indxMaxT][1]
            # z_maxT = xyz[indxMaxT][2]

            # find minimum temperature and its location
            # indxMinT = np.argwhere(Tarray==Tarray.min())[0][0]
            # x_minT = xyz[indxMinT][0]
            # y_minT = xyz[indxMinT][1]
            # z_minT = xyz[indxMinT][2]

            # find temperature of the surface orthogonal to the location
            #  of maximum temperature 
            # DO LATER IF NEEDED/WANTED
            
            
            # write out raw simulation data
            XDMF_Tfile.write(Tn, t)
            # XDMF_gradFile.write(gradient, t)

        # .......................................................
        end = timer()

        print(" -------- ")
        print("Elapsed time in solution loop: {:3.2f} seconds".format((end - start)))
        print("   ")

        # close the main output data files
        XDMF_Tfile.close()
        # XDMF_gradFile.close()

        # if writing out only maximums for each simulation (max over all timesteps)
        maxMaxT_lst.append(max(maxT_lst))
        maxMaxGradT_lst.append(max(maxGradT_lst))

        now = datetime.now()
        simulatedOn.append(now.strftime("%d/%B/%Y %H:%M:%S"))
        
        # write out timeseries of max temperature and max gradient to HDF5 file
        maxT_C_arr = np.array(maxT_lst) - 273.15
        maxT_F_arr = maxT_C_arr*(9/5) + 32
        df = pd.DataFrame({'time_h': time_h_lst,
                           'maxT_K': maxT_lst,
                           'maxT_C': maxT_C_arr,
                           'maxT_F': maxT_F_arr,
                           'maxGradT_C/m': maxGradT_lst})
        h5Filename = 'output/'+filename[0:-5] + '_timeSeriesOfMaxima.h5'
        df.to_hdf(h5Filename, key=thisSimFilename, mode='a')



    # finished simulating all the rows
    # use the input data dataframe, and append the maxima
    simInputsDF['simulated on'] = simulatedOn
    simInputsDF['maxT_K'] = maxMaxT_lst
    simInputsDF['maxT_C'] = np.add(maxMaxT_lst, - 273.15)
    simInputsDF['maxT_F'] = (np.array(maxMaxT_lst) - 273.15)*(9/5)+32
    simInputsDF['maxGradT_C/m'] = maxMaxGradT_lst

    # write out the maximum data to the Excel file using Pandas
    # get the filename without the ".xlsx"
    # baseFilename = filename[0:-5]
    # simFileName = baseFilename + '_simSummary.xlsx'
    simFileName = filename
    with pd.ExcelWriter(simFileName) as writer:
        simInputsDF.to_excel(writer, sheet_name='Summary')

    







# ==============================================================================================
# Sets up the array of cuboids used in simulation; these are 'full size' and don't
#   take advantage of symmetries so that only portions of the cuboid are simulated
# ----------------------------------------------------------------------------------------------
def define_cuboids(smallest, biggest, step):
    stopAt = biggest + step
    if np.equal(((biggest/step - smallest/step) - 1), 0.0):
        W = np.linspace(biggest, smallest, num=2)
    else:
        W = np.arange(smallest, stopAt, step)   # width
    D = W                                   # depth
    H = W                                   # height

    DD, HH, WW = np.meshgrid(D, H, W) 

    # for reference later, compute volume, surface area,
    #   area to volume, and the nondimensional ratio A^3/V^2
    # these matrices are symmetric; to prevent redundnacy and
    #   to make indexing easier, take the lower triangle of each
    #   [using np.tril()]and flatten the matrices to vectors
    #   [using np.ravel()]
    # volume
    V = DD*HH*WW 

    # surface area
    A = 2*(WW*DD) + 2*(WW*HH) + 2*(DD*HH)

    # surface area to volume
    AtoV = np.ravel(np.tril(A/V))

    # nondimensional surface area to volume A^3/V^2
    ndAtoV = np.ravel( np.tril(A**3/(V**2)) )

    V = np.ravel(np.tril(V))
    A = np.ravel(np.tril(A))

    # for indexing
    WW = np.ravel(WW)
    DD = np.ravel(DD)
    HH = np.ravel(HH)

    # get rid of the remaining zeros
    p = np.nonzero(V)
    WWout = WW[np.nonzero(V)]
    DDout = DD[np.nonzero(V)]
    HHout = HH[np.nonzero(V)]
    Aout = A[np.nonzero(V)]
    Vout = V[np.nonzero(V)]
    AtoVout = AtoV[np.nonzero(V)]
    ndAtoVout = ndAtoV[np.nonzero(V)]


    return WWout, DDout, HHout, Aout, Vout, AtoVout, ndAtoVout;




# ==============================================================================================
# Another way to set up an array of cuboids used in simulation; here a user manually specifies
# width, depth, and height them rather than "winding a clock to make cuboids"
#   these are 'full size' and don't
#   take advantage of symmetries so that only portions of the cuboid are simulated
# ----------------------------------------------------------------------------------------------
def establish_cuboids(W, D, H):
    """
    W, D, and H are lists giving individual sizes
    """
    # volume
    V = W*D*H
    
     # surface area
    A = 2*(W*D) + 2*(W*H) + 2*(D*H)
    
    # heat transfer surface area
    qA = A - (W*D)
    
    AtoV = A/V
    qAtoV = qA/V
    
    ndAtoV = A**3/V**2 
    ndqAtoV = qA**3/V**2 

    return W, D, H, A, V, AtoV, ndAtoV;





# ==============================================================================================
# Simply plots a cuboid in 3D
# ----------------------------------------------------------------------------------------------
def plot_cuboid(cuboid_definition, maxLength):
    cuboid_definition_array = [
        np.array(list(item))
        for item in cuboid_definition
    ]

    points = []
    points += cuboid_definition_array
    vectors = [
        cuboid_definition_array[1] - cuboid_definition_array[0],
        cuboid_definition_array[2] - cuboid_definition_array[0],
        cuboid_definition_array[3] - cuboid_definition_array[0]
    ]

    points += [cuboid_definition_array[0] + vectors[0] + vectors[1]]
    points += [cuboid_definition_array[0] + vectors[0] + vectors[2]]
    points += [cuboid_definition_array[0] + vectors[1] + vectors[2]]
    points += [cuboid_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,1,0.1))

    ax.add_collection3d(faces)

    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    ax.set_xlim3d(0, maxLength)
    ax.set_ylim3d(0, maxLength)
    ax.set_zlim3d(0, maxLength)

    # Functions from @Mateen Ulhaq and @karlo
    def set_axes_equal(ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        _set_axes_radius(ax, origin, radius)

    def _set_axes_radius(ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])
    
    ax.set_box_aspect((1, 1, 1))
    set_axes_equal(ax)

    # plt.show()
    plt.show(block=False) 
    plt.pause(3)
    plt.close("all")


if __name__ == '__main__':
    print(' - ')
    filename = 'TEST.xlsx'
    print(filename)
    simulate_cuboid(filename)
    print(' - ')
    print('  ')





