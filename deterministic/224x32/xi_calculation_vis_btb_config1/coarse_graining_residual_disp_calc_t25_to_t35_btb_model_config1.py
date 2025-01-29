import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
np.set_printoptions(legacy='1.25') # avoid printing np.float64
from firedrake import *
from firedrake.__future__ import interpolate
import time

dX = []
ua_truth, Ta_truth, uo_truth, To_truth = [], [], [], []

gridpoints = np.array([[1.75,0.25], [3.5,0.25], [5.25,0.25],[1.75,0.75], [3.5,0.75], [5.25,0.75]])

print(f'Monitoring fields at points: {gridpoints}')

Dt_uc = 0.04 # assumed decorrelated time (same as Dt)
t_start = 25
t_end = 35
time_array = np.arange(t_start, t_end, Dt_uc)
t_stamps = np.round(time_array, 2)

print("time_stamps:", time_array)

res = 32
Nx = 7*res
Ny = res
c_mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")
c_mesh.name = "coarse_mesh"

V1c = VectorFunctionSpace(c_mesh, "CG", 1)
V2c = FunctionSpace(c_mesh, "CG", 1)
V4c = VectorFunctionSpace(c_mesh, "CG", 2, variant='alfeld')

coords_func_coarse_a = Function(V1c).interpolate(SpatialCoordinate(c_mesh))
coords_coarse_a = coords_func_coarse_a.dat.data

coords_func_coarse_o = Function(V4c).interpolate(SpatialCoordinate(c_mesh))
coords_coarse_o = coords_func_coarse_o.dat.data

uac = Function(V1c)
Tac = Function(V2c)
uoc = Function(V4c)
Toc = Function(V2c)

n_iter = 0
for i in t_stamps:
    
    print('time:', i)
    print("loading fine resolution mesh and velocity.....",
      "current time:",time.strftime("%H:%M:%S", time.localtime()))
    with CheckpointFile("../../896x128/h5_files/btb_CM_1_grid_128_fields_at_time_t="+str(i)+".h5", 'r') as afile:
        mesh = afile.load_mesh("mesh_128")
        ua_ = afile.load_function(mesh, "atm_velocity") 
        Ta_ = afile.load_function(mesh, "atm_temperature")
        uo_ = afile.load_function(mesh, "ocean_velocity") 
        To_ = afile.load_function(mesh, "ocean_temperature")
    
    print("finished loading! Now getting value of loaded functions on coord.... ",
    time.strftime("%H:%M:%S", time.localtime()))

    V1f = VectorFunctionSpace(mesh, "CG", 1)

    print("Coarse graining atm. model variables .........",
        time.strftime("%H:%M:%S", time.localtime()))

    #####Averaging and Coarse graining#########
    ua_trial = TrialFunction(V1f)
    ua_test = TestFunction(V1f)
    ua_avg = Function(V1f)

    ca_sqr = Constant(1/(res**2)) # averaging solution within box of size 1/32x1/32

    aa_vel = (ca_sqr * inner(grad(ua_trial), grad(ua_test)) + inner(ua_trial, ua_test)) * dx
    la_vel = inner(ua_, ua_test) * dx

    bca = [DirichletBC(V1f.sub(1), Constant(0.0), (1,2))] # making sure that n.v is zero after coarse graining

    # step 1: spatial averaging using Helmholtz operator
    solve(aa_vel==la_vel, ua_avg, bcs = bca)

    print("solved the PDEs for atm. (alpha-regularization)",
    time.strftime("%H:%M:%S", time.localtime()))

    print("retrieving atm. velocity data........",
    time.strftime("%H:%M:%S", time.localtime()))
    uac.assign(0)
    ua_avg_vals = np.array(ua_avg.at(coords_coarse_a, tolerance=1e-10))
    uac.dat.data[:] = ua_avg_vals

    print("calculating (u - u_avg)Dt........",
        time.strftime("%H:%M:%S", time.localtime()))

    dX.append(Dt_uc*(np.array(ua_.at(coords_coarse_a, tolerance=1e-10)) 
                - np.array(uac.at(coords_coarse_a, tolerance=1e-10))))
    
    print("Calculation done, saving the data into a separate file",
        time.strftime("%H:%M:%S", time.localtime()))
    
    dX_x = np.array(dX)[:,:,0]
    print("shape of dX1_x:", dX_x.shape)

    dX_y = np.array(dX)[:,:,1]
    print("shape of dX1_y:", dX_y.shape)

    data_file_1 = './data_for_xi_calculation/dX_data_t='+str(t_start)+'_to_t='+str(t_end)+'_grid_32_decor_t_1Dt_config1.npz'
    np.savez(data_file_1, dX_x = dX_x, dX_y = dX_y)

    if round(i - int(i), 2) == 0:

        print("solving the PDE (alpha-regularization) for atm temp. ocean temp and ocean vel.",
        time.strftime("%H:%M:%S", time.localtime()))

        V2f = FunctionSpace(mesh, "CG", 1)
        Ta_trial = TrialFunction(V2f)
        Ta_test = TestFunction(V2f)
        Ta_avg = Function(V2f)

        aa_temp = (ca_sqr * inner(grad(Ta_trial), grad(Ta_test)) + Ta_trial*Ta_test) * dx
        la_temp = Ta_*Ta_test* dx

        solve(aa_temp==la_temp, Ta_avg)

        print("solved the PDEs for atm. temp (alpha-regularization)",time.strftime("%H:%M:%S", time.localtime()))

        print("retrieving atm. temperature data........",time.strftime("%H:%M:%S", time.localtime()))
        Tac.assign(0)
        Ta_avg_vals = np.array(Ta_avg.at(coords_coarse_a, tolerance=1e-10))
        Tac.dat.data[:] = Ta_avg_vals

        print("Coarse graining ocean model variables .........",time.strftime("%H:%M:%S", time.localtime()))

        V4f = VectorFunctionSpace(mesh, "CG", 2, variant='alfeld') # barycentric refinement

        #####Averaging and Coarse graining#########
        uo_trial = TrialFunction(V4f)
        To_trial = TrialFunction(V2f)

        uo_test = TestFunction(V4f)
        To_test = TestFunction(V2f)

        uo_avg = Function(V4f)
        To_avg = Function(V2f)

        co_sqr = Constant(1/((2*res)**2)) # averaging solution within box of size 1/32x1/32

        ao_vel = (co_sqr * inner(grad(uo_trial), grad(uo_test)) + inner(uo_trial, uo_test)) * dx
        lo_vel = inner(uo_, uo_test) * dx

        ao_temp = (ca_sqr * inner(grad(To_trial), grad(To_test)) + To_trial*To_test) * dx
        lo_temp = To_*To_test* dx

        bco = [DirichletBC(V4f.sub(1), Constant(0.0), (1,2))] # making sure that n.v is zero after coarse graining

        # step 1: spatial averaging using Helmholtz operator
        solve(ao_vel==lo_vel, uo_avg, bcs = bco)
        solve(ao_temp==lo_temp, To_avg)

        print("solved the PDEs for ocean (alpha-regularization)",
            time.strftime("%H:%M:%S", time.localtime())) 
        
        print("retrieving ocean velocity data........",
            time.strftime("%H:%M:%S", time.localtime()))
        uoc.assign(0)
        uo_avg_vals = np.array(uo_avg.at(coords_coarse_o, tolerance=1e-10))
        uoc.dat.data[:] = uo_avg_vals

        print("retrieving ocean temperature data........",
            time.strftime("%H:%M:%S", time.localtime()))
        Toc.assign(0)
        To_avg_vals = np.array(To_avg.at(coords_coarse_a, tolerance=1e-10))
        Toc.dat.data[:] = To_avg_vals

        ua_truth.append(np.array(uac.at(gridpoints, tolerance=1e-10)))
        Ta_truth.append(np.array(Tac.at(gridpoints, tolerance=1e-10)))
        uo_truth.append(np.array(uoc.at(gridpoints, tolerance=1e-10)))
        To_truth.append(np.array(Toc.at(gridpoints, tolerance=1e-10)))

        data_file_2 = '../data_at_obs_points_btb_config1/coarse_grained_vel_temp_data_t='+str(t_start)+'_to_t='+str(t_end)+'_grid_32_decor_1Dt_config1.npz'
        np.savez(data_file_2, gridpoints = gridpoints, ua_truth = np.array(ua_truth), Ta_truth= np.array(Ta_truth), uo_truth = np.array(uo_truth),  To_truth = np.array(To_truth))

print("simulation completed !!!", time.strftime("%H:%M:%S", time.localtime()))
