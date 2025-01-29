import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
np.set_printoptions(legacy='1.25') # avoid printing np.float64
from firedrake import *
from firedrake.__future__ import interpolate
import time

ua_truth, Ta_truth = [], []

Dx = 1/4 ; Dy = 1/4
n1 = 3 ; n2 = 28
gridpoints = np.array([[ i * Dx, Dy + j * Dy] for j in range(n1) for i in range(n2)])

print(f'Monitoring fields at points: {gridpoints}')

Dt_ob = 1 # observing at an interval of Dt_ob
t_start = 25.0
t_end = 35.0
t_stamps = np.arange(t_start, t_end, Dt_ob)

print("time_stamps:", t_stamps)

res = 32
Nx = 7*res
Ny = res
c_mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x")
c_mesh.name = "coarse_mesh"

V1c = VectorFunctionSpace(c_mesh, "CG", 1)
V2c = FunctionSpace(c_mesh, "CG", 1)

coords_func_coarse_a = Function(V1c).interpolate(SpatialCoordinate(c_mesh))
coords_coarse_a = coords_func_coarse_a.dat.data

uac = Function(V1c)
Tac = Function(V2c)

outfile = VTKFile("./results/coarse_grained_fields_btb_model_config1_t25_to_t35.pvd")

n_iter = 0
for i in t_stamps:
    
    print('time:', i)
    print("loading fine resolution mesh and velocity.....",
      "current time:",time.strftime("%H:%M:%S", time.localtime()))
    with CheckpointFile("../896x128/h5_files/btb_CM_1_grid_128_fields_at_time_t="+str(i)+".h5", 'r') as afile:
        mesh = afile.load_mesh("mesh_128")
        ua_ = afile.load_function(mesh, "atm_velocity") 
        Ta_ = afile.load_function(mesh, "atm_temperature")
    
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

    print("solving the PDE (alpha-regularization) for atm temp...",
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

    print("calculating ua, Ta, at observation points........",
        time.strftime("%H:%M:%S", time.localtime()))

    ua_truth.append(np.array(uac.at(gridpoints, tolerance=1e-10)))
    Ta_truth.append(np.array(Tac.at(gridpoints, tolerance=1e-10)))

    data_file = './data_at_more_obs_points_config1/coarse_grained_vel_temp_data_t='+str(t_start)+'_to_t='+str(t_end)+'_grid_32_more_obs_config1.npz'
    np.savez(data_file, gridpoints = gridpoints, ua_truth = np.array(ua_truth), Ta_truth= np.array(Ta_truth))

    # write field in to .pvd file
    uac.rename("cg_atm_vel")
    Tac.rename("cg_atm_temp")
    outfile.write(uac, Tac, time=i)

print("simulation completed !!!", time.strftime("%H:%M:%S", time.localtime()))
