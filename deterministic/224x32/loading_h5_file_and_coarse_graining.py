import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
np.set_printoptions(legacy='1.25') # avoid printing np.float64
from firedrake import *
from firedrake.__future__ import interpolate
import time
from firedrake.petsc import PETSc

t_instance = 45.0

print("loading high resolution mesh, velocity and temp at time t="+str(t_instance)+"......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("../896x128/h5_files/btb_CM_1_grid_128_fields_at_time_t="+str(t_instance)+".h5", 'r') as afile:
    mesh = afile.load_mesh("mesh_128")
    ua_ = afile.load_function(mesh, "atm_velocity") 
    Ta_ = afile.load_function(mesh, "atm_temperature")
    uo_ = afile.load_function(mesh, "ocean_velocity") 
    To_ = afile.load_function(mesh, "ocean_temperature")


print("finished loading! Now getting value of loaded functions on coord.... ",
    time.strftime("%H:%M:%S", time.localtime()))

res = 32

V1f = VectorFunctionSpace(mesh, "CG", 1)
V2f = FunctionSpace(mesh, "CG", 1)

print("Coarse graining atm. model variables .........",
    time.strftime("%H:%M:%S", time.localtime()))

#####Averaging and Coarse graining#########
ua_trial = TrialFunction(V1f)
Ta_trial = TrialFunction(V2f)

ua_test = TestFunction(V1f)
Ta_test = TestFunction(V2f)

ua_avg = Function(V1f)
Ta_avg = Function(V2f)

ca_sqr = Constant(1/(res**2)) # averaging solution within box of size 1/32x1/32

aa_vel = (ca_sqr * inner(grad(ua_trial), grad(ua_test)) + inner(ua_trial, ua_test)) * dx
la_vel = inner(ua_, ua_test) * dx

aa_temp = (ca_sqr * inner(grad(Ta_trial), grad(Ta_test)) + Ta_trial*Ta_test) * dx
la_temp = Ta_*Ta_test* dx

bca = [DirichletBC(V1f.sub(1), Constant(0.0), (1,2))] # making sure that n.v is zero after coarse graining

# step 1: spatial averaging using Helmholtz operator
solve(aa_vel==la_vel, ua_avg, bcs = bca)
solve(aa_temp==la_temp, Ta_avg)

print("solved the PDEs for atm. (alpha-regularization)",
    time.strftime("%H:%M:%S", time.localtime()))

# projecting on coarse grid
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

print("retrieving atm. velocity data........",
    time.strftime("%H:%M:%S", time.localtime()))
uac.assign(0)
ua_avg_vals = np.array(ua_avg.at(coords_coarse_a, tolerance=1e-10))
uac.dat.data[:] = ua_avg_vals

print("retrieving atm. temperature data........",
    time.strftime("%H:%M:%S", time.localtime()))
Tac.assign(0)
Ta_avg_vals = np.array(Ta_avg.at(coords_coarse_a, tolerance=1e-10))
Tac.dat.data[:] = Ta_avg_vals

print("Coarse graining ocean model variables .........",
    time.strftime("%H:%M:%S", time.localtime()))

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

# projecting on coarse grid
V4c = VectorFunctionSpace(c_mesh, "CG", 2, variant='alfeld')

coords_func_coarse_o = Function(V4c).interpolate(SpatialCoordinate(c_mesh))
coords_coarse_o = coords_func_coarse_o.dat.data

uoc = Function(V4c)
Toc = Function(V2c)

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

uac.rename("cg_atm_vel")
Tac.rename("cg_atm_temp")
uoc.rename("cg_ocean_vel")
Toc.rename("cg_ocean_temp")

outfile = VTKFile("./results/coarse_grained_fields_btb_model_config1_t="+str(t_instance)+".pvd")
outfile.write(uac, Tac, uoc, Toc)

# saving coarse grained solution into a .h5 file
print("saving coarse grained solution into a .h5 file.....",
    time.strftime("%H:%M:%S", time.localtime()))

# uncomment to save results into .h5 file

h5_file = "./h5_files/coarse_grained_vel_temp_btb_model_config1_at_t="+str(t_instance)+"_to_mesh_32.h5"

with CheckpointFile(h5_file, 'w') as afile:
    afile.save_mesh(c_mesh)
    afile.save_function(uac)
    afile.save_function(Tac)
    afile.save_function(uoc)
    afile.save_function(Toc)

print("Simulation completed !",
    time.strftime("%H:%M:%S", time.localtime()))
