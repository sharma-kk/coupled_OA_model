import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
np.set_printoptions(legacy='1.25') # avoid printing np.float64
from firedrake import *
import time
from firedrake.petsc import PETSc

t_start = 25.0
t_end = 45.0
Dt = 0.04
time_array = np.arange(t_start, t_end + Dt, Dt)
t_stamps = np.round(time_array, 2)

outfile = VTKFile("./results/vel_fields_at_coarse_tsteps_t25_to_t45.pvd")

for t in t_stamps:

    print("loading high resolution mesh, velocity and temp at time t="+str(t)+"......",
        "current time:",time.strftime("%H:%M:%S", time.localtime()))

    with CheckpointFile("../896x128/h5_files/btb_CM_1_grid_128_fields_at_time_t="+str(t)+".h5", 'r') as afile:
        mesh = afile.load_mesh("mesh_128")
        ua = afile.load_function(mesh, "atm_velocity") 
    
    print("finished loading! ", time.strftime("%H:%M:%S", time.localtime()))

    outfile.write(ua, time = t)

