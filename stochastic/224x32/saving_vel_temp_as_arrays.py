import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
np.set_printoptions(legacy='1.25') # avoid printing np.float64
from firedrake import *
from firedrake.__future__ import interpolate
import time

init_time = 25.0

print("loading high resolution mesh, velocity and temp at time t="+str(init_time)+"......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("../../deterministic/224x32/h5_files/coarse_grained_vel_temp_btb_model_config1_at_t="+str(init_time)+"_to_mesh_32.h5", 'r') as afile:
    mesh = afile.load_mesh("coarse_mesh")
    ua = afile.load_function(mesh, "cg_atm_vel") 
    Ta = afile.load_function(mesh, "cg_atm_temp")
    uo = afile.load_function(mesh, "cg_ocean_vel") 
    To = afile.load_function(mesh, "cg_ocean_temp")

print("finished loading!",
    time.strftime("%H:%M:%S", time.localtime()))

data_file ='./vel_temp_data/vel_temp_fields_coarse_grained_as_arrays_config1_t'+str(init_time)+'.npz'
np.savez(data_file, ua_array = ua.dat.data, Ta_array = Ta.dat.data, uo_array = uo.dat.data, To_array = To.dat.data)