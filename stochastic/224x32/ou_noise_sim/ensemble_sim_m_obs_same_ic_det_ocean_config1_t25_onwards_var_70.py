import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
np.set_printoptions(legacy='1.25') # avoid printing np.float64
from firedrake import *
from firedrake.__future__ import interpolate
import time
from firedrake.petsc import PETSc
from utilities import OU_mat

my_ensemble = Ensemble(COMM_WORLD, 1)
spatial_comm = my_ensemble.comm
ensemble_comm = my_ensemble.ensemble_comm

PETSc.Sys.Print(f'size of ensemble is {ensemble_comm.size}')
PETSc.Sys.Print(f'size of spatial communicators is {spatial_comm.size}')

var = 70 # variance level in percentage
res = 32
Dt = 0.04
init_time = 25.0

# dimensionless constants for atmosphere
Ro_a = 0.3 # Rossby number
C_a = 0.02
vis_a = 1/(10**4) # eddy viscosity
diff_a = 1/(10**4) # diffusion coefficient

# dimensionless constants for ocean
Ro_o = Ro_a # Rossby number
vis_o = 1/(10**4) # eddy viscosity
diff_o = 1/(10**4) # diffusion coefficient

# coupling coefficients
gamma = -10
sigma = -0.1

PETSc.Sys.Print(f"Simulation parameters: Resolution = {res}, Dt = {Dt}, eddy viscosity = {vis_a}, diffusion coef. = {diff_a}, coup. coef. sigma = {sigma}, coup. coef. gamma = {gamma}")

Nx = 7*res
Ny = res
mesh = PeriodicRectangleMesh(Nx, Ny, 7, 1, direction="x", comm = spatial_comm)

V0 = FunctionSpace(mesh, "DG", 0)
V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V3 = FunctionSpace(mesh, "CG", 2)
V4 = VectorFunctionSpace(mesh, "CG", 2, variant='alfeld') # barycentric refinement
V5 = FunctionSpace(mesh, "DG", 1, variant='alfeld')

x, y = SpatialCoordinate(mesh)

ua_ = Function(V1) # on a normal mesh
Ta_ = Function(V2)
uo_ = Function(V4) # on a bary split macro element
To_ = Function(V2)

data = np.load('../vel_temp_data/vel_temp_fields_coarse_grained_as_arrays_config1_t'+str(init_time)+'.npz')

ua_array = data['ua_array']
Ta_array = data['Ta_array']
uo_array = data['uo_array']
To_array = data['To_array']

ua_.dat.data[:] = ua_array
Ta_.dat.data[:] = Ta_array
uo_.dat.data[:] = uo_array
To_.dat.data[:] = To_array

seed_no = ensemble_comm.rank + 51

# pvd_print = 10
# if ensemble_comm.rank%pvd_print == 0: # printing results corresponding to every pvd_print particle
#     To_.rename("ocean_temperature")
#     Ta_.rename("atm_temperature")
#     ua_.rename("atm_velocity")
#     uo_.rename("ocean_velocity")
#     outfile = VTKFile('../results/OU_sim_var_'+str(var)+'_mesh_32_particle_'+str(ensemble_comm.rank + 1)+'_det_ocean_config1_t'+str(int(init_time))+'_onwards_same_ic.pvd', comm = spatial_comm)
#     outfile.write(ua_, Ta_, uo_, To_, time=init_time)

################ Atmosphere model #########################################
Z1 = V1*V2 # functions for the atm. model
w1 = Function(Z1)
ua,Ta = split(w1)
va,phi_a = TestFunctions(Z1)
ua_pert = Function(V1)

perp = lambda arg: as_vector((-arg[1], arg[0]))

F1 = ( inner(ua - ua_, va)
    + Dt*0.5*(inner(dot(ua, nabla_grad(ua)), va) + inner(dot(ua_, nabla_grad(ua_)), va))
    + Dt*0.5*(1/Ro_a)*inner(perp(ua) + perp(ua_), va)
    - Dt*0.5*(1/C_a)*(Ta + Ta_)* div(va)
    + Dt *0.5 *vis_a*inner((grad(ua) + grad(ua_)), grad(va))
    + np.sqrt(Dt)*0.5*(inner(dot(ua_pert, nabla_grad(ua)), va) + inner(dot(ua_pert, nabla_grad(ua_)), va))
    + np.sqrt(Dt)*0.5*(inner((ua_[0]+ua[0])*grad(ua_pert[0]) + (ua_[1]+ua[1])*grad(ua_pert[1]) , va))
    + (Ta -Ta_)*phi_a
    - Dt*0.5*inner(Ta_*ua_ + Ta*ua, grad(phi_a))
    + Dt*0.5*(diff_a)*inner((grad(Ta) + grad(Ta_)),grad(phi_a))
    - Dt*gamma*(0.5*(Ta +Ta_) - To_)* phi_a
    + np.sqrt(Dt)*0.5*inner(ua_pert, grad(Ta) + grad(Ta_))*phi_a)* dx 

bc1 = [DirichletBC(Z1.sub(0).sub(1), Constant(0.0), (1,2))] # no penetration bc

prob1 = NonlinearVariationalProblem(F1, w1, bcs=bc1)
solver1 = NonlinearVariationalSolver(prob1)

############Helmholtz problem to extract div. free part######################
q_trial = TrialFunction(V3)
psi = TestFunction(V3)
div_u = Function(V2)

Aq = -inner(grad(q_trial), grad(psi))*dx
Lq = div_u*psi*dx

q = Function(V3)
prob_q = LinearVariationalProblem(Aq, Lq, q)
solver_q = LinearVariationalSolver(prob_q, nullspace=VectorSpaceBasis(constant=True,comm=COMM_WORLD))

########## Ocean model ########################################################
Z2 = V4*V5*V2 # functions for the ocean model
w2 = Function(Z2)
uo, p, To = split(w2)
vo, eta, phi_o = TestFunctions(Z2)

ua_sol = Function(V1)
ua_sol_avg = Function(V1)
ua_sol_ = Function(V1)
ua_sol_avg_ = Function(V1)
u_sum = Function(V1)

F2 = ( inner(uo-uo_,vo)
    + Dt*0.5*(inner(dot(uo, nabla_grad(uo)), vo) + inner(dot(uo_, nabla_grad(uo_)), vo))
    + Dt*0.5*(1/Ro_o)*inner(perp(uo) + perp(uo_), vo)
    - Dt*(1/Ro_o)*p*div(vo) 
    + Dt *0.5 *vis_o*inner((grad(uo) + grad(uo_)), grad(vo))
    - Dt*0.5*sigma*inner((uo - ua_sol_avg + uo_ - ua_sol_avg_),vo)
    + Dt*div(uo)*eta
    + (To -To_)*phi_o 
    + Dt*0.5*(inner(uo_,grad(To_)) + inner(uo,grad(To)))*phi_o
    + Dt*0.5*diff_o*inner((grad(To) + grad(To_)),grad(phi_o)))* dx 

bc2 = [DirichletBC(Z2.sub(0).sub(1), Constant(0.0), (1,2))] # no penetration bc
ns_ocean = MixedVectorSpaceBasis(Z2, [Z2.sub(0),VectorSpaceBasis(constant=True,comm=COMM_WORLD), Z2.sub(2)]) # null space for the ocean equations

prob2 = NonlinearVariationalProblem(F2, w2, bcs=bc2)
solver2 = NonlinearVariationalSolver(prob2, nullspace=ns_ocean)

# solving Helmholtz problem to get ua_sol at t=0
div_u.assign(Function(V2).interpolate(div(ua_))) 
solver_q.solve()
ua_sol_.assign(ua_ - Function(V1).interpolate(grad(q)))

# ensemble_comm.Barrier() # we don't need this since allreduce is already a blocking operation !
my_ensemble.allreduce(ua_sol_, u_sum)
ua_sol_avg_.dat.data[:] = (1/ensemble_comm.size)*u_sum.dat.data

xi_data = np.load('../../../deterministic/224x32/xi_calculation_vis_btb_config1/xi_vec_data/xi_matrix_53_eigv_grid_32_t=25_to_t=45_config1.npz')

xi_mat = xi_data['xi_mat']
PETSc.Sys.Print(f'loaded the xi matrix for particle {ensemble_comm.rank}, local time: {time.strftime("%H:%M:%S", time.localtime())}')

t_start = init_time + Dt
t_end = 45.0

PETSc.Sys.Print(f'seed_no for particle {ensemble_comm.rank} is {seed_no}, local time: {time.strftime("%H:%M:%S", time.localtime())}')

n_t_steps = int((t_end - t_start)/Dt)
n_EOF = 10 # desired no. of EOFs to include in stochastic sim
#######################
# 53 EOFs >>>>>> 99 % variance
# 20 EOFs >>>>>> 90 % variance
# 10 EOFs >>>>>> 70 % variance
########################
acf_data = np.load('../../../deterministic/224x32/acf_data/acf_1_data_53_EOFs_mesh_32_t25_to_t45_config1.npz')
acf1_data = acf_data['acf1_data'] # this is an array containing the lag 1 ACF for the SVD time-series data corresponding to each xi

PETSc.Sys.Print(f'loaded the acf1 data corresponding to time-series of first {acf1_data.size} xi')
np.random.seed(seed_no)
rand_mat = OU_mat(n_t_steps+2, n_EOF, acf1_data) # time-series generated from OU process


ua_data_sto, Ta_data_sto = [], []
Dx = 1/4 ; Dy = 1/4
n1 = 3 ; n2 = 28
gridpoints = np.array([[ i * Dx, Dy + j * Dy] for j in range(n1) for i in range(n2)])

ua_data_sto.append(np.array(ua_.at(gridpoints, tolerance=1e-10)))
Ta_data_sto.append(np.array(Ta_.at(gridpoints, tolerance=1e-10)))


t = init_time + Dt
iter_n = 1
freq1 = int(1/Dt)*1 # time steps in one time unit
freq2 = int(1/Dt)*5  # printing .pvd files after every freq2 time steps
big_t_step = freq1*Dt 
current_time = time.strftime('%H:%M:%S', time.localtime())
PETSc.Sys.Print("Local time at the start of simulation:",current_time)
start_time = time.time()

PETSc.Sys.Print(f'particle no:{seed_no}, local time:{round(t,4)}')

while (round(t,4)<=t_end):
    # xi calculation
    vec_u_pert = np.zeros((xi_mat.shape[1], 2))
    for i in range(n_EOF):
        vec_u_pert +=  rand_mat[iter_n-1,i]*xi_mat[i, :,:]

    ua_pert.assign(0)
    ua_pert.dat.data[:] = vec_u_pert

    # solve atmosphere model
    solver1.solve()
    ua,Ta= w1.subfunctions

    # solve Helmholtz problem
    div_u.assign(Function(V2).interpolate(div(ua))) 
    solver_q.solve()
    ua_sol.assign(ua - Function(V1).interpolate(grad(q)))

    # take average of ua_sol and pass to ocean model
    my_ensemble.allreduce(ua_sol, u_sum)
    ua_sol_avg.dat.data[:] = (1/ensemble_comm.size)*u_sum.dat.data

    # solve ocean model
    solver2.solve()
    uo, p, To = w2.subfunctions

    if iter_n%freq1 == 0:
        if iter_n==freq1:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 
            PETSc.Sys.Print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/big_t_step)*execution_time
            PETSc.Sys.Print("Approx. total running time: %.2f minutes:" %total_execution_time)
        
        PETSc.Sys.Print(f"t = {round(t,4)}, local time: {time.strftime('%H:%M:%S', time.localtime())}")
        ua_data_sto.append(np.array(ua.at(gridpoints, tolerance=1e-10)))
        Ta_data_sto.append(np.array(Ta.at(gridpoints, tolerance=1e-10)))
        data_file = '../data_stoch_more_obs/vel_temp_data_m_obs_particle_'+str(seed_no)+'_var_'+str(var)+'_grid_32_t'+str(int(init_time))+'_onwards_OU_same_ic.npz'
        np.savez(data_file, gridpoints = gridpoints, ua_data_sto = np.array(ua_data_sto), Ta_data_sto = np.array(Ta_data_sto))

    # if iter_n%freq2 == 0:
    #     if ensemble_comm.rank%pvd_print == 0: # printing results corresponding to every pvd_print particle
    #         To.rename("ocean_temperature")
    #         Ta.rename("atm_temperature")
    #         ua.rename("atm_velocity")
    #         uo.rename("ocean_velocity")
    #         outfile.write(ua_, Ta_, uo_, To_, time=round(t,4))

    ua_.assign(ua)
    Ta_.assign(Ta)
    uo_.assign(uo)
    To_.assign(To)
    ua_sol_avg_.assign(ua_sol_avg)

    t += Dt
    iter_n +=1

print("Local time at the end of simulation:",time.strftime('%H:%M:%S', time.localtime()))