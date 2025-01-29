import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
np.set_printoptions(legacy='1.25') # avoid printing np.float64
from firedrake import *
from firedrake.__future__ import interpolate
import time
from firedrake.petsc import PETSc

init_time = 35.0

print("loading high resolution mesh, velocity and temp at time t="+str(init_time)+"......",
    "current time:",time.strftime("%H:%M:%S", time.localtime()))

with CheckpointFile("./h5_files/coarse_grained_vel_temp_btb_model_config1_at_t="+str(init_time)+"_to_mesh_32.h5", 'r') as afile:
    mesh = afile.load_mesh("coarse_mesh")
    ua_ = afile.load_function(mesh, "cg_atm_vel") 
    Ta_ = afile.load_function(mesh, "cg_atm_temp")
    uo_ = afile.load_function(mesh, "cg_ocean_vel") 
    To_ = afile.load_function(mesh, "cg_ocean_temp")

res = 32
Dt = 0.04
file_name = "det_run_grid_32_t"+str(init_time)+"_onwards_btb_config_1" # file name where the results are saved

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

print(f"Simulation parameters: Resolution = {res}, Dt = {Dt}, eddy viscosity = {vis_a}, diffusion coef. = {diff_a}, coup. coef. sigma = {sigma}, coup. coef. gamma = {gamma}")

V0 = FunctionSpace(mesh, "DG", 0)
V1 = VectorFunctionSpace(mesh, "CG", 1)
V2 = FunctionSpace(mesh, "CG", 1)
V3 = FunctionSpace(mesh, "CG", 2)
V4 = VectorFunctionSpace(mesh, "CG", 2, variant='alfeld') # barycentric refinement
V5 = FunctionSpace(mesh, "DG", 1, variant='alfeld')

################ Atmosphere model #########################################
Z1 = V1*V2 # functions for the atm. model
w1 = Function(Z1)
ua,Ta = split(w1)
va,phi_a = TestFunctions(Z1)

perp = lambda arg: as_vector((-arg[1], arg[0]))

F1 = ( inner(ua - ua_, va)
    + Dt*0.5*(inner(dot(ua, nabla_grad(ua)), va) + inner(dot(ua_, nabla_grad(ua_)), va))
    + Dt*0.5*(1/Ro_a)*inner(perp(ua) + perp(ua_), va)
    - Dt*0.5*(1/C_a)*(Ta + Ta_)* div(va)
    + Dt *0.5 *vis_a*inner((grad(ua) + grad(ua_)), grad(va))
    + (Ta -Ta_)*phi_a
    - Dt*0.5*inner(Ta_*ua_ + Ta*ua, grad(phi_a))
    + Dt*0.5*(diff_a)*inner((grad(Ta) + grad(Ta_)),grad(phi_a))
    - Dt*gamma*(0.5*(Ta +Ta_) - To_)* phi_a)* dx 

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
ua_sol_ = Function(V1)

F2 = ( inner(uo-uo_,vo)
    + Dt*0.5*(inner(dot(uo, nabla_grad(uo)), vo) + inner(dot(uo_, nabla_grad(uo_)), vo))
    + Dt*0.5*(1/Ro_o)*inner(perp(uo) + perp(uo_), vo)
    - Dt*(1/Ro_o)*p*div(vo) 
    + Dt *0.5 *vis_o*inner((grad(uo) + grad(uo_)), grad(vo))
    - Dt*0.5*sigma*inner((uo - ua_sol + uo_ - ua_sol_),vo)
    + Dt*div(uo)*eta
    + (To -To_)*phi_o 
    + Dt*0.5*(inner(uo_,grad(To_)) + inner(uo,grad(To)))*phi_o
    + Dt*0.5*diff_o*inner((grad(To) + grad(To_)),grad(phi_o)))* dx 

bc2 = [DirichletBC(Z2.sub(0).sub(1), Constant(0.0), (1,2))] # no penetration bc
ns_ocean = MixedVectorSpaceBasis(Z2, [Z2.sub(0),VectorSpaceBasis(constant=True,comm=COMM_WORLD), Z2.sub(2)]) # nullspace for the ocean equations

prob2 = NonlinearVariationalProblem(F2, w2, bcs=bc2)
solver2 = NonlinearVariationalSolver(prob2, nullspace=ns_ocean)

# solving Helmholtz problem to get ua_sol 
div_u.assign(Function(V2).interpolate(div(ua_))) 
solver_q.solve()
ua_sol_.assign(ua_ - Function(V1).interpolate(grad(q)))

# saving initial conditions
To_.rename("ocean_temperature")
Ta_.rename("atm_temperature")
ua_.rename("atm_velocity")
uo_.rename("ocean_velocity")

outfile = VTKFile("./results/"+file_name+".pvd")
print(f"Saving initial condition as .pvd file....., local time: {time.strftime('%H:%M:%S', time.localtime())}")
outfile.write(ua_, Ta_, uo_, To_, time= init_time)

# energy_ = 0.5*(norm(ua_)**2)
# print(f"kinetic energy at t=0.0: {round(energy_,6)}")

ua_data_det, Ta_data_det = [], []

Dx = 1/4 ; Dy = 1/4
n1 = 3 ; n2 = 28
gridpoints = np.array([[ i * Dx, Dy + j * Dy] for j in range(n1) for i in range(n2)])

ua_data_det.append(np.array(ua_.at(gridpoints, tolerance=1e-10)))
Ta_data_det.append(np.array(Ta_.at(gridpoints, tolerance=1e-10)))

t_start = init_time + Dt
t_end = 45

t = init_time + Dt
iter = 1

freq1 = int(1/Dt)*1 # time steps in one time unit
freq2 = int(1/Dt)*5  # printing .pvd files after every freq2 time steps
# freq3 = int(1/Dt)*5 # saving .h5 files after every freq3 time steps
big_tstep = freq1*Dt # printing results at time-steps of size big_tstep

current_time = time.strftime('%H:%M:%S', time.localtime())
print("Local time at the start of simulation:",current_time)
start_time = time.time()

while (round(t,4)<=t_end):
    # solve atmosphere model
    solver1.solve()
    ua,Ta= w1.subfunctions

    # solve Helmholtz problem
    div_u.assign(Function(V2).interpolate(div(ua))) 
    solver_q.solve()
    ua_sol.assign(ua - Function(V1).interpolate(grad(q)))

    # solve ocean model
    solver2.solve()
    uo, p, To = w2.subfunctions

    if iter%freq1 == 0:
        if iter==freq1:
            end_time = time.time()
            execution_time = (end_time-start_time)/60 
            print("Approx. running time for one t_step: %.2f minutes" %execution_time)
            total_execution_time = ((t_end - t_start)/big_tstep)*execution_time
            print("Approx. total running time: %.2f minutes:" %total_execution_time)
        
        # energy = 0.5*(norm(ua)**2)
        print(f"t = {round(t,4)}, local time: {time.strftime('%H:%M:%S', time.localtime())}")
        # print(f"kinetic energy: {round(energy,6)}")
        ua_data_det.append(np.array(ua.at(gridpoints, tolerance=1e-10)))
        Ta_data_det.append(np.array(Ta.at(gridpoints, tolerance=1e-10)))
        data_file = './data_at_more_obs_points_config1/vel_temp_det_data_t'+str(int(init_time))+'_onwards_m_obs.npz'
        np.savez(data_file, gridpoints = gridpoints, ua_data_det = np.array(ua_data_det), Ta_data_det = np.array(Ta_data_det))

    if iter%freq2 == 0:
        To.rename("ocean_temperature")
        Ta.rename("atm_temperature")
        ua.rename("atm_velocity")
        uo.rename("ocean_velocity")
        outfile.write(ua, Ta, uo, To, time=round(t,4))
        print(f"Saved fields as .pvd file.........")

    ua_.assign(ua)
    Ta_.assign(Ta)
    uo_.assign(uo)
    To_.assign(To)
    ua_sol_.assign(ua_sol)

    t += Dt
    iter +=1

print("Local time at the end of simulation:",time.strftime('%H:%M:%S', time.localtime()))