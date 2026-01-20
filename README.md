# Numerical simulation of a structure-preserving idealized stochastic climate model
This repository contains code for the simulation of a structure-preserving idealized stochastic climate model. 

## The climate model
The climate model can be described by 

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign%2A%7D%0A%5Ctext%7BAtmosphere%7D%3A%20%5C%20%26%5Cmathrm%7Bd%7D%20%5Cmathbf%7Bu%7D%5Ea%20%2B%20%28%5Cmathbf%7Bu%7D%5Ea%20%5Cmathrm%7Bd%7D%20t%20%2B%20%5Csum_i%20%5Cboldsymbol%7B%5Cxi%7D_i%20%5Ccirc%20%5Cmathrm%7Bd%7D%20W%5Ei_t%29%5Ccdot%20%5Cnabla%20%5Cmathbf%7Bu%7D%5Ea%20%2B%20%5Cfrac%7B1%7D%7BRo%5Ea%7D%20%5Chat%7B%5Cmathbf%7Bz%7D%7D%20%5Ctimes%20%5Cmathbf%7Bu%7D%5Ea%5Cmathrm%7Bd%7D%20t%20%5C%5C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%26%5Cqquad%20%5Cqquad%20%2B%20%5Csum_i%20%28u_1%5Ea%20%5Cnabla%20%5Cxi_%7Bi%2C1%7D%20%2B%20u_2%5Ea%20%5Cnabla%20%5Cxi_%7Bi%2C2%7D%20%29%5Ccirc%20%5Cmathrm%7Bd%7D%20W%5Ei_t%20%3D%20%28-%5Cfrac%7B1%7D%7BC%5Ea%7D%20%5Cnabla%20%5Ctheta%20%2B%20%5Cnu%5Ea%20%5CDelta%20%5Cmathbf%7Bu%7D%5Ea%29%20%5Cmathrm%7Bd%7D%20t%2C%20%5C%5C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%26%5Cmathrm%7Bd%7D%20%5Ctheta%5Ea%20%2B%20%5Cnabla%5Ccdot%20%28%5Ctheta%5Ea%20%5Cmathbf%7Bu%7D%5Ea%29%5Cmathrm%7Bd%7D%20t%20%2B%20%5Csum_i%20%28%5Cboldsymbol%7B%5Cxi%7D_i%20%5Ccirc%20%5Cmathrm%7Bd%7D%20W%5Ei_t%29%20%5Ccdot%20%5Cnabla%20%5Ctheta%5Ea%20%3D%20%28%5Cgamma%28%5Ctheta%5Ea%20-%20%5Ctheta%5Eo%29%20%2B%20%5Cnu%5Ea%20%5CDelta%20%5Ctheta%20%29%5Cmathrm%7Bd%7D%20t%2C%5C%5C%0A%5Ctext%7BOcean%7D%3A%20%5C%20%26%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bu%7D%5Eo%7D%7B%5Cpartial%20t%7D%20%2B%20%28%5Cmathbf%7Bu%7D%5Eo%5Ccdot%20%5Cnabla%29%5Cmathbf%7Bu%7D%5Eo%20%2B%20%5Cfrac%7B1%7D%7BRo%5Eo%7D%20%5Chat%7B%5Cmathbf%7Bz%7D%7D%20%5Ctimes%20%5Cmathbf%7Bu%7D%5Eo%20%2B%20%5Cfrac%7B1%7D%7BRo%5Eo%7D%20%5Cnabla%20p%5Ea%20%3D%20%5Csigma%28%5Cmathbf%7Bu%7D%5Eo%20-%20%5Cmathbb%7BE%7D%5Cmathbf%7Bu%7D_%7Bsol%7D%5Ea%29%20%2B%20%5Cnu%5Eo%20%5CDelta%20%5Cmathbf%7Bu%7D%5Eo%2C%5C%5C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%26%20%5Cnabla%20%5Ccdot%20%5Cmathbf%7Bu%7D%5Eo%20%3D%200%2C%5C%5C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%26%5Cfrac%7B%5Cpartial%20%5Ctheta%5Eo%7D%7B%5Cpartial%20t%7D%20%2B%20%5Cmathbf%7Bu%7D%5Eo%20%5Ccdot%20%5Cnabla%20%5Ctheta%5Eo%20%3D%20%5Ceta%5Eo%20%5CDelta%20%5Ctheta%5Eo%2C%5C%5C%0A%5Cend%7Balign%2A%7D" alt="Stochastic OA model equations">
</p>

<!-- $$
\begin{align*}
\text{Atmosphere}: \ &\mathrm{d} \mathbf{u}^a + (\mathbf{u}^a \mathrm{d} t + \sum_i \boldsymbol{\xi}_i \circ \mathrm{d} W^i_t)\cdot \nabla \mathbf{u}^a + \frac{1}{Ro^a} \hat{\mathbf{z}} \times \mathbf{u}^a\mathrm{d} t \\
                &\qquad \qquad + \sum_i (u_1^a \nabla \xi_{i,1} + u_2^a \nabla \xi_{i,2} )\circ \mathrm{d} W^i_t = (-\frac{1}{C^a} \nabla \theta + \nu^a \Delta \mathbf{u}^a) \mathrm{d} t, \\
        &\mathrm{d} \theta^a + \nabla\cdot (\theta^a \mathbf{u}^a)\mathrm{d} t + \sum_i (\boldsymbol{\xi}_i \circ \mathrm{d} W^i_t) \cdot \nabla \theta^a = (\gamma(\theta^a - \theta^o) + \nu^a \Delta \theta )\mathrm{d} t,\\
\text{Ocean}: \ &\frac{\partial \mathbf{u}^o}{\partial t} + (\mathbf{u}^o\cdot \nabla)\mathbf{u}^o + \frac{1}{Ro^o} \hat{\mathbf{z}} \times \mathbf{u}^o + \frac{1}{Ro^o} \nabla p^a = \sigma(\mathbf{u}^o - \mathbb{E}\mathbf{u}_{sol}^a) + \nu^o \Delta \mathbf{u}^o,\\
    & \nabla \cdot \mathbf{u}^o = 0,\\
    &\frac{\partial \theta^o}{\partial t} + \mathbf{u}^o \cdot \nabla \theta^o = \eta^o \Delta \theta^o,
\end{align*}
$$ -->
where the vector variable $\mathbf{{u}}$ and the scalar variables $\theta$ and  $p$ (with superscripts for the atmosphere and ocean components) denote the velocity, temperature, and pressure fields, respectively. $W_t^i$ are independent Brownian motions and, $u_j^a$ and $\xi_{i,j}$ denote the $j$ th components of atmosphere velocity $\mathbf{u}^a$ and the spatial correlation vectors $\boldsymbol{\xi}_i$, respectively.

We discretize the climate model equations using finite element method (FEM) and solve the discrete equations on [Firedrake](https://www.firedrakeproject.org/) open-source package. The simulation results are visualized using [Paraview](https://www.paraview.org/). 
## Model calibration
One of the first steps in the numerical simulation of stochastic climate model is the model calibration. This involves estimation of the unknown correlation vectors $\boldsymbol{\xi}_i$ representing the unresolved transport dynamics. In our work, we use data from high-resolution simulation of the deterministic version of the climate model to determine $\boldsymbol{\xi}_i$.

The deterministic climate model equations are 
$$
\begin{align*}
    \text{Atmosphere}: \ &\frac{\partial \mathbf{u}^a}{\partial t} + (\mathbf{u}^a\cdot \nabla)\mathbf{u}^a + \frac{1}{Ro^a} \hat{\mathbf{z}} \times \mathbf{u}^a + \frac{1}{C^a} \nabla \theta^a = {\nu}^a \Delta \mathbf{u}^a, \\
        &\frac{\partial \theta^a}{\partial t} + \nabla \cdot (\mathbf{u}^a \theta^a) = \gamma(\theta^a - \theta^o) + \eta^a \Delta \theta^a,\\
    \text{Ocean}: \ &\frac{\partial \mathbf{u}^o}{\partial t} + (\mathbf{u}^o\cdot \nabla)\mathbf{u}^o + \frac{1}{Ro^o} \hat{\mathbf{z}} \times \mathbf{u}^o + \frac{1}{Ro^o} \nabla p^o = \sigma(\mathbf{u}^o - \mathbf{u}_{sol}^a) + \nu^o \Delta \mathbf{u}^o,\\
    & \nabla \cdot \mathbf{u}^o = 0,\\
    &\frac{\partial \theta^o}{\partial t} + \mathbf{u}^o \cdot \nabla \theta^o = \eta^o \Delta \theta^o.
\end{align*}
$$


## This repository
It contains codes for solving the deterministic and stochastic climate model equations as **python** scripts. These are designed to be run on **Firedrake** software. The folders `deterministic` and `stochastic` contain the code for deterministic and stochastic models, respectively.

Additionally, it contains *jupyter notebooks* ( in `.deterministic/224x32/xi_calculation_vis_btb_config1`) which we used to estimate  $\boldsymbol{\xi}_i$.

We used **numpy** and **matplotlib** libraries for further analyzing the simulation data and plotting results. **Jupyter notebooks** used for this purpose are also contained in this repository. 

## Simulation results
We show some of the main simulation results here. The model equations are solved on a rectangular domain with periodic boundary conditions in the $x$ direction and free-slip boundary conditions in the $y$ direction. 
### Deterministic climate model
<figure>
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_atm_temp_t0.svg" alt="Atmospheric temperature" width="85%">
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_atm_vel_t0.svg" alt="Atmospheric velocity" width="85%">
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_ocean_temp_t0.svg" alt="Ocean temperature" width="85%">
  <figcaption>Initial atmospheric temperature (top), atmospheric velocity (middle), and ocean temperature (bottom) fields at t=0. The ocean velocity is set to zero at t=0.</figcaption>
</figure>

<figure>
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_atm_temp_t25.svg" alt="Atmospheric temperature" width="85%">
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_atm_vel_t25.svg" alt="Atmospheric velocity" width="85%">
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_ocean_temp_t25.svg" alt="Ocean temperature" width="85%">
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_ocean_vel_t25.svg" alt="Ocean temperature" width="85%">
  <figcaption>Atmospheric (top two) and oceanic (bottom two) velocity and temperature fields at t = 25 for the deterministic climate model simulation on the 896x128 grid.</figcaption>
</figure>

<figure>
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_atm_vort_t25.svg" alt="Atmospheric vorticity" width="85%">
  <img src="deterministic/896x128/plots/thesis_plots/CM_high_res_ocean_vort_t25.svg" alt="ocean velocity" width="85%">
  <figcaption>Atmospheric vorticity (top) and ocean vorticity (bottom) fields at t=25 for the deterministic model simulation on the 896x128 grid.</figcaption>
</figure>

### Stochastic climate model
<figure>
  <img src="stochastic/224x32/plots/thesis_plots/vort_stoch_p1_same_99_var_ic_at_t35.svg" alt="Atm. vort. particle 1" width="85%">
  <img src="stochastic/224x32/plots/thesis_plots/vort_stoch_p2_same_99_var_ic_at_t35.svg" alt="Atm. vort. particle 2" width="85%">
  <img src="stochastic/224x32/plots/thesis_plots/vort_stoch_p3_same_99_var_ic_at_t35.svg" alt="Atm. vort. particle 3" width="85%">
  <img src="stochastic/224x32/plots/thesis_plots/vort_det_same_ic_at_t35.svg" alt="atm. vort. det." width="85%">
  <img src="stochastic/224x32/plots/thesis_plots/vort_truth_at_t35.svg" alt="atm. vort. truth" width="85%">
  <figcaption>Atmospheric vorticity fields from three independent realizations of the SPDE on the 224x32 grid at t = 35 (top three plots), compared with the deterministic model result (fourth plot) and the coarse-grained high-resolution solution (last plot).</figcaption>
</figure>

### Uncertainty quantification
<figure>
  <img src="stochastic/224x32/plots/ou_same_ic/cm_res_32_atm_ux_var_99_part_50_t25_onwards.svg" alt="UQ" width="85%">
  <figcaption>Evolution of atmospheric velocity (x component) at six observation points on the grid over time (t=25 to t=45). The solution from the stochastic model is compared with the coarse-grained high-resolution solution (truth) and the deterministic model solution.</figcaption>
</figure>
<figure>
  <img src="stochastic/224x32/plots/ou_same_ic/spread_rmse_res_32_atm_ux_var_99_part_50_t25_onwards.svg" alt="UQ" width="85%">
  <figcaption>Ensemble RMSE and spread of velocity (x component) at six observation points in the grid over time.</figcaption>
</figure>

### Stochastic versus deterministic ensemble
<figure>
  <img src="stochastic/224x32/plots/rand_ic/uq_ux_det_v_stoch_50_part_var_99_ensem.svg" alt="UQ" width="85%">
  <figcaption>Comparison of atmospheric velocity (x component) generated by the stochastic ensemble and the deterministic ensemble. Both ensembles start with same initial conditions (perturbed velocity fields at t=25). The stochastic model is parameterized using SALT.</figcaption>
</figure>
<figure>
  <img src="stochastic/224x32/plots/rand_ic/crps_ux_det_v_stoch_ensem_50_part_var_99_ensem.svg" alt="UQ" width="85%">
  <figcaption>CRPS plots comparing the performance of the stochastic ensemble and the deterministic ensemble.</figcaption>
</figure>