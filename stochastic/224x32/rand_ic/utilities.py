import numpy as np

def acf(x,n_lags):
    '''manualy compute, non partial'''

    mean = np.mean(x)
    var = np.var(x)
    xp = x - mean
    corr = [1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in np.arange(n_lags)]

    return np.array(corr)

def auto_reg_gen(acf_1, t_steps):
    """generates AR(1) process starting from a random number ~ N(0,1)
    :arg acf_1: the ACF with lag 1
    :arg t_steps: generate AR(1) for t_steps number of time-steps
    """
    phi = acf_1 # autocorrelation for lag 1
    sigma = np.sqrt(1 - phi**2) # std. dev. of additive noise
    y_arr = np.zeros(t_steps)
    y_arr[0] = np.random.randn()
    for i in range(t_steps-1):
        y_arr[i+1] = phi*y_arr[i] + sigma*np.random.randn()
    return y_arr

def OU_mat(n_tsteps, n_eofs, acf1_data):
    M = np.zeros((n_tsteps, n_eofs)) # matrix that stores the OU generated noise
    phi = acf1_data[:n_eofs] # autocorrelation for lag 1 array for n_eofs
    sigma = np.sqrt(1 - phi**2) # std. dev. of additive noise
    M[0,:] = np.random.normal(size=M[0,:].shape)
    for i in range(n_tsteps -1):
        M[i+1, :] = phi*M[i,:] + sigma*np.random.normal(size= sigma.shape)
    return M
