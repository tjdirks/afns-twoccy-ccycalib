# defines functions used in AFNS_forecasting
import numpy as np

def calc_gamma1(sigma, vlambda, kappa_p):
    # sigma_priv and kappa_priv need to be 3x3 matrices, lambda_priv is a single value from which Kappa Q is built
    kappa_q = np.zeros((3,3))
    kappa_q[1,1] = vlambda
    kappa_q[1,2] = -vlambda
    kappa_q[2,2] = vlambda
    gamma1 = np.linalg.inv(sigma) @ (kappa_q - kappa_p)
    return gamma1

def calc_gamma0(sigma, kappa_p, theta_p):
    return (np.linalg.inv(sigma) @ kappa_p @ theta_p).reshape((3,1))

def calc_cap_gamma(gamma0, gamma1, factors):
    return gamma0 + gamma1 @ factors

def next_fx_rate(cap_gamma_d, cap_gamma_f, rd_t, rf_t, s_t, delta_t):
    psi0 = rd_t-rf_t + cap_gamma_d.T @(cap_gamma_d-cap_gamma_f)
    change = psi0*delta_t
    s_tnext = s_t*(1+change)
    return np.float_(s_tnext)
