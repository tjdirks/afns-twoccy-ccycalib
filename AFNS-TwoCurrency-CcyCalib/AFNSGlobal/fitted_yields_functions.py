import numpy as np
from scipy.linalg import expm
import pandas as pd

def factor_loadings_single_curr(tenors_priv, lambda_priv):
    ntenors = len(tenors_priv)
    x_loadings = np.zeros((3, ntenors))
    x_loadings[0, :] = np.ones(ntenors)
    x_loadings[1, :] = (np.ones(ntenors) - np.exp(-lambda_priv * tenors_priv)) / (lambda_priv * tenors_priv)
    x_loadings[2, :] = (np.ones(ntenors) - np.exp(-lambda_priv * tenors_priv)) / (lambda_priv * tenors_priv) - np.exp(
        -lambda_priv * tenors_priv)
    return x_loadings.T

# Yield-Adjustment Term C/(T-t) - Double checked with EXCEL
def c_t_T(sigma_y, lambda_y, delta_t_y):
    c_a = sigma_y[0, 0] ** 2
    c_b = sigma_y[1, 1] ** 2
    c_c = sigma_y[2, 2] ** 2
    c_aux_a = delta_t_y ** 2 / 6
    c_aux_b = (1 / (2 * lambda_y ** 2)) - (1 / lambda_y ** 3) * (
            1 - np.exp(-lambda_y * delta_t_y)) / delta_t_y + 1 / (
                      4 * lambda_y ** 3) * (1 - np.exp(-2 * lambda_y * delta_t_y)) / delta_t_y
    c_aux_c = (1 / (2 * lambda_y ** 2)) + (1 / lambda_y ** 2) * np.exp(-lambda_y * delta_t_y) - 1 / (
            4 * lambda_y) * delta_t_y * np.exp(-2 * lambda_y * delta_t_y) - 3 / (4 * lambda_y ** 2) * np.exp(
        -2 * lambda_y * delta_t_y) - (2 / lambda_y ** 3) * (1 - np.exp(-lambda_y * delta_t_y)) / delta_t_y + 5 / (
                      8 * lambda_y ** 3) * (1 - np.exp(-2 * lambda_y * delta_t_y)) / delta_t_y
    c = c_a * c_aux_a + c_b * c_aux_b + c_c * c_aux_c
    return c


def ya_matrix_single_curr(tenors_priv, sigma_priv, lambda_priv):
    ntenors = len(tenors_priv)
    yield_adj_matrix = np.zeros(ntenors)
    for i in range(ntenors):
        yield_adj_matrix[i] = c_t_T(sigma_priv, lambda_priv, tenors_priv[i])
    return yield_adj_matrix

def calc_fitted_yield(tenors_priv, factors_priv, lambda_priv, sigma_priv):
    factor_loadings_matrix = factor_loadings_single_curr(tenors_priv, lambda_priv)
    yield_adjustment_matrix = ya_matrix_single_curr(tenors_priv, sigma_priv, lambda_priv)
    yt = factor_loadings_matrix @ factors_priv - yield_adjustment_matrix
    return yt

def forecast_yields(tenors, factors, kappa_p, theta_p, sigma, vlambda, fc_horizon):
    list_e_xt = []
    for dt in fc_horizon:
        dt = dt/12
        phi_0 = (np.eye(3) - expm(-kappa_p * dt)) @ theta_p
        phi_1 = expm(-kappa_p * dt)
        e_xt = phi_0.reshape((3,1)) + phi_1@factors
        list_e_xt.append(e_xt)
    factor_loadings_matrix = factor_loadings_single_curr(tenors, vlambda)
    yield_adjustment_matrix = ya_matrix_single_curr(tenors, sigma, vlambda)
    list_yt = []
    for x in list_e_xt:
        yt = factor_loadings_matrix @ x - yield_adjustment_matrix.reshape((len(tenors),1))
        list_yt.append(yt.reshape((len(tenors))))
    return np.stack(list_yt)

