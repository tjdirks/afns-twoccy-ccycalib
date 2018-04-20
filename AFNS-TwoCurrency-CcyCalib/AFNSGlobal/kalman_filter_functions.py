import numpy as np
import pandas as pd
from scipy.linalg import expm
from random import uniform
from AFNSGlobal.fx_functions import calc_gamma0, calc_gamma1


def parameter_matrix_conv(n_currencies, a_parameters_priv):
    kappa_p = []
    theta_p = []
    sigma = []
    vlambda = []
    for i in range(n_currencies):
        kappa_p.append(np.zeros((3, 3)))
        theta_p.append(np.zeros(3))
        sigma.append(np.zeros((3, 3)))
        vlambda.append(np.zeros(1))
    # the part below has to be adapted to the number of currencies used
    # kappa_p
    np.fill_diagonal(kappa_p[0], a_parameters_priv[6:9])
    np.fill_diagonal(kappa_p[1], a_parameters_priv[[6, 17, 18]])
    kappa_p.append(np.zeros((1 + n_currencies * 2, 1 + n_currencies * 2)))
    np.fill_diagonal(kappa_p[n_currencies], a_parameters_priv[[6, 7, 8, 17, 18]])
    # theta_p
    theta_p[0] = a_parameters_priv[3:6]
    theta_p[1] = a_parameters_priv[[3, 15, 16]]
    theta_p.append(a_parameters_priv[[3, 4, 5, 15, 16]])
    # SIGMA
    np.fill_diagonal(sigma[0], a_parameters_priv[:3])
    np.fill_diagonal(sigma[1], a_parameters_priv[[0, 13, 14]])
    sigma.append(np.zeros((1 + n_currencies * 2, 1 + n_currencies * 2)))
    np.fill_diagonal(sigma[n_currencies], a_parameters_priv[[0, 1, 2, 13, 14]])
    # LAMBDA
    vlambda[0] = a_parameters_priv[9]
    vlambda[1] = a_parameters_priv[19]
    return sigma, theta_p, kappa_p, vlambda


# INITIAL GUESS (must fit constraints), creates new random initial values until they fit the constraint
def con_kappa_p_eig(param):
    kappa_p = parameter_matrix_conv(2, param)[2]
    eig1 = np.linalg.eig(kappa_p[0])[0]
    eig2 = np.linalg.eig(kappa_p[1])[0]
    eig1 = np.real(np.amin(eig1))
    eig2 = np.real(np.amin(eig2))
    return np.amin([eig1, eig2])


# INITIAL GUESS (must fit constraints), creates new random initial values  within the boundaries until they fit the constraint
def init_guess(boundaries_priv):
    x = False
    while x is False:
        initial_guess = np.array([uniform(*boundaries_priv[i]) for i in range(len(boundaries_priv))])
        x = con_kappa_p_eig(initial_guess) > 0
    return initial_guess


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


def ya_matrix(tenors_priv, sigma_priv, lambda_priv):
    ntenors = len(tenors_priv)
    n_currencies = 2
    yield_adj_matrix = []
    for n in range(n_currencies):
        for i in range(ntenors):
            yield_adj_matrix.append(c_t_T(sigma_priv[n], lambda_priv[n], tenors_priv[i]))
    return np.array(yield_adj_matrix)


def factor_loadings(tenors_priv, lambda_priv):
    n_tenors = len(tenors_priv)
    n_currencies = 2
    x_loadings = np.zeros((n_currencies * n_tenors, 1 + 2 * n_currencies))
    x_loadings[:, 0] = np.ones(n_currencies * n_tenors)
    x_loadings[:10, 1] = (np.ones(n_tenors) - np.exp(-lambda_priv[0] * tenors_priv)) / (lambda_priv[0] * tenors_priv)
    x_loadings[:10, 2] = (np.ones(n_tenors) - np.exp(-lambda_priv[0] * tenors_priv)) / (
            lambda_priv[0] * tenors_priv) - np.exp(
        -lambda_priv[0] * tenors_priv)
    x_loadings[10:20, 3] = (np.ones(n_tenors) - np.exp(-lambda_priv[1] * tenors_priv)) / (lambda_priv[1] * tenors_priv)
    x_loadings[10:20, 4] = (np.ones(n_tenors) - np.exp(-lambda_priv[1] * tenors_priv)) / (
            lambda_priv[1] * tenors_priv) - np.exp(
        -lambda_priv[1] * tenors_priv)
    return x_loadings

def create_sobs_matrix(parameters, ntenors):
    return_matrix = np.zeros((2 * ntenors, 2 * ntenors))
    sigma_obs_diag = np.array(
        5 * [parameters[0]] + 3 * [parameters[1]] + 2 * [parameters[2]] + 5 * [parameters[0]] + 3 * [
            parameters[1]] + 2 * [parameters[2]])
    np.fill_diagonal(return_matrix, sigma_obs_diag)
    return return_matrix


# the following functions extend the matrices for currency calib
def mod_factor_loadings(x_loadings, gamma0, gamma1, x):
    fl_fx = np.zeros((4, 5))
    fc_horizons = np.array([1/12, 3/12, 6/12, 1])
    x_d = x[:3].reshape((3, 1))
    x_f = x[[0, 3, 4]].reshape((3, 1))
    # part dA/dX
    psi21 = gamma1[0].T @ gamma0[0]
    psi21 = np.append(psi21, [0, 0])
    psi22 = gamma1[1].T @ gamma0[0]
    psi22 = np.insert(psi22, 1, [0, 0])
    psi23 = gamma1[0].T @ (gamma0[0] - gamma0[1])
    psi23 = np.append(psi23, [0, 0])
    psi24 = gamma1[0].T @ gamma1[0] @ x_d
    psi24 = np.append(psi24, [0, 0])
    psi25 = gamma1[0].T @ gamma1[0] @ x_d
    psi25 = np.append(psi25, [0, 0])
    psi26 = gamma1[0].T @ gamma1[1] @ x_f
    psi26 = np.append(psi26, [0, 0])
    psi27 = gamma1[1].T @ gamma1[0] @ x_d
    psi27 = np.insert(psi27, 1, [0, 0])
    psi2 = psi21 - psi22 + psi23 + psi24 + psi25 - psi26 - psi27
    fl_fx[0,:] = psi2 * fc_horizons[0]
    fl_fx[1, :] = psi2 * fc_horizons[1]
    fl_fx[2, :] = psi2 * fc_horizons[2]
    fl_fx[3, :] = psi2 * fc_horizons[3]
    mod_vec = np.concatenate((x_loadings, fl_fx))
    return mod_vec


def mod_yield_adj(ya_vector, gamma0, gamma1, x):
    ya_fx = np.array([1/12, 3/12, 6/12, 1]) #warning FC horizons are hardcoded
    x_d = x[:3].reshape((3, 1))
    x_f = x[[0, 3, 4]].reshape((3, 1))
    # part A
    psi11 = x_d[1] - x_f[1]
    psi12 = gamma0[0].T @ (gamma0[0] - gamma0[1])\
           + gamma0[0].T @ gamma1[0] @ x_d\
           - gamma0[0].T @ gamma1[1] @ x_f\
           + x_d.T @ gamma1[0].T @ (gamma0[0] - gamma0[1])\
           + x_d.T @ gamma1[0].T @ gamma1[0] @ x_d\
           - x_d.T @ gamma1[0].T @ gamma1[1] @ x_f
    psi1 = psi11 + psi12
    # part dA/dX
    psi21 = gamma1[0].T @ gamma0[0]
    psi21 = np.append(psi21, [0,0])
    psi22 = gamma1[1].T @ gamma0[0]
    psi22 = np.insert(psi22, 1, [0,0])
    psi23 = gamma1[0].T @ (gamma0[0] - gamma0[1])
    psi23 = np.append(psi23, [0,0])
    psi24 = gamma1[0].T @ gamma1[0] @ x_d
    psi24 = np.append(psi24, [0,0])
    psi25 = gamma1[0].T @ gamma1[0] @ x_d
    psi25 = np.append(psi25, [0, 0])
    psi26 = gamma1[0].T @ gamma1[1] @ x_f
    psi26 = np.append(psi26, [0,0])
    psi27 = gamma1[1].T @ gamma1[0] @ x_d
    psi27 = np.insert(psi27, 1, [0,0])
    psi2 = psi21 - psi22 + psi23 + psi24 + psi25 - psi26 - psi27
    ya_fx = (-psi1 + psi2 @ x) * ya_fx
    mod_vec = np.concatenate((ya_vector, ya_fx.flatten()))
    return mod_vec


def mod_sobs_matrix(sobs_matrix):  # extends the diagonal measurement error matrix by three values for fx with sobs1
    sobs_vector = np.diag(sobs_matrix)
    sobs_fx = np.zeros(4)
    sobs_fx.fill(sobs_vector[0])
    mod_matrix = np.concatenate((sobs_vector, sobs_fx))
    return np.diag(mod_matrix)


def kalman_afns(parameters, delta_t, tenors, rates, r_flag):  # parameters, delta_time, tenors, rates

    # same tenors for both currencies
    ntenors = len(tenors)

    # same tenors for both currencies
    ntenors = len(tenors)
    n_observations = rates["eur"].shape[0]
    n_currencies = 2
    # define parameters
    sigma, theta_p, kappa_p, vlambda = parameter_matrix_conv(n_currencies, parameters)
    sigma_obs = create_sobs_matrix(parameters[10:13], ntenors)
    sigma_obs = mod_sobs_matrix(sigma_obs)

    # create yield adjustment term for each maturity and currency
    yield_adj_matrix = ya_matrix(tenors, sigma, vlambda)

    # factor loadings
    factor_loadings_matrix = factor_loadings(tenors, vlambda)

    # transition - CORRECT
    phi_0 = (np.eye(1 + 2 * n_currencies) - expm(-kappa_p[n_currencies] * delta_t)) @ theta_p[n_currencies]
    phi_1 = expm(-kappa_p[n_currencies] * delta_t)

    # Gammas for FX, 0 domestic, 1 foreign
    gamma0 = [calc_gamma0(sigma[0], kappa_p[0], theta_p[0]), calc_gamma0(sigma[1], kappa_p[1], theta_p[1])]
    gamma1 = [calc_gamma1(sigma[0], vlambda[0], kappa_p[0]), calc_gamma1(sigma[1], vlambda[1], kappa_p[1])]

    # 1 initialize state vector with X=ThetaP, Sigma0 = ... (CDR, p. 10)

    x = theta_p[n_currencies]

    cap_sigma = np.diag(
        np.diag(sigma[n_currencies]) * np.diag(sigma[n_currencies]) / np.diag(2 * kappa_p[n_currencies]) * (
            np.diag(np.eye(1 + 2 * n_currencies) - expm(-2 * kappa_p[n_currencies] * delta_t))))

    loglikelihood = np.zeros(n_observations)
    list_factors = []

    for i in np.arange(n_observations):
        # 2 yt shape (10,), factor_loadings shape (10,3)
        if i == 0:
            # Q equals VAR(Zti..]/ModelVariance -> shape (10,10)
            factor_loadings_matrix_current = mod_factor_loadings(factor_loadings_matrix, gamma0, gamma1, x)
            yield_adj_matrix_current = mod_yield_adj(yield_adj_matrix, gamma0, gamma1, x)
            yt = factor_loadings_matrix_current @ x - yield_adj_matrix_current
            qt = factor_loadings_matrix_current @ cap_sigma @ factor_loadings_matrix_current.T + sigma_obs
            qt = (qt + qt.T) / 2
        else:
            factor_loadings_matrix_current = mod_factor_loadings(factor_loadings_matrix, gamma0, gamma1, x_predict)
            yield_adj_matrix_current = mod_yield_adj(yield_adj_matrix, gamma0, gamma1, x_predict)
            yt = factor_loadings_matrix_current @ x_predict - yield_adj_matrix_current
            qt = factor_loadings_matrix_current @ x_var_predict @ factor_loadings_matrix_current.T + sigma_obs
            qt = (qt + qt.T) / 2

        # shape v_error = (20,)
        rates_v = np.concatenate((rates["eur"].iloc[i, :], rates["gbp"].iloc[i, :], rates["fxgbpeur"].iloc[i, :]))
        v_error = rates_v - yt

        # 3 Update
        if i == 0:
            kalman_gain = cap_sigma @ factor_loadings_matrix_current.T @ np.linalg.inv(qt)
            x_update = x + kalman_gain @ v_error
            x_var_update = (np.eye(1 + 2 * n_currencies) - kalman_gain @ factor_loadings_matrix_current) @ cap_sigma
        else:
            kalman_gain = x_var_predict @ factor_loadings_matrix_current.T @ np.linalg.inv(qt)
            x_update = x_predict + kalman_gain @ v_error
            x_var_update = (np.eye(1 + 2 * n_currencies) - kalman_gain @ factor_loadings_matrix_current) @ x_var_predict

        # 4 - equal to nextxmean  and nextxvariance
        x_predict = phi_0 + phi_1 @ x_update
        x_var_predict = phi_1 @ x_var_update @ phi_1 + cap_sigma

        # 5
        llh = -0.5 * (n_currencies * ntenors) * np.log(2 * np.pi) - 0.5 * (
                    np.linalg.slogdet(qt)[1] + v_error @ np.linalg.inv(
                qt) @ v_error)
        loglikelihood[i] = llh
        # state variables
        if r_flag:
            dict_factors = {"Level G": x_update[0], "Slope D": x_update[1],
                            "Curvature D": x_update[2],
                            "Slope F": x_update[3], "Curvature F": x_update[4]}
            list_factors.append(pd.DataFrame(dict_factors, index=[i]))
    if r_flag:
        df_factors_ts = pd.concat(list_factors)
        df_factors_ts.index = rates["eur"].index
        df_factors_ts = df_factors_ts[["Level G", "Slope D", "Curvature D", "Slope F", "Curvature F"]]
        return (-sum(loglikelihood)), df_factors_ts
    else:
        return (-sum(loglikelihood))
