from AFNSGlobal.kalman_filter_functions import *
from scipy.optimize import minimize
from AFNSGlobal.fitted_yields_functions import *
import pandas as pd
import numpy as np
from pyswarm import pso
import time

start_time = time.time()
# import rates
rates_eur = pd.read_pickle("pickle_bootstrapped_eur.pickle")
rates_usd = pd.read_pickle("pickle_bootstrapped_usd.pickle")

drop_list = [6, 7, 9, 10, 11, 12, 14, 16, 18]
rates_usd.drop(rates_usd.columns[drop_list], axis=1, inplace=True)
rates_usd.drop(rates_usd.index[:100], inplace=True)
rates_eur.drop(rates_eur.columns[drop_list], axis=1, inplace=True)
rates_eur.drop(rates_eur.index[:100], inplace=True)
rates_dict = {"usd": rates_usd, "eur": rates_eur}
tenors = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 5, 10, 15, 25])
delta_t = 1 / 12

### MINIMIZATION
# bounds
b_sigma = (0.01, 0.1)
b_theta_p = (-0.07, 0.07)
b_kappa_p = (0.1, 1)
b_lambda = (0.01, 1)
b_sigma_obs = (0.0000001, 0.1)

lbnds = 3 * [b_sigma[0]] + 3 * [b_theta_p[0]] + 3 * [b_kappa_p[0]] + [b_lambda[0]] + 3 * [b_sigma_obs[0]] + 2 * [
    b_sigma[0]] + 2 * [b_theta_p[0]] + 2 * [b_kappa_p[0]] + [b_lambda[0]]
ubnds = 3 * [b_sigma[1]] + 3 * [b_theta_p[1]] + 3 * [b_kappa_p[1]] + [b_lambda[1]] + 3 * [b_sigma_obs[1]] + 2 * [
    b_sigma[1]] + 2 * [b_theta_p[1]] + 2 * [b_kappa_p[1]] + [b_lambda[1]]

# other arguments to transfer to kalman_afns
other_args = (delta_t, tenors, rates_dict, False)

# Number of iterations for the minimizer
iterations = 20
result_columns = ["LLH", "Level G", "Slope D", "Curvature D", "Slope F", "Curvature F"]
parameter_columns = ["Sigma11G", "Sigma22D", "Sigma33D", "ThetaP1G", "ThetaP2D", "ThetaP3D", "KappaP11G", "KappaP22D",
                     "KappaP33D", "LambdaD",
                     "RSigmaST", "RSigmaMT", "RSigmaLT", "Sigma22F", "Sigma33F", "ThetaP2F",
                     "ThetaP3F", "KappaP22F",
                     "KappaP33F", "LambdaF"]
llh_best = np.inf
list_parameters = []
list_factors = []
list_llh = []

for i in range(iterations):
    # res = minimize(kalman_afns, initial_guess, args=other_args, method="SLSQP", bounds=bnds, constraints=cons,
    #                options=opt)
    res, fopt = pso(kalman_afns, args=other_args, lb=lbnds, ub=ubnds, maxiter=50, debug=False, swarmsize=100, minstep=1e-3)
    llh, df_factor_ts = kalman_afns(res, delta_t, tenors, rates_dict, True)
    df_parameters = pd.DataFrame(np.reshape(res, (1, 20)), columns=parameter_columns, index=[i])
    df_parameters = df_parameters.assign(loglh=[llh])
    if llh < llh_best:
        # df_factors_ts = df_factors_ts.assign(it=[i]*df_factors_ts.shape[0])
        llh_best = llh
        print("Best: %s" % llh_best)
        df_factor_results = df_factor_ts
        df_parameters_best = df_parameters
    list_parameters.append(df_parameters_best)
    list_factors.append(df_factor_ts)
    list_llh.append(llh)
    print((i + 1) / iterations * 100, "%")
    print("---Elapsed time: %s seconds ---" % (time.time() - start_time))
# save with timestamp
# time = str(time())
df_factor_results.to_pickle("factors.pickle")
df_parameters_best.to_pickle("parameters.pickle")

print(llh_best)

df_factor_results.to_excel("results.xlsx", sheet_name="Factors", index=True)
df_parameters_best.to_excel("parameters.xlsx", sheet_name="Parameters", index=True)

print("--- %s seconds ---" % (time.time() - start_time))
