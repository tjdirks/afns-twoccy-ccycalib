# This forecasts for a 3,6 and 12 months horizon. Training window is 5 years

from AFNSGlobal.kalman_filter_functions import *
from scipy.optimize import minimize
from AFNSGlobal.fitted_yields_functions import *
from AFNSGlobal.fx_functions import *
import pandas as pd
import numpy as np
from pyswarm import pso
import time

start_time = time.time()

# data import and selection
rates_usd = pd.read_pickle("pickle_bootstrapped_usd.pickle")
rates_eur = pd.read_pickle("pickle_bootstrapped_eur.pickle")
rates_jpy = pd.read_pickle("pickle_bootstrapped_jpy.pickle")
rates_gbp = pd.read_pickle("pickle_bootstrapped_gbp.pickle")
drop_list = [6, 7, 9, 10, 11, 12, 14, 16, 18]
rates_usd.drop(rates_usd.columns[drop_list], axis=1, inplace=True)
rates_eur.drop(rates_eur.columns[drop_list], axis=1, inplace=True)
rates_jpy.drop(rates_jpy.columns[drop_list], axis=1, inplace=True)
rates_gbp.drop(rates_gbp.columns[drop_list], axis=1, inplace=True)
rates_dict = {"usd": rates_usd, "eur": rates_eur, "jpy": rates_jpy, "gbp": rates_gbp}
tenors = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 5, 10, 15, 25])
df_fx = pd.read_pickle("fx_rates.pickle")

df_fx.loc[:, "USDEUR_1MCh"] = -df_fx.loc[:, "USDEUR"].diff(-1) / df_fx.loc[:, "USDEUR"]
df_fx.loc[:, "USDEUR_3MCh"] = -df_fx.loc[:, "USDEUR"].diff(-3) / df_fx.loc[:, "USDEUR"]
df_fx.loc[:, "USDEUR_6MCh"] = -df_fx.loc[:, "USDEUR"].diff(-6) / df_fx.loc[:, "USDEUR"]
df_fx.loc[:, "USDEUR_12MCh"] = -df_fx.loc[:, "USDEUR"].diff(-12) / df_fx.loc[:, "USDEUR"]
df_fx.loc[:, "GBPEUR_1MCh"] = -df_fx.loc[:, "GBPEUR"].diff(-1) / df_fx.loc[:, "GBPEUR"]
df_fx.loc[:, "GBPEUR_3MCh"] = -df_fx.loc[:, "GBPEUR"].diff(-3) / df_fx.loc[:, "GBPEUR"]
df_fx.loc[:, "GBPEUR_6MCh"] = -df_fx.loc[:, "GBPEUR"].diff(-6) / df_fx.loc[:, "GBPEUR"]
df_fx.loc[:, "GBPEUR_12MCh"] = -df_fx.loc[:, "GBPEUR"].diff(-12) / df_fx.loc[:, "GBPEUR"]
df_fx.loc[:, "JPYEUR_1MCh"] = -df_fx.loc[:, "JPYEUR"].diff(-1) / df_fx.loc[:, "JPYEUR"]
df_fx.loc[:, "JPYEUR_3MCh"] = -df_fx.loc[:, "JPYEUR"].diff(-3) / df_fx.loc[:, "JPYEUR"]
df_fx.loc[:, "JPYEUR_6MCh"] = -df_fx.loc[:, "JPYEUR"].diff(-6) / df_fx.loc[:, "JPYEUR"]
df_fx.loc[:, "JPYEUR_12MCh"] = -df_fx.loc[:, "JPYEUR"].diff(-12) / df_fx.loc[:, "JPYEUR"]

rates_dict = {"usd": rates_usd, "eur": rates_eur, "jpy": rates_jpy, "gbp": rates_gbp,
              "fxusdeur": df_fx.loc[:, ['USDEUR_1MCh', 'USDEUR_3MCh', 'USDEUR_6MCh','USDEUR_12MCh']],
              "fxgbpeur": df_fx.loc[:, ['GBPEUR_1MCh', 'GBPEUR_3MCh', 'GBPEUR_6MCh','GBPEUR_12MCh']],
              "fxjpyeur": df_fx.loc[:, ['JPYEUR_1MCh', 'JPYEUR_3MCh', 'JPYEUR_6MCh','JPYEUR_12MCh']]
              }

#
iterations = 3
pso_maxiter = 150
pso_minstep = 1e-6
n_swarm = 200
training_window = 5  # numbers of years on which estimation is run
sample_freq = 12
forecasting_freq = 3  # forecasting frequency in months
delta_t = 1 / 12  # timedelta between observations

# MINIMIZATION bounds
b_sigma = (0.01, 0.1)
b_theta_p = (-0.07, 0.07)
b_kappa_p = (0.1, 1)
b_lambda = (0.01, 1)
b_sigma_obs = (0.0000001, 0.1)

lbnds = 3 * [b_sigma[0]] + 3 * [b_theta_p[0]] + 3 * [b_kappa_p[0]] + [b_lambda[0]] + 3 * [b_sigma_obs[0]] + 2 * [
    b_sigma[0]] + 2 * [b_theta_p[0]] + 2 * [b_kappa_p[0]] + [b_lambda[0]]
ubnds = 3 * [b_sigma[1]] + 3 * [b_theta_p[1]] + 3 * [b_kappa_p[1]] + [b_lambda[1]] + 3 * [b_sigma_obs[1]] + 2 * [
    b_sigma[1]] + 2 * [b_theta_p[1]] + 2 * [b_kappa_p[1]] + [b_lambda[1]]

# end_training_window = rates_dict["eur"].index[0]+pd.DateOffset(years=training_window)

list_fc_dates = []
last_date = rates_dict["eur"].iloc[-13, :].name
i = 1
while rates_dict["eur"][:last_date].shape[0] > training_window * sample_freq:
    list_fc_dates.append(last_date)
    last_date = rates_dict["eur"].iloc[rates_dict["eur"].index.get_loc(last_date) - forecasting_freq, :].name
    i = i + 1

list_parameters_best = []
list_factors_best = []

for fc_date in list_fc_dates[::-1]:
    begin_estimation_window = fc_date + pd.DateOffset(months=-sample_freq * training_window + 1)
    begin_estimation_window = None
    print("Estimation Window: {} - {}".format(begin_estimation_window, fc_date))
    result_columns = ["LLH", "Level G", "Slope D", "Curvature D", "Slope F", "Curvature F"]
    parameter_columns = ["Sigma11G", "Sigma22D", "Sigma33D", "ThetaP1G", "ThetaP2D", "ThetaP3D", "KappaP11G",
                         "KappaP22D",
                         "KappaP33D", "LambdaD",
                         "RSigmaST", "RSigmaMT", "RSigmaLT", "Sigma22F", "Sigma33F", "ThetaP2F",
                         "ThetaP3F", "KappaP22F",
                         "KappaP33F", "LambdaF"]
    llh_best = np.inf

    curr_rates_dict = {"eur": rates_dict["eur"].truncate(before=begin_estimation_window, after=fc_date),
                       "gbp": rates_dict["gbp"].truncate(before=begin_estimation_window, after=fc_date),
                       "fxgbpeur": rates_dict["fxgbpeur"].truncate(before=begin_estimation_window, after=fc_date)}

    # other arguments to transfer to kalman_afns
    other_args = (delta_t, tenors, curr_rates_dict, False)

    for i in range(iterations):
        res, fopt = pso(kalman_afns, args=other_args, lb=lbnds, ub=ubnds, maxiter=pso_maxiter, debug=False,
                        swarmsize=n_swarm,
                        minstep=pso_minstep)
        llh, df_factor_ts = kalman_afns(res, delta_t, tenors, curr_rates_dict, True)
        df_parameters = pd.DataFrame(np.reshape(res, (1, 20)), columns=parameter_columns, index=[fc_date])
        df_parameters = df_parameters.assign(loglh=[llh])
        if llh < llh_best:
            df_factor_ts["fcdate"] = fc_date
            df_factor_ts.set_index('fcdate', append=True, inplace=True)
            df_factors_best = df_factor_ts
            df_parameters_best = df_parameters
            llh_best = llh
        print((i + 1) / iterations * 100, "%")
        print("---Elapsed time: %s seconds ---" % (time.time() - start_time))
    list_factors_best.append(df_factors_best)
    list_parameters_best.append(df_parameters_best)
    print(llh_best)
df_fc_factors = pd.concat(list_factors_best)
df_fc_parameters = pd.concat(list_parameters_best)
df_fc_factors.to_pickle("fc_factors.pickle")
df_fc_parameters.to_pickle("fc_parameters.pickle")

# slice for specific fc_date df_fc_factors.loc[(slice(None),"2018-01-31"),:]
