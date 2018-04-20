import pandas as pd
from AFNSGlobal.fx_functions import *
from AFNSGlobal.kalman_filter_functions import parameter_matrix_conv

df_fc_factors = pd.read_pickle("fc_factors.pickle")
# same_date = df_fc_factors.index.get_level_values(0) == df_fc_factors.index.get_level_values(1)
# df_fc_factors = df_fc_factors.loc[same_date]
# df_fc_factors.sort_index(inplace=True)
# df_fc_factors.reset_index(drop=True, level=1, inplace=True)
df_fc_parameters = pd.read_pickle("fc_parameters.pickle")
df_fx = pd.read_pickle("fx_rates.pickle")

# date = "2017-02-28"

fc_horizon = [1, 3, 6, 12]
column_names = ["current", "1mfc", "3mfc", "6mfc", "12mfc", "1mact", "3mact", "6mact", "12mact"]
column_names2 = ["eqpt1", "eqpt2"]
list_fc = []
list_fc_date = []
list_eq = []

for row in df_fc_factors.itertuples():
    fc_date, level, slope_d, curv_d, slope_f, curv_f = row
    factors_d = np.array([level, slope_d, curv_d]).reshape((3, 1))
    factors_f = np.array([level, slope_f, curv_f]).reshape((3, 1))
    params = np.array(df_fc_parameters.loc[fc_date, :])
    sigma, theta_p, kappa_p, vlambda = parameter_matrix_conv(2, params[:-1])
    gamma1_d = calc_gamma1(sigma[0], vlambda[0], kappa_p[0])
    gamma0_d = calc_gamma0(sigma[0], kappa_p[0], theta_p[0])
    cap_gamma_d = calc_cap_gamma(gamma0_d, gamma1_d, factors_d)
    gamma1_f = calc_gamma1(sigma[1], vlambda[1], kappa_p[1])
    gamma0_f = calc_gamma0(sigma[1], kappa_p[1], theta_p[1])
    cap_gamma_f = calc_cap_gamma(gamma0_f, gamma1_f, factors_f)
    rd_t = factors_d[:2].sum()
    rf_t = factors_f[:2].sum()
    #######
    eq = [rd_t - rf_t, float(cap_gamma_d.T @ (cap_gamma_d - cap_gamma_f))]
    list_eq.append(dict(zip(column_names2, eq)))
    ######
    s_t = df_fx.loc[fc_date, "GBPEUR"] ##### CHANGE CURRENCY HERE
    s_next = [next_fx_rate(cap_gamma_d, cap_gamma_f, rd_t, rf_t, s_t, k / 12) for k in fc_horizon]
    marketrates = [df_fx.shift(-k).loc[fc_date, "GBPEUR"] for k in fc_horizon] #########CHANGE CURRENCY HERE
    data = np.concatenate([[s_t], s_next, marketrates])
    list_fc.append(dict(zip(column_names, data)))
    list_fc_date.append(fc_date)

df_fc_results = pd.DataFrame(list_fc, index=list_fc_date)
df_fc_eq = pd.DataFrame(list_eq, index=list_fc_date)