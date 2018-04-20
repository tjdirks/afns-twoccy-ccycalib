import pandas as pd
from sklearn.metrics import mean_squared_error
from AFNSGlobal.fx_functions import *
from AFNSGlobal.kalman_filter_functions import parameter_matrix_conv
from AFNSGlobal.fitted_yields_functions import forecast_yields
import time

df_fc_factors = pd.read_pickle("fc_factors.pickle")
# df_fc_factors.sort_index(inplace=True)
# same_date = df_fc_factors.index.get_level_values(0) == df_fc_factors.index.get_level_values(1)
# df_fc_factors = df_fc_factors.loc[same_date]
# df_fc_factors.sort_index(inplace=True)
df_fc_parameters = pd.read_pickle("fc_parameters.pickle")
df_fc_fx = pd.read_pickle("fx_rates.pickle")
rates_d = pd.read_pickle("C:/Users/tdirks/PycharmProjects/Thesis/Bootstrapping/pickle_bootstrapped_eur.pickle")
rates_f = pd.read_pickle("C:/Users/tdirks/PycharmProjects/Thesis/Bootstrapping/pickle_bootstrapped_gbp.pickle")

fc_horizon = [3, 6, 12]

tenors = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25])
tenors_col = ["1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y",
              "20Y", "25Y"]
tenors_col_fc = ["fc" + s for s in tenors_col]
tenors_col_act = ["act" + s for s in tenors_col]
list_fc_d = []
list_fc_f = []

for row in df_fc_factors.itertuples():
    fc_date, level, slope_d, curv_d, slope_f, curv_f = row
    factors_d = np.array([level, slope_d, curv_d]).reshape((3, 1))
    factors_f = np.array([level, slope_f, curv_f]).reshape((3, 1))
    params = np.array(df_fc_parameters.loc[fc_date, :])
    sigma, theta_p, kappa_p, vlambda = parameter_matrix_conv(2, params[:-1])
    yields_d = forecast_yields(tenors, factors_d, kappa_p[0], theta_p[0], sigma[0], vlambda[0], fc_horizon)
    df_fc_ir_d = pd.DataFrame(yields_d, index=fc_horizon, columns=tenors_col)
    df_fc_ir_d["fcdate"] = fc_date
    df_fc_ir_d.set_index('fcdate', append=True, inplace=True)
    list_fc_d.append(df_fc_ir_d)
    yields_f = forecast_yields(tenors, factors_f, kappa_p[1], theta_p[1], sigma[1], vlambda[1], fc_horizon)
    df_fc_ir_f = pd.DataFrame(yields_f, index=fc_horizon, columns=tenors_col)
    df_fc_ir_f["fcdate"] = fc_date
    df_fc_ir_f.set_index('fcdate', append=True, inplace=True)
    list_fc_f.append(df_fc_ir_f)

df_fc_ir_d = pd.concat(list_fc_d)
df_fc_ir_d.sort_index(inplace=True)
df_fc_ir_f = pd.concat(list_fc_d)
df_fc_ir_f.sort_index(inplace=True)

list_resid_d = []
list_resid_f = []

for fc_date in df_fc_ir_d.index.levels[1]:
    ahead03m_d = np.array(rates_d.iloc[rates_d.index.get_loc(fc_date) + 12, :-1])
    ahead06m_d = np.array(rates_d.iloc[rates_d.index.get_loc(fc_date) + 6, :-1])
    ahead12m_d = np.array(rates_d.iloc[rates_d.index.get_loc(fc_date) + 12, :-1])
    ahead03m_f = np.array(rates_f.iloc[rates_f.index.get_loc(fc_date) + 3, :-1])
    ahead06m_f = np.array(rates_f.iloc[rates_f.index.get_loc(fc_date) + 6, :-1])
    ahead12m_f = np.array(rates_f.iloc[rates_f.index.get_loc(fc_date) + 12, :-1])
    resid03m_d = pd.DataFrame(df_fc_ir_d.loc[(3, fc_date), :] - ahead03m_d).T
    resid06m_d = pd.DataFrame(df_fc_ir_d.loc[(6, fc_date), :] - ahead06m_d).T
    resid12m_d = pd.DataFrame(df_fc_ir_d.loc[(12, fc_date), :] - ahead12m_d).T
    resid03m_f = pd.DataFrame(df_fc_ir_f.loc[(3, fc_date), :] - ahead03m_f).T
    resid06m_f = pd.DataFrame(df_fc_ir_f.loc[(6, fc_date), :] - ahead06m_f).T
    resid12m_f = pd.DataFrame(df_fc_ir_f.loc[(12, fc_date), :] - ahead12m_f).T
    series_d = [resid03m_d, resid06m_d, resid12m_d]
    series_f = [resid03m_f, resid06m_f, resid12m_f]
    list_resid_d.append(pd.concat(series_d))
    list_resid_f.append(pd.concat(series_f))

df_resid_d = pd.concat(list_resid_d)
df_resid_f = pd.concat(list_resid_f)

mres_03m_d = (df_resid_d.loc[(3, slice(None)),:]).apply(lambda x: 10000*x.mean())
mres_06m_d = (df_resid_d.loc[(6, slice(None)),:]).apply(lambda x: 10000*x.mean())
mres_12m_d = (df_resid_d.loc[(12, slice(None)),:]).apply(lambda x: 10000*x.mean())

mres_03m_f = (df_resid_f.loc[(3, slice(None)),:]).apply(lambda x: 10000*x.mean())
mres_06m_f = (df_resid_f.loc[(6, slice(None)),:]).apply(lambda x: 10000*x.mean())
mres_12m_f = (df_resid_f.loc[(12, slice(None)),:]).apply(lambda x: 10000*x.mean())

rmse_03m_d = (df_resid_d.loc[(3, slice(None)),:]**2).apply(lambda x: 10000*np.sqrt(x.mean()))
rmse_06m_d = (df_resid_d.loc[(6, slice(None)),:]**2).apply(lambda x: 10000*np.sqrt(x.mean()))
rmse_12m_d = (df_resid_d.loc[(12, slice(None)),:]**2).apply(lambda x: 10000*np.sqrt(x.mean()))

rmse_03m_f = (df_resid_f.loc[(3, slice(None)),:]**2).apply(lambda x: 10000*np.sqrt(x.mean()))
rmse_06m_f = (df_resid_f.loc[(6, slice(None)),:]**2).apply(lambda x: 10000*np.sqrt(x.mean()))
rmse_12m_f = (df_resid_f.loc[(12, slice(None)),:]**2).apply(lambda x: 10000*np.sqrt(x.mean()))

dict_rmsfe = {"EUR RMSFE 3M": rmse_03m_d, "EUR RMSFE 6M": rmse_06m_d, "EUR RMSFE 12M": rmse_12m_d, "F RMSFE 3M": rmse_03m_f, "F RMSFE 6M": rmse_06m_f, "F RMSFE 12M": rmse_12m_f}
df_rmsfe = pd.DataFrame(dict_rmsfe)

