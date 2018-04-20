from AFNSGlobal.kalman_filter_functions import *
from AFNSGlobal.fitted_yields_functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tenors = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
tenors_col = ["1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y",
              "20Y", "25Y", "30Y"]
df_results = pd.read_pickle("factors.pickle")
df_parameters = pd.read_pickle("parameters.pickle")
df_results = pd.read_pickle("fc_factors_temp.pickle")
df_parameters = pd.read_pickle("fc_parameters_temp.pickle")
df_results.sort_index(inplace=True)
parameters = np.array(df_parameters.iloc[-5,:-1])
parameters_all = parameter_matrix_conv(2, parameters)
alist_sigma = parameters_all[0]
list_lambda = parameters_all[3]

fitted_yields_d = []
fitted_yields_f_usd = []

for i in range(df_results.shape[0]):
    factors_d = np.array(df_results.iloc[i, :3])
    factors_f_usd = np.array([df_results.iloc[i, 0], *df_results.iloc[i, 3:5]])
    factors_f_jpy = np.array([df_results.iloc[i, 0], *df_results.iloc[i, 5:7]])
    factors_f_gbp = np.array([df_results.iloc[i, 0], *df_results.iloc[i, 7:]])
    fitted_yields_d.append(calc_fitted_yield(tenors, factors_d, list_lambda[0], alist_sigma[0]))
    fitted_yields_f_usd.append(calc_fitted_yield(tenors, factors_f_usd, list_lambda[1], alist_sigma[1]))

df_fitted_yields_d = pd.DataFrame(fitted_yields_d, index=df_results.index, columns=tenors_col)
df_fitted_yields_f_usd = pd.DataFrame(fitted_yields_f_usd, index=df_results.index, columns=tenors_col)

df_fitted_yields_d.to_pickle("fitted_yields_d.pickle")
df_fitted_yields_f_usd.to_pickle("fitted_yields_f_usd.pickle")


df_fitted_yields_d.to_excel("fitted_yields_d.xlsx", sheet_name="Fitted Domestic", index=True)
df_fitted_yields_f_usd.to_excel("fitted_yields_f_usd.xlsx", sheet_name="Fitted Foreign", index=True)

df_fitted_yields_d.plot()
plt.show()
