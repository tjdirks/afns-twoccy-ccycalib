import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
import itertools

df_fitted_yields_d = pd.read_pickle("fitted_yields_d.pickle")
df_fitted_yields_f_usd = pd.read_pickle("fitted_yields_f_usd.pickle")
df_bootstrapped_yields_d = pd.read_pickle("pickle_bootstrapped_eur.pickle")
df_bootstrapped_yields_f_usd = pd.read_pickle("pickle_bootstrapped_usd.pickle")
df_factors = pd.read_pickle("factors.pickle")

# sqrt(mean_squared_error)

sns.set_style("white")

#########
#########
#########
### EURO - RMSE and Residuals
mse_d = np.array([mean_squared_error(df_bootstrapped_yields_d.iloc[:, i], df_fitted_yields_d.iloc[:, i]) for i in
                  range(df_bootstrapped_yields_d.shape[1])])
rmse_d_bp = np.sqrt(mse_d) * 10000
residual_d = np.array([(df_bootstrapped_yields_d.iloc[:, i] - df_fitted_yields_d.iloc[:, i]).mean() for i in
                       range(df_bootstrapped_yields_d.shape[1])])
residual_d = residual_d * 10000

### USD - RMSE and Residuals
mse_f_usd = np.array(
    [mean_squared_error(df_bootstrapped_yields_f_usd.iloc[:, i], df_fitted_yields_f_usd.iloc[:, i]) for i in
     range(df_bootstrapped_yields_d.shape[1])])
rmse_f_usd_bp = np.sqrt(mse_f_usd) * 10000
residual_f_usd = np.array([(df_bootstrapped_yields_f_usd.iloc[:, i] - df_fitted_yields_f_usd.iloc[:, i]).mean() for i in
                           range(df_bootstrapped_yields_d.shape[1])])
residual_f_usd = residual_f_usd * 10000

#########
#########
#########
######### LATEX TABLE OUTPUT
tenors = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
tenors_in_months = (tenors * 12).astype(int)
column_names = ["Res. Mean EUR", "RMSE EUR", "Res. Mean USD", "RMSE USD"]

df_sum_stats = pd.DataFrame(np.around(np.vstack((residual_d, rmse_d_bp, residual_f_usd, rmse_f_usd_bp)), 2).T,
                            index=tenors_in_months,
                            columns=column_names)
latex_dump = open("latex.txt", "w")
latex_dump.write(df_sum_stats.to_latex())
latex_dump.close()

#########
#########
#########
# PROXY CALCULATION
# Level: (6M+5Y+30Y)/3
# Slope: 30Y-6M
# Curvature: 2*5Y-6M-25Y

# EURO
proxy_level_d = df_bootstrapped_yields_d[
                    ['Euribor 6 Month ACT/360', 'EUR SWAP ANN (VS 6M) 5Y', 'EUR SWAP ANN (VS 6M) 30Y']].sum(axis=1) / 3
proxy_slope_d = df_bootstrapped_yields_d['EUR SWAP ANN (VS 6M) 30Y'] - df_bootstrapped_yields_d[
    'Euribor 6 Month ACT/360']
proxy_curv_d = 2 * df_bootstrapped_yields_d['EUR SWAP ANN (VS 6M) 5Y'] - df_bootstrapped_yields_d[
    'Euribor 6 Month ACT/360'] - df_bootstrapped_yields_d['EUR SWAP ANN (VS 6M) 25Y']

# USD
proxy_level_f_usd = df_bootstrapped_yields_f_usd[
                        ['ICE LIBOR USD 6M', 'USD SWAP SEMI 30/360 5YR', 'USD SWAP SEMI 30/360 30Y']].sum(axis=1) / 3
proxy_slope_f_usd = df_bootstrapped_yields_f_usd['USD SWAP SEMI 30/360 30Y'] - df_bootstrapped_yields_f_usd[
    'ICE LIBOR USD 6M']
proxy_curv_f_usd = 2 * df_bootstrapped_yields_f_usd['USD SWAP SEMI 30/360 5YR'] - df_bootstrapped_yields_f_usd[
    'ICE LIBOR USD 6M'] - df_bootstrapped_yields_f_usd['USD SWAP SEMI 30/360 25Y']

######################
# CORRELATIONS
######################
correl_factors_proxies = {}
correl_factors_proxies["Global Level"] = pearsonr(df_factors.iloc[:, 0], (proxy_level_d + proxy_level_f_usd) / 2)[0]
correl_factors_proxies["Domestic Slope"] = pearsonr(df_factors.iloc[:, 1], proxy_slope_d)[0]
correl_factors_proxies["Foreign Slope"] = pearsonr(df_factors.iloc[:, 3], proxy_slope_f_usd)[0]
correl_factors_proxies["Domestic Curvature"] = pearsonr(df_factors.iloc[:, 2], proxy_curv_d)[0]
correl_factors_proxies["Foreign Curvature"] = pearsonr(df_factors.iloc[:, 4], proxy_curv_f_usd)[0]


current_palette = sns.color_palette("Paired")
######################
# LEVEL FACTOR PLOT
######################
fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

host.set_xlabel("Time")
host.set_ylabel("Global Level Factor")
par1.set_ylabel("Global Level Proxy")
par1.tick_params(colors="grey")

p1, = host.plot(df_bootstrapped_yields_d.index, df_factors.iloc[:, 0], color="black", label="Global Level Factor")
p2, = par1.plot(df_bootstrapped_yields_d.index, (proxy_level_d + proxy_level_f_usd) / 2, color="grey",
                label="Global Level Proxy")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

# no x-ticks
# par2.xaxis.set_ticks([])
# Sometimes handy, same for xaxis
# par2.yaxis.set_ticks_position('right')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("global_level_vs_proxy.png", bbox_inches='tight')

######################
# SlOPE FACTOR PLOT - Domestic
######################
fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

host.set_xlabel("Time")
host.set_ylabel("Domestic Slope Factor")
par1.set_ylabel("Domestic Slope Proxy")
par1.tick_params(colors="grey")

p1, = host.plot(df_bootstrapped_yields_d.index, -df_factors.iloc[:, 1], color="black", label="Domestic Slope Factor")
p2, = par1.plot(df_bootstrapped_yields_d.index, proxy_slope_d, color="grey", label="Domestic Slope Proxy")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

# no x-ticks
# par2.xaxis.set_ticks([])
# Sometimes handy, same for xaxis
# par2.yaxis.set_ticks_position('right')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("domestic_slope_vs_proxy.png", bbox_inches='tight')

######################
# SlOPE FACTOR PLOT - Foreign
######################
fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

host.set_xlabel("Time")
host.set_ylabel("Foreign Slope Factor")
par1.set_ylabel("Foreign Slope Proxy")
par1.tick_params(colors="grey")

p1, = host.plot(df_bootstrapped_yields_d.index, -df_factors.iloc[:, 3], color="black", label="Foreign Slope Factor")
p2, = par1.plot(df_bootstrapped_yields_d.index, proxy_slope_f_usd, color="grey", label="Foreign Slope Proxy")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

# no x-ticks
# par2.xaxis.set_ticks([])
# Sometimes handy, same for xaxis
# par2.yaxis.set_ticks_position('right')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("foreign_slope_vs_proxy.png", bbox_inches='tight')

######################
# CURVATURE FACTOR PLOT - Domestic
######################
fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

host.set_xlabel("Time")
host.set_ylabel("Domestic Curvature Factor")
par1.set_ylabel("Domestic Curvature Proxy")
par1.tick_params(colors="grey")

p1, = host.plot(df_bootstrapped_yields_d.index, df_factors.iloc[:, 2], color="black", label="Domestic Curvature Factor")
p2, = par1.plot(df_bootstrapped_yields_d.index, proxy_curv_d, color="grey", label="Domestic Curvature Proxy")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

# no x-ticks
# par2.xaxis.set_ticks([])
# Sometimes handy, same for xaxis
# par2.yaxis.set_ticks_position('right')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("domestic_curv_vs_proxy.png", bbox_inches='tight')

######################
# CURVATURE FACTOR PLOT - Foreign
######################
fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

host.set_xlabel("Time")
host.set_ylabel("Foreign Curvature Factor")
par1.set_ylabel("Foreign Curvature Proxy")
par1.tick_params(colors="grey")

p1, = host.plot(df_bootstrapped_yields_d.index, -df_factors.iloc[:, 4], color="black", label="Foreign Curvature Factor")
p2, = par1.plot(df_bootstrapped_yields_d.index, proxy_curv_f_usd, color="grey", label="Foreign Curvature Proxy")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

# no x-ticks
# par2.xaxis.set_ticks([])
# Sometimes handy, same for xaxis
# par2.yaxis.set_ticks_position('right')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("foreign_curv_vs_proxy.png", bbox_inches='tight')

######################
# Fitted yields plot 6M
######################
palette = itertools.cycle(current_palette)

fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

host.xaxis_date()
par1.xaxis_date()

host.set_xlabel("Time")
host.set_ylabel("EUR 6M Yield")
par1.set_ylabel("USD 6M Yield")

p1, = host.plot(df_bootstrapped_yields_d.index, df_bootstrapped_yields_d["Euribor 6 Month ACT/360"], color=next(palette), label="EUR Bootstrapped")
p2, = host.plot(df_bootstrapped_yields_d.index, df_fitted_yields_d["6M"], color=next(palette), label="EUR Fitted")
p3, = par1.plot(df_bootstrapped_yields_d.index, df_bootstrapped_yields_f_usd["ICE LIBOR USD 6M"], color=next(palette), label="USD Bootstrapped")
p4, = par1.plot(df_bootstrapped_yields_d.index, df_fitted_yields_f_usd["6M"], color=next(palette), label="USD Fitted")

lns = [p1, p2, p3, p4]
host.legend(handles=lns, loc='best')

plt.tight_layout()

plt.savefig("6M Rates.png")

######################
# Fitted yields plot 5Y
######################
palette = itertools.cycle(current_palette)

fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()

host.xaxis_date()
par1.xaxis_date()

host.set_xlabel("Time")
host.set_ylabel("EUR 5Y Yield")
par1.set_ylabel("USD 5Y Yield")

p1, = host.plot(df_bootstrapped_yields_d.index, df_bootstrapped_yields_d["EUR SWAP ANN (VS 6M) 5Y"], color=next(palette), label="EUR Bootstrapped")
p2, = host.plot(df_bootstrapped_yields_d.index, df_fitted_yields_d["5Y"], color=next(palette), label="EUR Fitted")
p3, = par1.plot(df_bootstrapped_yields_d.index, df_bootstrapped_yields_f_usd["USD SWAP SEMI 30/360 5YR"], color=next(palette), label="USD Bootstrapped")
p4, = par1.plot(df_bootstrapped_yields_d.index, df_fitted_yields_f_usd["5Y"], color=next(palette), label="USD Fitted")

lns = [p1, p2, p3, p4]

host.legend(handles=lns, loc='best')

plt.tight_layout()

plt.savefig("5Y Rates.png")

######################
# Fitted yields plot 5Y
######################
palette = itertools.cycle(current_palette)

fig = plt.figure()
host = fig.add_subplot(111)


par1 = host.twinx()

host.xaxis_date()
par1.xaxis_date()

host.set_xlabel("Time")
host.set_ylabel("EUR 10Y Yield")
par1.set_ylabel("USD 10Y Yield")

p1, = host.plot(df_bootstrapped_yields_d.index, df_bootstrapped_yields_d["EUR SWAP ANN (VS 6M) 10Y"], color=next(palette), label="EUR Bootstrapped")
p2, = host.plot(df_bootstrapped_yields_d.index, df_fitted_yields_d["10Y"], color=next(palette), label="EUR Fitted")
p3, = par1.plot(df_bootstrapped_yields_d.index, df_bootstrapped_yields_f_usd["USD SWAP SEMI 30/360 10Y"], color=next(palette), label="USD Bootstrapped")
p4, = par1.plot(df_bootstrapped_yields_d.index, df_fitted_yields_f_usd["10Y"], color=next(palette), label="USD Fitted")

lns = [p1, p2, p3, p4]
host.legend(handles=lns, loc='best')

plt.tight_layout()

plt.savefig("10Y Rates.png")