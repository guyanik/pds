import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _get_period(data: pd.Series, corr_threshold: float = 0.2):
    acf = np.correlate(data, data, 'full')[-len(data):]
    inflection = np.diff(np.sign(np.diff(acf)))
    peaks = (inflection < 0).nonzero()[0] + 1
    if len(peaks) == 0:
        print('No seasonality found')
        return 1, 0
    period = peaks[acf[peaks].argmax()]
    corr = np.corrcoef(data[period:], data[: -period])[0, 1]
    if corr < corr_threshold or period == 1:
        print('No seasonality found')
        return 1, corr
    return period, corr


def _get_initial_params(df: pd.DataFrame, column: str, log: bool = True):
    period, corr = _get_period(df[column])

    df_log = df.copy()
    if log:
        df_log[column] = df_log[column].replace(0, np.nan).interpolate(method='cubicspline')
        df_log[column] = np.log(df_log[column])
    else:
        df_log[column] = df_log[column].replace(0, 3600)

    # deseaonalize
    df_log['Seasonal'] = seasonal_decompose(df_log[column], model='additive', period=period).seasonal
    df_log['DS'] = seasonal_decompose(df_log[column], model='additive', period=period).trend

    # linear regression with deseasonalized SECONDSSPENT
    reg_df = df_log[df_log['DS'].notna()]
    endog = np.array(reg_df['DS'])
    exog = sm.add_constant(np.array(reg_df.index.dayofyear))
    model = sm.OLS(endog, exog)
    results = model.fit()
    # print(results.summary())

    initial_level = results.params[0]
    initial_trend = results.params[1]
    df_log['DS'] = initial_level + initial_trend * np.array(df_log.index.dayofyear)
    df_log['Seasonal Factor'] = df_log[column] / df_log['DS']

    initial_seasonal = []
    if period == 1:
        initial_seasonal = [1]
    else:
        for i in range(period):
            initial_seasonal.append(df_log.iloc[i::period].mean()['Seasonal Factor'])
    
    return period, initial_level, initial_trend, initial_seasonal


def forecast(df: pd.DataFrame, 
            column: str, 
            log: bool = False,
            alpha: float = 0.5, 
            beta: float =  0.01, 
            gamma: float = 0.01):
    
    period, initial_level, initial_trend, initial_seasonal = _get_initial_params(df, column, log)
    print('Period: {}'.format(period))
    print('Initial Level: {}'.format(initial_level))
    print('Initial Trend: {}'.format(initial_trend))
    print('Initial Seasonal: {}'.format(initial_seasonal))
    
    df_log = df.copy()
    if log:
        df_log[column] = df_log[column].replace(0, np.nan).interpolate(method='cubicspline')
        df_log[column] = np.log(df_log[column])
    else:
        df_log[column] = df_log[column].replace(0, 3600)
    
    if period != 1:
        winters_model = ExponentialSmoothing(
        df_log[column],
        trend="add",
        seasonal="mul",
        seasonal_periods=period,
        initialization_method="known",
        initial_level=initial_level,
        initial_trend=initial_trend,
        initial_seasonal=initial_seasonal
    ).fit(smoothing_level=alpha,
        smoothing_trend=beta,
        smoothing_seasonal=gamma,
        )
    else:
        winters_model = ExponentialSmoothing(
        df_log[column],
        trend="add",
        seasonal=None,
        initialization_method="known",
        initial_level=initial_level,
        initial_trend=initial_trend
    ).fit(smoothing_level=alpha,
        smoothing_trend=beta
        )

    last_day = df_log.index[-1]
    num_periods = period
    df_log["FORECAST"] = winters_model.fittedvalues
    forecast = pd.DataFrame(
        {'FORECAST': list(winters_model.forecast(num_periods))},
        index=pd.date_range(start=last_day + pd.DateOffset(days=1), periods=num_periods, freq='D')
    )

    df_forecast = df_log.append(forecast)
    if log:
        df_forecast[[column,'FORECAST']] = np.exp(df_forecast[[column,'FORECAST']])
    print(df_forecast.tail(period*2))

    return df_forecast


def plot_forecast(df: pd.DataFrame, column: str, title: str = ''):
    df_forecast = forecast(df, column)
    # df_plot = df_forecast[df_forecast.index >= '2022-08-10']
    df_plot = df_forecast[-66:]

    test_size = int(len(df_plot) * 0.2)
    test = df_plot[-test_size:]
    mae = np.mean(np.abs(test[column] - test['FORECAST'])) / 3600
    print('MAE: {:.3f}'.format(mae))

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 6)
    ax.plot(df_plot.index, df_plot[column] / 3600, 
            linestyle='none', 
            marker='s',
            markerfacecolor='cornflowerblue', 
            markeredgecolor='black',
            markersize=7,
            label='Hours spent per day')
    ax.plot(df_plot.index, df_plot['FORECAST'] / 3600, 
            linestyle='-',
            marker='o',
            markersize=5,
            color='red',
            label='Forecast')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
    ax.set_xlabel('Date')
    ax.set_ylabel('Hours Spent')
    plt.xticks(rotation=75)
    ax.legend(loc='upper left')
    plt.show()

