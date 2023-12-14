import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_dataframe(filename: str='daily_total.csv') -> pd.DataFrame:
    df = pd.read_csv(filename)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df[df['DATE'] > '2022-08-10']
    # fill missing dates with 0
    df.index = df['DATE']
    df = df.reindex(pd.date_range(min(df['DATE']), max(df['DATE'])), fill_value=0).drop('DATE', axis=1)
    return df


def forecast(df: pd.DataFrame, column: str, periods: int = 7) -> pd.DataFrame:
    model = pm.auto_arima(df[column], 
                        m=periods, 
                        seasonal=True, 
                        start_p=0, 
                        start_q=0, 
                        max_order=4, 
                        test='adf', 
                        trace=True, 
                        error_action='ignore', 
                        suppress_warnings=True, 
                        stepwise=True)
    
    train, test = df[:-periods], df[-periods:]
    model.fit(train[column])

    # add 7 days to df
    last_day = df.index[-1]
    forecast = model.predict(n_periods=periods + 7, return_conf_int=True)
    forecast_range = pd.date_range(start=last_day - pd.DateOffset(days=periods - 1), periods=periods + 7, freq='D')
    forecast_df = pd.DataFrame(forecast[0], index=forecast_range, columns=['FORECAST'])
    df_forecast = pd.concat([df[column], forecast_df], axis=1)
    return df_forecast


def plot_forecast(df_forecast: pd.DataFrame, column: str, periods: int = 7) -> None:
    df_plot = df_forecast[-(periods + 5):]
    date = df_plot.index

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 6)
    ax.plot(date, df_plot[column] / 3600, 
            linestyle='none', 
            marker='s',
            markerfacecolor='cornflowerblue', 
            markeredgecolor='black',
            markersize=7,
            label='Hours spent per day')
    ax.plot(date, df_plot['FORECAST'] / 3600, 
            linestyle='-',
            marker='o',
            markersize=5,
            color='red',
            label='Prediction')
    # set min and max of y-axis
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
    ax.set_xlabel('Date')
    ax.set_ylabel('Hours Spent')
    plt.xticks(rotation=75)
    ax.legend(loc='upper left')
    plt.show()