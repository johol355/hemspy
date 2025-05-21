import pandas as pd
import numpy as np
import pandas as pd
import numpy as np


def hems_minima(row=None):
    daylight = row['daylight']
    cloud_ceiling = row['metar_cloud_ceiling']
    cloud_base = row['metar_cloud_base']
    visibility = row['metar_visibility']

    if daylight:
        if np.isnan(cloud_ceiling) or np.isnan(visibility):
            return np.NaN
        elif cloud_ceiling >= 400 and cloud_ceiling < 500:
            return visibility >= 1000
        elif cloud_ceiling >= 300 and cloud_ceiling < 400:
            return visibility >= 2000
        elif cloud_ceiling >= 500:
            return visibility >= 800
        else:
            return False

    if not daylight:
        if np.isnan(cloud_base) or np.isnan(visibility):
            return np.NaN
        else:
            return cloud_base >= 1200 and visibility >= 2500


def hems_minima_window(d=None, window='241Min'):
    grouped_by_airport = d.groupby('icao')

    # define a function to infer the weather conditions for a time window
    def weather_minima(period):
        if period.isna().all():
                return np.NaN
        else:
                return True if period.any() else False
    
    # iterate over airports and infer weahter conditions using the previous function
    results = []
    for airport, group in grouped_by_airport:   
        # sort
        group.sort_values(by='time_utc', inplace=True)

        # Apply rolling window and infer weather conditions
        result = group.rolling(on='time_utc', window=window, center=True)['hems_minima'].apply(weather_minima)
        # Add inferred weather conditions to the DataFrame
        group['hems_minima_window'] = result
        results.append(group)

    # Concatenate results for all airports
    res_df = pd.concat(results)
    res_df['hems_minima_window'] = res_df['hems_minima_window'].map({1: True, 0: False, None: np.NaN})

    return res_df




