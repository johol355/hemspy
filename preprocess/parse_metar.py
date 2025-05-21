import os
import pandas as pd
import ephem
import warnings
from metar import Metar

def concatenate_csv_files(path="/Users/JO/PhD/neurocritical-transfers/data/metar"):
    '''
    Concatenate multiple CSV files containing METAR data into a single DataFrame.

    Parameters:
    - path (str): Path to the directory containing the CSV files.

    Returns:
    - pandas.DataFrame: A DataFrame containing concatenated METAR data with columns: 'ICAO', 'time_utc', 'METAR'.

    This function reads all CSV files in the specified directory `path` and concatenates them into a single DataFrame.
    It assumes that each CSV file contains METAR data with the following structure:
    - Column 0: ICAO code.
    - Columns 1-5: Year, month, day, hour, and minute of the METAR observation timestamp in UTC.
    - Column 6: METAR observation string.
    Any rows with parsing errors are skipped.

    Example:
    >>> concatenated_metar = concatenate_csv_files("/path/to/csv_files")
    >>> print(concatenated_metar.head())
          ICAO                 time_utc                                              METAR
    0     KJFK  2022-01-01 00:00:00  KJFK 010000Z 32010KT 10SM FEW065 SCT075 OVC110 ...
    1     KJFK  2022-01-01 01:00:00  KJFK 010100Z 32011KT 10SM FEW055 SCT070 OVC090 ...
    2     KJFK  2022-01-01 02:00:00  KJFK 010200Z 32012KT 10SM FEW050 SCT060 OVC080 ...
    3     KJFK  2022-01-01 03:00:00  KJFK 010300Z 32012KT 10SM FEW045 SCT055 OVC070 ...
    4     KJFK  2022-01-01 04:00:00  KJFK 010400Z 32012KT 10SM FEW040 SCT050 OVC060 ...
    '''

    #dfs = []
    dfs = [pd.read_csv(os.path.join(path, file), on_bad_lines='skip', header=None) for file in os.listdir(path) if file.endswith('.csv')]
    concatenated_df = pd.concat(dfs, ignore_index=True)
    #for file in os.listdir(directory):
    #    if file.endswith('.csv'):
    #        try:
    #            df = pd.read_csv(os.path.join(directory, file), on_bad_lines='skip', header=None)
    #            dfs.append(df)
    #        except pd.errors.ParserError:
    #            continue
    #concatenated_df = pd.concat(dfs, ignore_index=True)

    # Fix the timestamp from the METAR .csv
    time_utc = pd.to_datetime({
        'year': concatenated_df[1],
        'month': concatenated_df[2],
        'day': concatenated_df[3],
        'hour': concatenated_df[4],
        'minute': concatenated_df[5]
    }, utc=True)

    concatenated_df['time_utc'] = time_utc
    concatenated_df.rename({0: 'icao', 6: 'metar'}, axis=1, inplace=True)
    final_df = concatenated_df[['icao', 'time_utc', 'metar']]

    return final_df

def preprocess_metar(metar):
    """
    Preprocess METAR string by removing the third item (timestamp) and any occurrences of "COR".

    Parameters:
    - metar (str): METAR string to preprocess.

    Returns:
    - str: Preprocessed METAR string.
    """

    # Remove "COR"
    metar = metar.replace("COR", "")

    # Remove timestamp
    metar_parts = metar.split()
    if len(metar_parts) >= 3:
        del metar_parts[2]  # Remove the third item (timestamp)
        return ' '.join(metar_parts)
    else:
        return None  # Invalid METAR format

def parse_metar_str(metar=None):
    '''
    Parse METAR (Meteorological Aerodrome Report) data to extract weather information.

    Parameters:
    - metar (pandas.Series): Series containing METAR strings.

    Returns:
    - tuple: A tuple containing three pandas.Series objects:
             1. Horizontal visibility in meters.
             2. Cloud ceiling in feet.
             3. Cloud base in feet.
             
             Each Series corresponds to the respective weather parameter extracted from the METAR data.

    This function internally utilizes three helper functions to extract specific weather parameters:
    - get_horizontal_visibility: Extracts horizontal visibility in meters.
    - get_ceiling: Extracts cloud ceiling in feet.
    - get_base: Extracts cloud base in feet.

    The extracted weather parameters are applied to the entire pandas.Series using the `apply` method.

    Note: If parsing encounters errors, it returns descriptive error messages.
    '''
    metar = metar.apply(preprocess_metar)

    # define a function to get the horisontal visibility in meters from METAR
    def get_horisontal_visibility(metar=None):
        if metar is None:
            return None 
        if 'CAVOK' in metar:
                return 9999  # Set ceiling to 9999 if CAVOK is present
        else:
            try:
                observation = Metar.Metar(metar, strict=False)
                if observation.vis:
                    return observation.vis.value(units="M")
                else:
                    return None
            except Metar.ParserError as pe:
                return 'ParserError'

    # define a function of get the cloud ceiling in ft from METAR
    def get_ceiling(metar=None):
        if metar is None:
            return None 
        if 'CAVOK' in metar:
                return 9999  # Set ceiling to 9999 if CAVOK is present
        else:
            try:
                observation = Metar.Metar(metar, strict=False)
                if observation.sky:
                    try:
                        bkn = [tup[1].value(units="FT") for tup in observation.sky if tup[0] == 'BKN']
                        ovc = [tup[1].value(units="FT") for tup in observation.sky if tup[0] == 'OVC']
                        vv = [tup[1].value(units="FT") for tup in observation.sky if tup[0] == 'VV']
                        ncd = [9999 for tup in observation.sky if tup[0] == 'NCD']
                        nsc = [9999 for tup in observation.sky if tup[0] == 'NSC']
                    except:
                        return None
                    
                    # Create a list of non-empty lists
                    ceilings = [value for value in (ovc, bkn, vv, ncd, nsc) if value]
                    
                    # Flatten the list of lists into a single list
                    ceilings = [item for sublist in ceilings for item in sublist]

                    # Check if there are any elements in the list
                    if ceilings:
                        # Find the minimum among non-None values
                        min_value = min(ceilings)
                    else:
                        min_value = 9999
                    return min_value
                else:
                    return None
            except Metar.ParserError as pe:
                return 'ParserError'
        
    # define a function of get the cloud base in ft from METAR
    def get_base(metar=None):
        if metar is None:
            return None 
        if 'CAVOK' in metar:
                return 9999  # Set ceiling to 9999 if CAVOK is present
        else:
            try:
                observation = Metar.Metar(metar, strict=False)
                if observation.sky:
                    try:
                        few = [tup[1].value(units="FT") for tup in observation.sky if tup[0] == 'FEW']
                        sct = [tup[1].value(units="FT") for tup in observation.sky if tup[0] == 'SCT']
                        bkn = [tup[1].value(units="FT") for tup in observation.sky if tup[0] == 'BKN']
                        ovc = [tup[1].value(units="FT") for tup in observation.sky if tup[0] == 'OVC']
                        vv = [tup[1].value(units="FT") for tup in observation.sky if tup[0] == 'VV']
                        ncd = [9999 for tup in observation.sky if tup[0] == 'NCD']
                        nsc = [9999 for tup in observation.sky if tup[0] == 'NSC']
                    except:
                        return None          
                    
                # Create a list of non-empty lists
                    bases = [value for value in (few, sct, ovc, bkn, vv, ncd, nsc) if value]
                    
                    # Flatten the list of lists into a single list
                    bases = [item for sublist in bases for item in sublist]

                    # Check if there are any elements in the list
                    if bases:
                        # Find the minimum among non-None values
                        min_value = min(bases)
                    else:
                        min_value = None
                    return min_value
                else:
                    return None
            except Metar.ParserError as pe:
                return 'ParserError'

    visbility = metar.apply(get_horisontal_visibility)
    ceiling = metar.apply(get_ceiling)
    base = metar.apply(get_base)

    return visbility, ceiling, base

def get_twilight(times=None, airports=None, airport_coords_path=None):
    '''
    Calculate whether twilight occurs at specified times for given airports.

    Parameters:
    - times (list-like): A list-like object containing timestamps.
    - airports (pandas.Series): Series containing airport ICAO codes.
    - airport_coords_path (str): Path to a CSV file containing airport coordinates data.

    Returns:
    - list: A list containing boolean values indicating whether twilight occurs at each specified time for each airport.

    This function calculates whether twilight occurs at specified times for given airports.
    It utilizes PyEphem library to compute the altitude of the Sun for each airport at the specified times.
    Twilight is considered to occur when the altitude of the Sun is greater than -6 degrees.
    The input DataFrame `airports` is expected to have the following columns:
    - 'icao': ICAO code of the airport.
    - 'Latitude': Latitude of the airport in decimal degrees.
    - 'Longitude': Longitude of the airport in decimal degrees.
    The input CSV file specified by `airport_coords_path` should contain airport coordinates data with columns: 'icao', 'Latitude', 'Longitude'.

    Example:
    >>> times = ['2024/03/22 06:00', '2024/03/22 18:00', '2024/03/22 12:00']
    >>> airports = pd.DataFrame({'icao': ['ESSA', 'ESSB', 'ESSA']})
    >>> airport_coords_path = "/Users/JO/PhD/neurocritical-transfers/data/icu-airport-mapping-with-coordinates.csv"
    >>> twilight_results = get_twilight(times=times, airports=airports, airport_coords_path=airport_coords_path)
    >>> print(twilight_results)
    [True, False, True]
    '''

    airports_mapping = pd.read_csv(airport_coords_path, sep=";")
    airports_mapping = airports_mapping.drop_duplicates(subset='icao').dropna(subset='icao')
    airports_w_coords = pd.merge(airports, airports_mapping, how='left', left_on='icao', right_on='icao')
    
    tw = []
    lat = []
    lon = []

    for index, row in airports_w_coords.iterrows():
        if row['Latitude'] is None or row['Longitude'] is None:
            tw.append(None)
        else:
            airport_observer = ephem.Observer()
            airport_observer.lat = str(row['Latitude'])
            airport_observer.lon = str(row['Longitude'])
            airport_observer.date = times[index]
            sun = ephem.Sun()
            sun.compute(airport_observer)
            sun_altitude = float(sun.alt) * 180.0 / ephem.pi
            twilight_degree = -6
            tw.append(sun_altitude > twilight_degree)
    
    return tw

def get_metar_data(PATH_METAR, PATH_AIRPORT_COORDS, save_csv_path=None):
    '''
    Fetch METAR data, parse it, and extract relevant information.

    Parameters:
    - PATH_METAR (str): Path to the directory containing METAR CSV files.
    - PATH_AIRPORT_COORDS (str): Path to the CSV file containing airport coordinates data.
    - save_csv_path (str, optional): Path to save the resulting DataFrame as a CSV file. Default is None.

    Returns:
    - pandas.DataFrame: DataFrame containing METAR data with additional columns for parsed information.

    This function fetches METAR data from CSV files located at PATH_METAR, parses the data, and extracts relevant information such as visibility, cloud ceiling, cloud base, and daylight status.
    The function also calculates whether it's daylight at each airport based on the provided airport coordinates and timestamps.
    If save_csv_path is specified, the resulting DataFrame is saved as a CSV file at the specified location.
    '''
    warnings.filterwarnings("ignore")

    metar_df = concatenate_csv_files(PATH_METAR)
    visibility, ceiling, base = parse_metar_str(metar_df['metar'])
    metar_df['metar_visibility'] = visibility
    metar_df['metar_cloud_ceiling'] = ceiling
    metar_df['metar_cloud_base'] = base
    tw = get_twilight(times=metar_df['time_utc'], airports=metar_df['icao'], airport_coords_path=PATH_AIRPORT_COORDS)
    metar_df['daylight'] = tw

    if save_csv_path is not None:
        try:
            metar_df.to_csv(path_or_buf=save_csv_path, sep=";", index=False)
        except Exception as e:
            print(f"Error saving DataFrame to CSV: {e}")

    return metar_df