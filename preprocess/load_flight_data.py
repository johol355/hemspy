import pandas as pd
import geopandas as gpd
import os
import re

def read_airports_data(path='/Users/JO/PhD/hemspy/data/helipad-data/raw-data/helipad-coordinates.csv', radius_column='radius'):
    """
    Read airports data from a CSV file and perform necessary transformations.

    Parameters:
    - path (str): Path to the airports data CSV file.
    - radius_column (str): Name of the column containing the radius information.

    Returns:
    - airports_gdf (geopandas.geodataframe.GeoDataFrame): GeoDataFrame containing the transformed airports data.
    """
    airports_path = path 
    airports = pd.read_csv(airports_path, sep=';')
    radius = airports[radius_column]
    airports_gdf = gpd.GeoDataFrame(airports, geometry=gpd.points_from_xy(airports.longitude, airports.latitude), crs="EPSG:4326")
    airports_gdf = airports_gdf.to_crs("EPSG:32634") #to metric coords
    airports_gdf.geometry = airports_gdf.geometry.buffer(distance=radius)
    airports_gdf = airports_gdf.to_crs("EPSG:4326") #back to conventional
    return airports_gdf

def read_flights_data(path='/Users/JO/PhD/hemspy/data/fr24-data/raw-data-unzipped-rearranged/flights'):
    """
    Read flight data from CSV files and return a concatenated DataFrame.

    Parameters:
    path (str): The path to the directory containing the flight data CSV files. Default is '/Users/JO/PhD/hemspy/data/fr24-data/raw-data-unzipped-rearranged/flights'.

    Returns:
    pandas.DataFrame: A DataFrame containing the concatenated flight data.

    """
    flights_path = path
    flights_files = [f for f in os.listdir(flights_path) if f.endswith('.csv')]
    flights_df_list = [pd.read_csv(os.path.join(flights_path, file)) for file in flights_files]
    flights_df = pd.concat(flights_df_list, ignore_index=True)
    flights_df['flight_id'] = flights_df['flight_id'].astype(int)
    return flights_df

def read_positions_data(path='/Users/JO/PhD/hemspy/data/fr24-data/raw-data-unzipped-rearranged/positions', drop_last=True):
    """
    Read and process flight positions data from CSV files.

    Args:
        path (str, optional): The path to the directory containing the CSV files. Defaults to '/Users/JO/PhD/hemspy/data/fr24-data/raw-data-unzipped-rearranged/positions'.
        drop_last (bool, optional): Whether to drop the last position snapshot for each flight.

    Returns:
        pandas.DataFrame: A DataFrame containing the flight positions data with columns ['snapshot_id', 'altitude', 'latitude', 'longitude', 'speed', 'flight_id'].
    """
    positions_path = path
    positions_files = [os.path.join(positions_path, file) for file in os.listdir(positions_path) if file.endswith('.csv')]
    positions_df_list = []
    regex_pattern = r'_(.*?)\.'
    regex = re.compile(regex_pattern)
    for file in positions_files:
        flight_id_match = regex.search(os.path.basename(file))
        if flight_id_match:
            flight_id = flight_id_match.group(1)
            df = pd.read_csv(file, usecols=['snapshot_id', 'altitude', 'latitude', 'longitude', 'speed'])
            df['flight_id'] = flight_id
            positions_df_list.append(df)
    positions_df = pd.concat(positions_df_list, ignore_index=True)
    positions_df['flight_id'] = positions_df['flight_id'].astype(int)
    if drop_last:
        positions_df = positions_df.groupby('flight_id').apply(lambda x: x.iloc[:-1], include_groups=False)
    return positions_df

def merge_flights_and_positions_data(positions_df, flights_df):
    """
    Merge flight positions data with flight information data.

    Args:
        positions_df (pandas.DataFrame): DataFrame containing flight positions data.
        flights_df (pandas.DataFrame): DataFrame containing flight information data.

    Returns:
        pandas.DataFrame: Merged DataFrame with flight positions and information.

    """
    d = pd.merge(positions_df, flights_df, on='flight_id', how='left')
    d['UTC'] = pd.to_datetime(d['snapshot_id'], unit='s', utc=True)
    d['UTC_str'] = d['UTC'].astype(str)

    d['date'] = d['UTC'].dt.date
    d['year'] = d['UTC'].dt.year
    return d

def filter_flight_data(d, airports_gdf, include_equip=['EC45', 'A139', 'A169', 'S76', 'AS65'], exclude_callsign=['SEJSR', 'SEJSP', 'SEJRH', 'SEJRI', 'SERJR', 'SEJRK', 'SEJRL', 'SEJRM', 'SEJRN']):
    """
    Filters flight data based on equipment and callsign criteria.

    Args:
        d (pandas.DataFrame): Flight data to be filtered.
        airports_gdf (geopandas.GeoDataFrame): GeoDataFrame containing airport information.
        include_equip (list, optional): List of equipment codes to include. Defaults to ['EC45', 'A139', 'A169', 'S76', 'AS65'].
        exclude_callsign (list, optional): List of callsigns to exclude. Defaults to ['SEJSR', 'SEJSP', 'SEJRH', 'SEJRI', 'SERJR', 'SEJRK', 'SEJRL', 'SEJRM', 'SEJRN'].

    Returns:
        pandas.DataFrame: Filtered flight data.
    """
    d = d[d['equip'].isin(include_equip)]
    d = d[~(d['reg'].isin(exclude_callsign))]
    d = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.longitude, d.latitude), crs="EPSG:4326")
    d = gpd.sjoin(d, airports_gdf, how='left', predicate='within')
    d.drop(['index_right', 'icao', 'is_primary_helipad', 'latitude_right', 'reserved', 'flight', 'callsign', 'longitude_right', 'real_to', 'schd_from', 'schd_to'], axis=1, inplace=True)
    d = d.groupby('aircraft_id').apply(lambda x: x.sort_values('UTC'), include_groups=False)
    return d

def load_flight_data(drop_last, path_airports_data, path_flights_data):
    """
    Process flight data by reading airports, flights, and positions data,
    merging them, and filtering the data based on airports.

    Returns:
        DataFrame: Processed flight data.

    """
    airports_gdf = read_airports_data(path=path_airports_data, radius_column='radius')
    flights_df = read_flights_data(path=path_flights_data)
    positions_df = read_positions_data(drop_last=drop_last)
    d = merge_flights_and_positions_data(positions_df, flights_df)
    d = filter_flight_data(d, airports_gdf)
    return d