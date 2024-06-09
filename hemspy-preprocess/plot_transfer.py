import pandas as pd
import geopandas

def plot_transfer(raw_df, transfer_df, transfer_id):
    """
    Plot all flights for a given aircraft within the timespan of a transfer.

    Args:
        raw_df (pd.DataFrame): DataFrame containing raw flight data.
        transfer_df (pd.DataFrame): DataFrame containing transfer data.
        transfer_id (int): ID of the transfer.

    Returns:
        A geopandas plot
    """
    # Filter transfer data
    transfer = transfer_df[transfer_df['transfer_id'] == transfer_id]
    aircraft = transfer['aircraft_id_receiving'].iloc[0]
    start_time = transfer['UTC_out_sending'].iloc[0]
    end_time = transfer['UTC_receiving'].iloc[0]

    # Filter raw data based on aircraft and time range
    filtered_df = raw_df[raw_df.index.get_level_values(0) == aircraft]
    filtered_df = filtered_df[(filtered_df['UTC'] > start_time) & (filtered_df['UTC'] < end_time)]

    # Get unique flight IDs within the time range
    flights_set = filtered_df['flight_id'].unique()

    # Filter raw data based on flight IDs
    transfer_flights = raw_df[raw_df['flight_id'].isin(flights_set)]
    transfer_flights = transfer_flights[['geometry', 'flight_id', 'UTC_str', 'reg', 'zone_name', 'altitude', 'speed', 'snapshot_id']]

    # Explore the transfer flights
    return transfer_flights.explore()