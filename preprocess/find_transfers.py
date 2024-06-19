import pandas as pd
import geopandas
import numpy as np

def extract_entries_and_exits(d):
    """
    Clean the entries and exits data by performing the following operations:
    - Identify transitions into and out of helipad zones
    - Extract the entry and exit times
    - Keep only entries and exits
    - Create a column for UTC_out of landing zone
    - Keep only the zone dwellings with time longer than min_dwell_time
    - Clean up the primary hospital row
    Parameters:
    - d (geopandas.geodataframe.GeoDataFrame): DataFrame containing flights
    - min_dwell_time (int): Minimum time (in minutes) for a zone dwelling to be considere
    Returns:
    - cleaned_data (geopandas.geodataframe.GeoDataFrame): Cleaned DataFrame
    """
    # Identify transitions into and out of helipad zones
    d_entries_and_exits = d.copy()
    d_entries_and_exits['in_helipad_zone'] = ~d_entries_and_exits['zone_name'].isna()
    d_entries_and_exits['zone_change'] = d_entries_and_exits['in_helipad_zone'].ne(d_entries_and_exits['in_helipad_zone'].shift())
    # Extract the entry and exit times
    d_entries_and_exits['entry'] = (d_entries_and_exits['zone_change']) & (d_entries_and_exits['in_helipad_zone'])
    d_entries_and_exits['exit'] = (d_entries_and_exits['zone_change']) & (~d_entries_and_exits['in_helipad_zone'])
    d_entries_and_exits['UTC_str'] = d_entries_and_exits['UTC'].astype(str)

    # Keep only entries and exits
    d_entries_and_exits = d_entries_and_exits[d_entries_and_exits['entry'] | d_entries_and_exits['exit']]
    # Create column for UTC_out of landing zone
    d_entries_and_exits['UTC_out'] = d_entries_and_exits.groupby('aircraft_id')['UTC'].shift(-1)
    # Keep only the zone dwellings with time longer than min_dwell_time
    d_entries_and_exits['time_in_zone'] = (d_entries_and_exits['in_helipad_zone'] * (d_entries_and_exits['UTC_out'] - d_entries_and_exits['UTC'])).dt.seconds / 60
    
    # adaptive dwell time
    flyover_filter = d_entries_and_exits['time_in_zone'] > d_entries_and_exits['dwell_time']
    d_entries_and_exits_no_flyovers = d_entries_and_exits[flyover_filter]

    # Clean up primary hospital row
    d_entries_and_exits_no_flyovers['is_primary_hospital'] = d_entries_and_exits_no_flyovers['is_primary_hospital'].map({1: True, 0: False})
    return d_entries_and_exits_no_flyovers

def create_transfer_dataframe(d: pd.DataFrame, max_transit_time: int = 3, remove_outliers: bool = False, outlier_factor: int = 2, outlier_offset: int = 5) -> pd.DataFrame:
    """
    Create a transfer dataframe by matching primary hospital landings with corresponding university hospital landings.

    Args:
        d (pd.DataFrame): The input dataframe containing hospital landing data.
        max_transit_time (int): The maximum transit time in hours for matching landings.
        remove_outliers (bool, optional): Whether to remove outliers in transit time. Defaults to False.
        outlier_factor (int, optional): The factor used to determine the upper threshold for transit time outliers. Defaults to 2.
        outlier_offset (int, optional): The offset used to determine the upper threshold for transit time outliers. Defaults to 5.

    Returns:
        pd.DataFrame: The transfer dataframe containing matched landings and calculated travel information.
    """
    d_transfer = d.copy()
    d_transfer.reset_index(inplace=True)
    d_transfer['transfer_id'] = np.nan  # Initialize the transfer_id column
    
    # Filter for primary and university hospital landings
    d_primary = d_transfer[d_transfer['is_primary_hospital']]
    d_tertiary = d_transfer[~d_transfer['is_primary_hospital']]
    
    transfer_id = 0
    
    # Iterate over each primary hospital landing
    for idx_primary, row_primary in d_primary.iterrows():
        # Find matching university hospital landings for the same aircraft within the time window
        valid_tertiary_landings = d_tertiary[
            (d_tertiary['aircraft_id'] == row_primary['aircraft_id']) &
            (d_tertiary['UTC'] >= row_primary['UTC_out']) &
            (d_tertiary['UTC'] <= row_primary['UTC_out'] + pd.Timedelta(hours=max_transit_time))
        ]
        
        if not valid_tertiary_landings.empty:
            # Pick the first valid university landing
            first_valid_idx = valid_tertiary_landings.index[0]
            transfer_id += 1
            d_transfer.loc[idx_primary, 'transfer_id'] = transfer_id
            d_transfer.loc[first_valid_idx, 'transfer_id'] = transfer_id
    
    d_transfer.dropna(subset=['transfer_id'], inplace=True)
    
    # Split into sending and receiving dataframe
    d_transfers_sending = d_transfer[d_transfer['is_primary_hospital']]
    d_transfer_receiving = d_transfer[~d_transfer['is_primary_hospital']]
    
    # Left join in transfer id
    d_transfers_merged = pd.merge(d_transfers_sending, d_transfer_receiving, on='transfer_id', suffixes=('_sending', '_receiving'))
    
    # Calculate travel distance in km
    d_transfers_merged['estimated_distance'] = d_transfers_merged['geometry_sending'].to_crs("EPSG:32634").distance(d_transfers_merged['geometry_receiving'].to_crs("EPSG:32634")) / 1000 
    
    # Calculate expected travel time, account for zone radius
    d_transfers_merged['expected_transit_time'] = (d_transfers_merged['estimated_distance'] / 250) * 60
    
    # Clean up columns
    columns_to_keep = ['transfer_id', 'hospital_name_sending', 'hospital_name_receiving', 'year_sending', 'reg_sending', 'UTC_sending', 'UTC_out_sending', 'time_in_zone_sending', 'UTC_receiving', 'zone_name_sending', 'zone_name_receiving', 'radius_sending', 'radius_receiving', 'geometry_sending', 'geometry_receiving', 'estimated_distance', 'expected_transit_time', 'flight_id_receiving', 'aircraft_id_receiving']
    d_transfers_merged = d_transfers_merged[columns_to_keep]
    
    # Calculate transit time (this will be the time from exiting a zone to entering a zone)
    d_transfers_merged['transit_time'] = (d_transfers_merged['UTC_receiving'] - d_transfers_merged['UTC_out_sending']).dt.total_seconds() / 60
    
    d_transfers_merged['transit_time_outlier'] = np.where(
        d_transfers_merged['transit_time'] > (d_transfers_merged['expected_transit_time'] * outlier_factor + outlier_offset), 
        True, 
        False
    )
    
    d_transfers_merged['transit_time_ratio'] = d_transfers_merged['transit_time'] / d_transfers_merged['expected_transit_time']
    
    if remove_outliers:
        d_transfers_merged = d_transfers_merged[d_transfers_merged['transit_time_outlier'] == False]
    
    return d_transfers_merged

def find_transfers(d, max_transit_time=3, remove_outliers=False, outlier_factor=2, outlier_offset=5):
    """
    Finds transfers in a given dataset.

    Parameters:
    - d: The dataset containing entries and exits.
    - min_dwell_time: The minimum amount of time a person needs to spend at a location to be considered an entry or exit.
    - max_transit_time: The maximum amount of time allowed between an exit and the subsequent entry to be considered a transfer.
    - remove_outliers: A flag indicating whether to remove outliers from the transfer dataframe.
    - outlier_factor: The factor used to determine outliers in transit time.
    - outlier_offset: The offset used to determine outliers in transit time.

    Returns:
    - final_df: The dataframe containing the transfers.

    Example usage:
    ```
    dataset = load_dataset()
    transfers = find_transfers(dataset, min_dwell_time=10, max_transit_time=2, remove_outliers=True, outlier_factor=2, outlier_offset=5)
    print(transfers)
    ```

    """
    d_entries_and_exits = extract_entries_and_exits(d)
    final_df = create_transfer_dataframe(d=d_entries_and_exits, max_transit_time=max_transit_time, outlier_factor=outlier_factor, outlier_offset=outlier_offset, remove_outliers=remove_outliers)
    return final_df