'''
This version uses a new set of features, instead of using best track directly.

Command line arguments:
    -v, --verbose: if present, enables verbose output
    --past-track-limit xx: specify an int (xx) to control the time sequence length per sample

dependencies: pandas, numpy, geographiclib, merge_baseline_dataset.py
'''
import argparse
import numpy as np
import pandas as pd
from dataset_utils import read_numpy_from_file, numpy_to_pandas, output_pandas_to_file
from geographiclib.geodesic import Geodesic
from merge_baseline_dataset import get_72h_signal_issuance, distance_to_HK, check_direct_strike

def azimuth_from_HK(lat: float, long: float):
    '''Takes in a latitude and a longitude, returns its forward azimuth from Hong Kong.'''
    # constants
    hko_lat = 22.302219
    hko_long = 114.174637
    azimuth = Geodesic.WGS84.Inverse(hko_lat, hko_long, lat, long)['azi1']
    # the given azimuth is in [-180, 180], I prefer [0, 360] instead
    return (azimuth + 360.0) % 360.0

def build_baseline_ds(issuance: pd.DataFrame, best_track: pd.DataFrame, verbose:bool=True, past_track_limit:int=24):
    '''
    Builds the baseline dataset.
    
    Paramters:
        issuance (pandas dataframe): HKO issuance records dataframe
        best_track (pandas dataframe): best track data dataframe
        verbose (bool): verbosity flag, defaults to True
        past_track_limit (int): number of hours to include in the time sequence, defaults to 24
    
    Returns:
        ds (pandas dataframe): baseline dataset.

    Columns and data types of the combined set, 9/1/2022:
    0 : int MM, d/t of the *last* record in sequence
    1 : int DD
    2 : bool MINIMAL_IMPACT, T1 ground truth
    3 : bool LIMITED_IMPACT, T3 ground truth
    4 : bool SUBSTANTIAL_IMPACT, T8-T10 ground truth
    5 : bool DIRECT_STRIKE, 100km range reached
    6+ : a time series of length = (past_track_limit // 6). Each item in the series is a six-tuple
           (float DISTxx, float AZMxx, int SPEEDxx, int DIRxx, int VMAXxx, int DVMAXxx)
         indicating the radial distance and azimuth of the storm from HK, storm speed, storm heading bearing,
         storm intensity and the change in storm intensity.
         The symbol xx stands for the number of hours behind the start of the sequence,
         i.e. xx in [0, 6, ..., past_track_limit]. DVMAX${past_track_limit} is not available.
    '''
    
    # define column names
    columns = [
        'MM', 'DD', 
        # rename columns so they are short enough to manoeuvre with
        'LOW_IMPACT', 'MID_IMPACT', 'BIG_IMPACT', 'DIRECT_STRIKE'
    ]
    for i in range(0, past_track_limit+6, 6):
        # new system
        columns.append('DIST{0:02d}'.format(i))
        columns.append('AZM{0:02d}'.format(i))
        columns.append('SPEED{0:02d}'.format(i))
        columns.append('DIR{0:02d}'.format(i))
        columns.append('VMAX{0:02d}'.format(i))

        if i != past_track_limit:
            columns.append('DVMAX{0:02d}'.format(i))

    storms = best_track['SID'].unique()
    print('Number of storms identified in the dataset: {0}'.format(storms.shape[0]))
    dataArray = []
    print("Combining best track data sequences with warning records.")
    if verbose: 
        num_storms_processed = 0
    for storm in storms:
        same_storm = best_track.query('SID == @storm', inplace=False)
        same_storm = same_storm.reset_index(drop=True)
        
        # generate data rows for the storm
        for index, row in same_storm.iloc[past_track_limit//6:].iterrows(): # skip to the first 24 hour time step
            
            if row['SEASON'] < 1946: continue # no TC warning signals records before 1946 in our data source
            
            data_row = []
            # time information (t)
            data_row.append(row['ISO_TIME_MONTH'])
            data_row.append(row['ISO_TIME_DAY'])

            # compute and feed in ground truth
            # get signal issuance: already considered next 72 hours
            # but TC w/o names need further consideration
            signal1 = get_72h_signal_issuance(
                row['NAME'], row['SEASON'], row['ISO_TIME_MONTH'], 
                row['ISO_TIME_DAY'], row['ISO_TIME_HOUR'], issuance, 1)
            signal3 = get_72h_signal_issuance(
                row['NAME'], row['SEASON'], row['ISO_TIME_MONTH'], 
                row['ISO_TIME_DAY'], row['ISO_TIME_HOUR'], issuance, 3)
            signal8 = get_72h_signal_issuance(
                row['NAME'], row['SEASON'], row['ISO_TIME_MONTH'], 
                row['ISO_TIME_DAY'], row['ISO_TIME_HOUR'], issuance, 8)
            # direct strike
            direct_strike = check_direct_strike(row['USA_LAT'], row['USA_LON'])
            
            # select next 72 hours to check:
            #   - if a nameless TC enters 800km radius of HK (likely the signal issuance values above are correct)
            #   - if a TC that is not in 800km radius of HK right now will enter that radius
            #   - if the direct strike situation will change (i.e. TC gets within 100km)
            # if TC dissipates (no more records) in the next 72 hours, just consider as far ahead as possible.
            if row['NAME'] == 'NOT_NAMED' or not direct_strike:
                if (index + 12) > same_storm.shape[0]:
                    next_data = same_storm.iloc[index+1:] # choose the rest
                else:
                    next_data = same_storm.iloc[index+1: index+13]
                enter_HK_800_km = False
                for idx, item in next_data.iterrows():
                    # check if distance ever decreased
                    dist = distance_to_HK(item['USA_LAT'], item['USA_LON'])
                    if dist <= 800:
                        enter_HK_800_km = True
                        # check if direct strike ground truth changed
                        if dist <= 100:
                            direct_strike = True
                if row['NAME'] == 'NOT_NAMED' and not enter_HK_800_km:
                    signal1 = False; signal3 = False; signal8 = False                    

            data_row.append(signal1)
            data_row.append(signal3)
            data_row.append(signal8)
            data_row.append(direct_strike)   
            # if direct_strike: print(row['SEASON'], row['NAME']) # for manual verification of correctness

            # best track location t-0 to t-24, i.e. the past 24/6 = 4 records with the current one
            cannot_build_past_track = False
            for hh in range(0, (past_track_limit//6)+1):
                if same_storm.iloc[index-hh]['USA_WIND'] < 0:
                    cannot_build_past_track = True
                    break # skip this whole record this wind speed value is not available

                # new system
                data_row.append(distance_to_HK(same_storm.iloc[index-hh]['USA_LAT'], same_storm.iloc[index-hh]['USA_LON']))
                azimuth = azimuth_from_HK(same_storm.iloc[index-hh]['USA_LAT'], same_storm.iloc[index-hh]['USA_LON'])
                data_row.append(azimuth)
                data_row.append(same_storm.iloc[index-hh]['STORM_SPEED'])
                data_row.append(same_storm.iloc[index-hh]['STORM_DIR'])
                vmax = same_storm.iloc[index-hh]['USA_WIND']
                data_row.append(vmax)
                if hh != (past_track_limit//6):
                    old_vmax = same_storm.iloc[index-hh-1]['USA_WIND']
                    data_row.append(vmax - old_vmax)

            if not cannot_build_past_track:
                dataArray.append(data_row) 
            elif verbose and cannot_build_past_track:
                print("A record of TC {0} ({1}) contains empty USA_WIND data and is abandoned.".format(row['NAME'], row['SEASON']))

        if verbose:
            num_storms_processed += 1
            if num_storms_processed % 200 == 0:
                print("Processed {0} storms out of {1}.".format(num_storms_processed,storms.shape[0]))      
    
    print()
    ds = pd.DataFrame(dataArray, columns=columns)
    print('Number of TC best track - warning record sequences: {0}'.format(ds.shape[0]))
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose',help="Enable verbose output",action='store_true')
    parser.add_argument('--past-track-limit', type=int, default=24, choices=range(6,80,6), \
        help="Number of hours of past position/intensity values per time series/record")
    args = parser.parse_args()
    verbose = False
    if args.verbose:
        print("[Verbose mode enabled]")
        verbose = True
    past_track_limit = args.past_track_limit

    # load both HKO and IBTrACS datasets (cleaned) from file
    data_HKO = read_numpy_from_file('issuance.npy')
    data_HKO = numpy_to_pandas(data_HKO)
    if verbose:
        print("HKO TC warning issuance record data:")
        print(data_HKO)
        print(data_HKO.dtypes)
    data_best_track = read_numpy_from_file('best_track.npy')
    data_best_track = numpy_to_pandas(data_best_track)
    if verbose:
        print("IBTrACS best track data:")
        print(data_best_track)
        print(data_best_track.dtypes)

    # combine into one set based on the needs
    print("Building dataset for baseline model.")
    dataset = build_baseline_ds(data_HKO, data_best_track, verbose, past_track_limit)
    print(dataset)
    print(dataset.dtypes)
    # write to file
    output_pandas_to_file(dataset, "baseline_dataset_newvars_{0}.gz".format(past_track_limit))

    return

if __name__ == "__main__":
    main()