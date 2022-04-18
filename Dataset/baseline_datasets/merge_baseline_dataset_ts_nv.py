'''
This version creates an explicit time series for each sample, using the new variable layout.

Command line arguments:
    -v, --verbose: if present, enables verbose output
    --past-track-limit xx: specify an int (xx) to control the time sequence length per sample

dependencies: pandas, numpy, merge_baseline_dataset.py
'''
import argparse
import numpy as np
import pandas as pd
from dataset_utils import read_numpy_from_file, numpy_to_pandas, output_pandas_to_file
from merge_baseline_dataset import get_72h_signal_issuance, distance_to_HK, check_direct_strike
from merge_baseline_dataset_newvars import azimuth_from_HK # future use
from datetime import datetime, timedelta

def get_immediate_signal_issuance(name: str, year: int, month: int, day: int, hour: int, issuance: pd.DataFrame, signal: int):
    '''
    Checks if the specified signal is being hoisted at the same time given these information:

    Parameters:
        name (str): TC name
        year (int): season
        month (int): month value
        day (int): day value
        hour (int): hour value
        issuance (pandas dataframe): HKO issuance records
        signal (int): the target signal, either 1, 3, or 8.

    Returns:
        result (bool): True if hoisted right now, False otherwise
    '''

    # handle name anomalies: merged storms and poorly tracked storms due to primitive technology
    if name == 'FLOSSIE:GRACE': return False # 1950/1966/1969, never affected HK despite the 3 incarnations
    elif name == 'LUCRETIA:NANCY': return False # 1950, same as above
    elif name == 'JEANNE:JEANNIE': return False # 1952, same as above, different names for the same TC
    elif name == 'KAREN:LUCILLE': return False # 1956, these two merged approx. 900km away from HK and never affected HK
    elif name == 'ANITA:WILDA': name = 'WILDA' # 1959, WILDA gave HK a T1
    elif name == 'OPAL:RUTH': return False # 1959, never affected HK
    elif name == 'NORA:PATSY': name = 'NORA' # 1959, NORA gave HK a nice T3
    elif name == 'BABE:BABS:CARLA' or name == 'BABE:CARLA:CHARLOTTE:CARLA': return False # 1962, never affected HK, naming shenanigans intensify
    elif name == 'EMMA:FREDA' or name == 'FREDA:GILDA': return False # 1962, never affected HK. They must have had a hard time tracking close TCs in 1962
    elif name == 'GILDA:IVY': return False # 1962, separate line in memory of Ivy who was murdered by the cannibal Gilda
    elif name == 'FRAN:GEORGIA' and year == 1964: name = 'GEORGIA' # 1964, GEORGIA gave HK a T1
    elif name == 'LOUISE:MARGE': return False # 1964, never affected HK, merged
    elif name == 'IVY:JEAN':  return False # 1965, never affected HK
    elif name == 'FRAN:GEORGIA' and year == 1967: name = 'FRAN' # 1967, this cursed pair again. FRAN led to TWO separate T1's.
    elif name == 'BILIE:BILLIE': return False # 1970, this never affected HK, different names for the same thing
    elif name == 'FAYE(GLORIA):GLORIA': return False # 1971, never affected HK, Faye annexed by Gloria early on
    elif name == 'HELEN:HELLEN': return False # 1975, never affected HK, different names
    elif name == 'BESS:BONNIE': name == 'BONNIE' # 1978, T3 in August
    elif name == 'TESS:VAL': name = 'TESS' # 1982, T3, VAL succeeded TESS
    elif name == 'KEN-LOLA:LOLA': return False # 1989, poorly organized TC, hit Shanghai but not HK
    elif name == 'PAT:RUTH': return False # 1994, did not affect HK, merged
    elif name == 'ABEL:BETH': name = 'BETH' # 1996, Beth gave HK a T1
    elif name == 'ORAJI:TORAJI': return False # 2018, but Oraji is a typo of Toraji
    elif name == 'BULBUL:MATMO': return False # 2019, Matmo went to the Indian Ocean and became Bulbul. No impact on HK

    if name == 'NOT_NAMED': name = 'no name' # naming conventions difference

    # select relevant HKO records and then narrow down the search
    if signal == 1:
        signals = [1]
    elif signal == 3:
        signals = [3]
    elif signal == 8:
        signals = [8, 9, 10]
    else: return None
    records = issuance.query("(Name == @name) and (StartYY == @year) and (Signal in @signals)")
    if records.shape[0] < 1: return False # this TC never led to any signals

    time_to_check = datetime(year, month, day, hour, 0)
    # for each potential match, check if the given current time is within the start-end period
    for _, row in records.iterrows():
        startTime = datetime(year, row['StartMM'], row['StartDD'], row['StartHH'], row['Startmm'])
        endTime = datetime(year, row['EndMM'], row['EndDD'], row['EndHH'], row['Endmm'])

        if startTime <= time_to_check <= endTime:
            return True

    return False


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

    Columns and data types of the combined set:
        The first four columns constitute the correct labels.
        0 : bool MINIMAL_IMPACT, T1 ground truth
        1 : bool LIMITED_IMPACT, T3 ground truth
        2 : bool SUBSTANTIAL_IMPACT, T8-T10 ground truth
        3 : bool DIRECT_STRIKE, direct strike ground truth

        Then there will be n = (past_track_limit // 6) repetitions of the following as a time series:

        0 : int MMxx, d/t and hour of this record in the series
        1 : int DDxx
        2 : int HHxx
        3 : bool MI_STATUSxx, whether T1 is hoisted at this point (MM/DD HH)
        4 : bool LI_STATUSxx, whether T3 is hoisted at this point
        5 : bool SI_STATUSxx, whether T8-T10 is hoisted at this point
        6 : bool DS_STATUSxx, whether there is a direct strike at this point 
        7 : float DISTxx, the radial distance of the storm from HK at this point
        8 : float AZMxx, azimuth of the storm from HK
        9 : int SPEEDxx, storm speed
        10 : float DIRxx, storm heading bearing
        11 : int VMAXxx, storm intensity
        12 : int DVMAXxx, change in storm intensity, unavailable for the oldest record in sequence.

        The symbol xx stands for the number of hours behind the start of the sequence,
        i.e. xx in [0, 6, ..., past_track_limit].
    '''

    # define column names
    columns = [
        # rename columns so they are short enough to work with
        'LOW_IMPACT', 'MID_IMPACT', 'BIG_IMPACT', 'DIRECT_STRIKE'
    ]
    for i in range(0, past_track_limit+6, 6):
        columns.append('MM{0:02d}'.format(i))
        columns.append('DD{0:02d}'.format(i))
        columns.append('HH{0:02d}'.format(i)) # kept in order to allow for comparison and better identification of records 

        columns.append('MI_STATUS{0:02d}'.format(i))
        columns.append('LI_STATUS{0:02d}'.format(i))
        columns.append('SI_STATUS{0:02d}'.format(i))
        columns.append('DS_STATUS{0:02d}'.format(i))

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
    print("Creating time series from best track data sequences and warning records.")
    if verbose: 
        num_storms_processed = 0
    
    for storm in storms:
        same_storm = best_track.query('SID == @storm', inplace=False)
        same_storm = same_storm.reset_index(drop=True)

        # generate data rows for the storm
        for index, row in same_storm.iloc[past_track_limit//6:].iterrows(): # skip to the first 24 hour time step
            
            if row['SEASON'] < 1946: continue # no TC warning signals records before 1946 in our data source
            
            data_row = []

            ### true labels ###
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
            # direct strike (right now)
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

            ### Time series from t-0 to t-24, i.e. the past 24/6 = 4 records with the current one
            cannot_build_past_track = False
            for hh in range(0, (past_track_limit//6)+1):
                if same_storm.iloc[index-hh]['USA_WIND'] < 0:
                    cannot_build_past_track = True
                    break # skip this whole record this wind speed value is not available

                # time information
                data_row.append(same_storm.iloc[index-hh]['ISO_TIME_MONTH'])
                data_row.append(same_storm.iloc[index-hh]['ISO_TIME_DAY'])
                data_row.append(same_storm.iloc[index-hh]['ISO_TIME_HOUR'])        

                # get current impact status
                signal1 = get_immediate_signal_issuance(
                    same_storm.iloc[index-hh]['NAME'], same_storm.iloc[index-hh]['SEASON'], same_storm.iloc[index-hh]['ISO_TIME_MONTH'], 
                    same_storm.iloc[index-hh]['ISO_TIME_DAY'], same_storm.iloc[index-hh]['ISO_TIME_HOUR'], issuance, 1)
                data_row.append(signal1)
                signal3 = get_immediate_signal_issuance(
                    same_storm.iloc[index-hh]['NAME'], same_storm.iloc[index-hh]['SEASON'], same_storm.iloc[index-hh]['ISO_TIME_MONTH'], 
                    same_storm.iloc[index-hh]['ISO_TIME_DAY'], same_storm.iloc[index-hh]['ISO_TIME_HOUR'], issuance, 3)
                data_row.append(signal3)
                signal8 = get_immediate_signal_issuance(
                    same_storm.iloc[index-hh]['NAME'], same_storm.iloc[index-hh]['SEASON'], same_storm.iloc[index-hh]['ISO_TIME_MONTH'], 
                    same_storm.iloc[index-hh]['ISO_TIME_DAY'], same_storm.iloc[index-hh]['ISO_TIME_HOUR'], issuance, 8)
                data_row.append(signal8)
                direct_strike = check_direct_strike(same_storm.iloc[index-hh]['USA_LAT'], same_storm.iloc[index-hh]['USA_LON'])
                data_row.append(direct_strike)

                # TC best track predictors
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
    output_pandas_to_file(dataset, "baseline_dataset_tsnv_{0}.gz".format(past_track_limit))

    return

if __name__ == "__main__":
    main()