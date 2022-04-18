'''
Cleans up, filters, converts and prepares the best track data, which are written to ./best_track.npy.

Command line options:
    -v, --verbose : if present, enables verbosity.

dependencies: numpy
'''

import numpy as np
import csv
import argparse
from datetime import datetime
import pytz
from dataset_utils import output_numpy_to_file
import numpy.lib.recfunctions as rf

def read_dataset_as_csv(filename='_ibtracs.csv'):
    '''Takes in a CSV file name and returns its contents as a list.'''

    # data = np.genfromtxt(filename,delimiter=',', skip_header=2)
    print("Reading file {0}... ".format(filename), end='')
    with open(filename, 'r') as file:
        data = list(csv.reader(file, delimiter=','))
    print("Done!")
    return data

def cleanup_data(data, verbose=True):
    '''Takes in a data list and a verbosity flag, returns the data filtered and subsetted as a numpy array'''

    # remove CSV headers
    data = data[2:]
    # convert to numpy array, for better processing
    ds = np.array(data)
    print("Full unprocessed dataset: {0} rows and {1} variables.".format(ds.shape[0], ds.shape[1]))

    # select JTWC records
    mask_jtwc = ds[:,17] == 'jtwc_wp'
    ds_jtwc = ds[mask_jtwc,:]
    print("JTWC subset selected.")
    
    if verbose:
        # the other sources around for reference
        mask_hko = ds[:,60] != ' ' # columns 60-64 (HKO)
        ds_hko = ds[mask_hko,:]
        print(">>>> HKO subset contains {0} rows and {1} columns.".format(ds_hko.shape[0], ds_hko.shape[1]))
        mask_jma = ds[:,43] != ' ' # columns 43-54 (TOKYO)
        ds_jma = ds[mask_jma,:]
        print(">>>> JMA subset contains {0} rows and {1} columns.".format(ds_jma.shape[0], ds_jma.shape[1]))
        mask_cma = ds[:,55] != ' ' # columns 55-59 (CMA)
        ds_cma = ds[mask_cma,:]
        print(">>>> CMA subset contains {0} rows and {1} columns.".format(ds_cma.shape[0], ds_cma.shape[1]))
        mask_wmo = ds[:,12] != ' ' # columns 9-13 (WMO)
        ds_wmo = ds[mask_wmo,:]
        print(">>>> WMO subset contains {0} rows and {1} columns.".format(ds_wmo.shape[0], ds_wmo.shape[1]))
    
    # select needed columns only    
    print("Cleaning JTWC records......")
    '''
    usable columns include: 
    > 1 (SID), 2 (SEASON), 3 (NUMBER), 6 (NAME), 7 (ISO_TIME), 8 (NATURE), 14 (TRACK_TYPE), 17 (IFLAG), 
      20-21, 24-25, 27-38 (USA data), 162 (STORM_SPEED), 163 (STORM_DIR).
    > GUST (152), SEAHGT and SEARAD (157-161) might be of interest, will revisit later.
    That is, drop columns 4-5, 9-13, 15-16, 18-19, 22-23, 26, 39-161. Some of the remaining columns can be dropped
    later, as they are only kept at the moment for bookkeeping purposes.
    Note that the above uses 1-indexing, so we should drop 3-4, 8-12, 14-15, 17-18, 21-22, 25, 38-160.
    '''
    columns_to_drop = [3,4] + list(range(8,13)) + [14,15,17,18,21,22, 25] + list(range(38,161))
    ds_jtwc = np.delete(ds_jtwc, columns_to_drop, 1)    
    print("JTWC subset with columns removed: {0} rows and {1} variables.".format(ds_jtwc.shape[0], ds_jtwc.shape[1]))
    '''
    Column numbers and names after selection:
    0 SID
    1 SEASON
    2 NUMBER (to be removed later)
    3 NAME
    4 ISO_TIME
    5 NATURE (to be removed later)
    6 TRACK_TYPE (to be removed later)
    7 IFLAG (to be removed later)
    8-9 USA_LAT and USA_LON 
    10-11 USA_WIND and USA_PRES
    12-23 wind radii (USA_Rxx_yy: xx in {34, 50, 64} and yy in {NE, SE, NW, SW})
    24-25 STORM_SPEED and STORM_DIR
    '''

    # remove records that are irrelevant
    # for column 5 (zero-indexed, NATURE), only select those with value TS (tropical)
    mask_nature = ds_jtwc[:,5] == 'TS'
    ds_jtwc = ds_jtwc[mask_nature,:]
    print(">> After NATURE filtered: {0} rows and {1} variables.".format(ds_jtwc.shape[0], ds_jtwc.shape[1]))
    # for column 6 (TRACK_TYPE), only select those with value MAIN (quality data, not provisional)
    mask_tracktype = ds_jtwc[:,6] == 'main'
    ds_jtwc = ds_jtwc[mask_tracktype,:]
    print(">> After TRACK_TYPE filtered: {0} rows and {1} variables.".format(ds_jtwc.shape[0], ds_jtwc.shape[1]))
    # manage column 7 (IFLAG) which tells interpolation status. select anything that is not '_' (missing).
    if verbose:    
        mask_iflag = [True if item[0] == 'P' else False for item in ds_jtwc[:,7]]
        print(">>>> Number of interpolated records: {0}".format(sum(mask_iflag)))
        mask_iflag = [True if item[0] == 'I' else False for item in ds_jtwc[:,7]]
        print(">>>> Number of intensity interpolated records: {0}".format(sum(mask_iflag)))
        mask_iflag = [True if item[0] == 'V' else False for item in ds_jtwc[:,7]]
        print(">>>> Number of other interpolated records: {0}".format(sum(mask_iflag)))
        mask_iflag = [True if item[0] == '_' else False for item in ds_jtwc[:,7]]
        print(">>>> Number of missing records: {0}, the first one being:".format(sum(mask_iflag)))
        print(ds_jtwc[mask_iflag,:][0])
        mask_iflag = [True if item[0] == 'O' else False for item in ds_jtwc[:,7]]
        print(">>>> Number of original records: {0}".format(sum(mask_iflag)))
    
    mask_iflag = [True if item[0] != '_' else False for item in ds_jtwc[:,7]]
    ds_jtwc = ds_jtwc[mask_iflag,:]
    print(">> After IFLAG filtered: {0} rows and {1} variables.".format(ds_jtwc.shape[0], ds_jtwc.shape[1]))
    
    # bookkeeping for later use
    # check how many records have wind radii (cols 12-23).
    def is_non_empty(row):
        for i in range(12, 24):
            if row[i] == ' ':
                return False
        return True
    mask_radii = np.apply_along_axis(is_non_empty, axis=1, arr=ds_jtwc)
    print(">> Number of records with wind radii estimates is {0}".format(sum(mask_radii)))
    if verbose:
        print(">>>> Sample data (first with radii estimates available):")
        print(ds_jtwc[mask_radii,:][0])
    # check number of records with wind speeds and MSLP
    mask_vmax = ds_jtwc[:,10] != ' '
    print(">> Number of records with USA_WIND is {0}".format(sum(mask_vmax)))
    mask_mslp = ds_jtwc[:,11] != ' '
    print(">> Number of records with USA_PRES is {0}".format(sum(mask_mslp)))
    # check how many records have storm direction and storm velocity (last two columns)
    mask_stormspeed = ds_jtwc[:,24] != ' '
    print(">> Number of records with STORM_SPEED is {0}".format(sum(mask_stormspeed)))
    mask_stormdir = ds_jtwc[:,25] != ' '
    print(">> Number of records with STORM_DIR is {0}".format(sum(mask_stormdir)))
    # compute distance between two consecutive track records
    sample0 = ds_jtwc[0,:]
    sample1 = ds_jtwc[1,:]
    date0 = datetime.strptime(sample0[4], '%Y-%m-%d %H:%M:%S')
    date1 = datetime.strptime(sample1[4], '%Y-%m-%d %H:%M:%S')
    dt = (date1 - date0).seconds//3600
    print(">> Time difference between consecutive records is {0} hours".format(dt))
    # remove columns that we do not need
    columns_to_drop = [2,5,6,7]
    ds_jtwc = np.delete(ds_jtwc, columns_to_drop, 1)
    '''
    Columns and the data types they are:
    0 - str SID
    1 - int SEASON
    2 - str NAME
    3 - timestamp ISO_TIME: UTC in YYYY-MM-DD HH:mm:ss
    4-5 - float USA_LAT and USA_LON: degrees N and E
    6-7 - int USA_WIND and USA_PRES: knots and mb
    8-19 - int wind radii (USA_Rxx_yy: xx in [34, 50, 64] and yy in [NE, SE, SW, NW]): nmile
    20 - int STORM_SPEED: knots
    21 - int STORM_DIR: degrees clockwise from N
    '''

    print("...... Cleaning complete.")
    return ds_jtwc

def convert_data_types(dataset, verbose=True):
    '''Takes in a dataset as a numpy array and a verbosity flag, returns the dataset with converted columns as a numpy array.'''

    if verbose:
        print(dataset)
    print("Previous dataset data types: {0}".format(dataset.dtype))
    dataset = dataset.tolist()

    converted_set = []
    for row in dataset:
        converted_row = []
        for i in range(len(row)):
            if i == 0 or i == 2: # string type
                converted_row.append(row[i])
            elif i == 3: # timestamp
                timestamp = datetime.strptime(row[i], '%Y-%m-%d %H:%M:%S')
                # timestamp = timestamp.replace(tzinfo=pytz.UTC) # no need timezones for modern numpy
                # obtain MM, DD, HH and mm components                
                converted_row.append(timestamp.month)
                converted_row.append(timestamp.day)
                converted_row.append(timestamp.hour)
                converted_row.append(timestamp.minute)
            elif i == 4 or i == 5: # float type
                if row[i] == " ":
                    converted_row.append(None) # NaN
                else: converted_row.append(float(row[i]))
            else: # default to integers
                if row[i] == " ":
                    converted_row.append(-99999) # placeholder
                else: converted_row.append(int(row[i]))
        converted_set.append(tuple(converted_row))

    '''
    Finalized columns and data types:
    0 - str SID
    1 - int SEASON
    2 - str NAME
    3-6 - int ISO_TIME: MM, DD, HH and mm in integers, year and seconds dropped due to irrelevance
    7-8 - float USA_LAT and USA_LON: degrees N and E
    9-10 - int USA_WIND and USA_PRES: knots and mb
    11-22 - int wind radii (USA_Rxx_yy: xx in [34, 50, 64] and yy in [NE, SE, SW, NW]): nmile
    23 - int STORM_SPEED: knots
    24 - int STORM_DIR: degrees clockwise from N
    '''
    dt = np.dtype([
        ('SID', 'U32'),
        ('SEASON', 'i'), 
        ('NAME', 'U32'),
        # ('ISO_TIME', 'datetime64[s]'),
        ('ISO_TIME_MONTH', 'i'), ('ISO_TIME_DAY', 'i'),
        ('ISO_TIME_HOUR', 'i'), ('ISO_TIME_MIN', 'i'),
        ('USA_LAT', 'f'),('USA_LON', 'f'),
        ('USA_WIND', 'i'),('USA_PRES', 'i'),
        ('USA_R34_NE', 'i'),('USA_R34_SE', 'i'),('USA_R34_SW', 'i'),('USA_R34_NW', 'i'),
        ('USA_R50_NE', 'i'),('USA_R50_SE', 'i'),('USA_R50_SW', 'i'),('USA_R50_NW', 'i'),
        ('USA_R64_NE', 'i'),('USA_R64_SE', 'i'),('USA_R64_SW', 'i'),('USA_R64_NW', 'i'),
        ('STORM_SPEED', 'i'),('STORM_DIR', 'i')
    ])
    dataset = np.array(converted_set, dtype=dt)
    if verbose:
        print(dataset)
    print("Processed dataset data types: {0}".format(dataset.dtype))
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose',help="Enable verbose output",action='store_true')
    args = parser.parse_args()
    verbose = False
    if args.verbose:
        print("[Verbose mode enabled]")
        verbose = True

    ds = read_dataset_as_csv()
    ds = cleanup_data(ds, verbose)
    ds = convert_data_types(ds)
    print("Cleaned data shape: {0} rows and {1} variables.".format(ds.shape[0], len(ds.dtype.names)))
    random_index = np.random.choice(ds.shape[0], size=1, replace=False)
    print("Sample data (row {0}):".format(random_index))
    print(ds[random_index])
    print("Sample data (row 1):")
    print(ds[0])

    output_numpy_to_file(ds, 'best_track.npy')
    return

if __name__ == "__main__":
    main()