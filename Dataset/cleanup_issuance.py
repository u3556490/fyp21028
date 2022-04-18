'''
Downloads, converts and cleans up the HKO TC warning signals records database, which is then saved to ./issuance.npy.

dependencies: beautifulsoup4, numpy
setup procedure:
    - conda create --name efyp
    - conda activate efyp
    - conda install beautifulsoup4
    - conda install numpy
'''

from urllib.request import urlopen
from bs4 import BeautifulSoup
import calendar
import numpy
from dataset_utils import output_numpy_to_file, delete_column_from_structured_array

# fetch webpage and convert to BeautifulSoup object
def get_webpage():
    '''Fetches the HKO warning signals database webpage and returns its BeautifulSoup object representation.'''

    print("Fetching data from HKO... ", end='')
    url = "https://www.hko.gov.hk/cgi-bin/hko/warndb_e1.pl?opt=1&sgnl=1.or.higher&start_ym=194601&end_ym=202110"
    page = urlopen(url)
    html_data = page.read().decode("utf-8")
    soup = BeautifulSoup(html_data, "html.parser")
    print("Done!")
    return soup

# convert BeautifulSoup object to Python object holding the extracted issuance data
def parse_soup_to_python_obj(soup):
    '''Takes in a BeautifulSoup object (HKO webpage), returns converted dataset as a list.'''

    # obtain table of data
    # the HKO website is (un)surprisingly poorly made, I could rant for an entire century about this.
    # the following is specifically engineered for the URL above, and is a mess to explain.
    # talk about 90s web design...
    table = soup.find_all('table')[1]
    row = table.find_all('tr')[0]
    table = row.find_all('table')[3]
    table = table.find('table')
    # print(table) # sanity check

    # column names for reference, hardcoded; columns include:
    #   - str Class: originally called "Intensity", this is the categorization of the TC by intensity; dropped later
    #   - str Name: TC name. HKO database does not use other means to identify TCs, making this a major source of headache.
    #   - int Signal: TC warning signal
    #   - int StartHH: Start hours of the signal
    #   - int Startmm: Start minutes of the signal
    #   - int StartDD: Start day
    #   - int StartMM: Start month
    #   - int StartYY: Start year
    #   - int EndHH: End hours of the signal
    #   - int Endmm: End minutes of the signal
    #   - int EndDD: End day
    #   - int EndMM: End month
    #   - int EndYY: End year
    #   - str Duration: Signal duration, hh mm.; dropped later
    columns = [
        # 'Class', 
        'Name', 'Signal', 
        'StartHH', 'StartMM', 'StartDD', 'StartMM', 'StartYY',
        'EndHH', 'EndMM', 'EndDD', 'EndMM', 'EndYY',
        # 'Duration'
    ]

    # helper function to convert month names to month numbers
    def month_name_to_number(name):
        reverse_lookup_table = {month: index for index, month in enumerate(calendar.month_abbr) if month}
        return reverse_lookup_table[name]

    # obtain rows and the data therein
    table_rows = table.find_all('tr')[2:] # start from 2, because there are 2 header rows.
    issuance_dataset = list() # the output Python object will be a list of lists, the nested list holding a row    
    print('Data parsing begins, number of rows found: {0}'.format(len(table_rows)))
    for j in range(len(table_rows)):       
        tr = table_rows[j] 
        table_row_cells = tr.find_all('td')

        data_row = list()
        dst = False # daylight savings time flag, for pre 21 Oct 1979
        for i in range(len(table_row_cells)):
            td = table_row_cells[i]
            text = td.text.replace('\n', '').replace('\xa0', '').rstrip() # text cleanup
            
            # process each column to our liking
            if i < 2: # first two columns need no further processing
                data_row.append(text)
            elif i == 2: # signal
                if '8' in text:
                    # strip out the direction sign in signal #8s
                    data_row.append(8)
                elif '3' in text:
                    # between 1 Jan 1950 and 14 Apr 1956, there is a Local Strong Wind Signal for winds due to TCs *and* monsoons.
                    data_row.append(3)
                else:
                    data_row.append(int(text))
            elif (i == 3) or (i == 5): # start time and end time
                # convert into hh and mm
                tokens = text.split(':')
                data_row.append(int(tokens[0])) # hh
                data_row.append(int(tokens[1])) # mm
            elif (i == 4) or (i == 6): # start date
                tokens = text.split('/')
                # check summer daylight savings time ("S", pre 21 Oct 1979)
                if 'S' in text:
                    dst = True
                    tokens = text[:-2].split('/')
                # break this row (with the DST symbol stripped) into day, month (as a number) and year                
                data_row.append(int(tokens[0])) # dd
                data_row.append(month_name_to_number(tokens[1])) # mm
                data_row.append(int(tokens[2])) # yy
            elif i == 7: # duration
                data_row.append(text)            
            
        # convert DST back to HKT
        if dst:
            # move back 1 hour
            data_row[3] = (int(data_row[3]) + 23) % 24
            data_row[8] = (int(data_row[8]) + 23) % 24
        issuance_dataset.append(tuple(data_row))

        if j % 200 == 0:
            print("Parsed {0} out of {1} records.".format(j, len(table_rows)))

    print("Parsing completed!")
    return issuance_dataset

# convert Python object to numpy array
def data_to_numpy(data):
    '''Takes in the dataset as a python array, returns its numpy structured array representation.'''
    dtype = numpy.dtype([
        ('Class', 'U20'),('Name', 'U20'), ('Signal','i'), 
        ('StartHH', 'i'), ('Startmm','i'), ('StartDD','i'), ('StartMM','i'), ('StartYY','i'),
        ('EndHH','i'), ('Endmm','i'), ('EndDD','i'), ('EndMM','i'), ('EndYY','i'),
        ('Duration', 'S8')
    ])
    return numpy.array(data, dtype=dtype)

# cleans up the data further by removing the two unneeded columns
def cleaned_numpy(data):
    '''
    Takes in the dataset as a numpy array, returns it with two columns fewer.

    Finalized columns:
    0 - str Name: TC name. HKO database does not use other means to identify TCs, making this a major source of headache.
    1 - int Signal: TC warning signal
    2 - int StartHH: Start hours of the signal
    3 - int Startmm: Start minutes of the signal
    4 - int StartDD: Start day
    5 - int StartMM: Start month
    6 - int StartYY: Start year
    7 - int EndHH: End hours of the signal
    8 - int Endmm: End minutes of the signal
    9 - int EndDD: End day
    10 - int EndMM: End month
    11 - int EndYY: End year
    '''
    data = delete_column_from_structured_array(data, 0)
    data = delete_column_from_structured_array(data, 12)
    return data

# main program
def main():
    soup = get_webpage()
    issuance_dataset = parse_soup_to_python_obj(soup)
    
    issuance_dataset = data_to_numpy(issuance_dataset)
    issuance_dataset = cleaned_numpy(issuance_dataset)
    print("Issuance data processed!")
    print("Number of issuance records: {0}".format(issuance_dataset.shape[0]))
    print("Number of variables: {0}".format(len(issuance_dataset.dtype.names)))
    print("Sample data (row 1):")
    print(issuance_dataset[0])
    random_index = numpy.random.choice(issuance_dataset.shape[0], size=1, replace=False)    
    print("Sample data (random row {0}):".format(random_index))
    print(issuance_dataset[random_index])

    output_numpy_to_file(issuance_dataset,'issuance.npy')
    return

# driver code
if __name__ == "__main__":
    main()