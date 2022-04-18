import numpy
import json
import pandas

def output_numpy_to_file(data:numpy.ndarray, filename:str):
    '''Takes in a numpy array and writes it to the specified filename (which must contain extensions) as pickled binary file.'''
    print("Writing to file {0} ... ".format(filename), end='')
    with open(filename, 'wb') as file:
        numpy.save(file, data)
    print("Done!")
    return

def output_json_to_file(data, filename:str):
    '''Takes in a numpy array and writes it to the specified filename (which must contain extensions) as JSON.'''
    print("Writing to file {0} ... ".format(filename), end='')
    json_string = json.dumps(data)
    with open(filename, 'w') as file:
        file.write(json_string)
    print("Done!")
    return

def read_numpy_from_file(filename:str, type:str='npy'):
    '''Takes in a filename, reads it, and returns the numpy array in the file.'''
    if type == 'json':
        return None # not implemented
    print('Reading file {0} ......'.format(filename), end='')
    data = numpy.load(filename, allow_pickle=True)
    print('Done!')
    return data

def numpy_to_pandas(data:numpy.ndarray):
    '''Takes in a numpy array and returns its Pandas DataFrame representation.'''
    df = pandas.DataFrame(data)
    return df

def delete_column_from_structured_array(data, index):
    '''Takes in a structured numpy array and an index, returns the former without the column specified by the index.'''
    names = list(data.dtype.names)
    new_names = names[:index] + names[index+1:]
    result = data[new_names]
    return result

def output_pandas_to_file(data: pandas.DataFrame, filename: str):
    '''Takes in a Pandas DataFrame and writes it to the specified filename (which must contain extensions) as pickles.'''
    print('Reading file {0} ......'.format(filename), end='')
    data.to_pickle(filename)
    print("Done!")
    return