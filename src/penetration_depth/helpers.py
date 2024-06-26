import os
import json


def load_json(filename):
    """
    loads a json file and returns the data

    Returns
    -------
    filename : str
        the filename of the json file
    """
    with open(filename) as json_file:
        return json.load(json_file)
    
    
def load_plot_parameters():
    """
    load and returns the parameters for the polarimetric parameter plots

    Returns
    -------
    plot_parameters : dict
        the parameters to plot the parameters maps
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return load_json(os.path.join(dir_path, 'data', 'parameters_plot.json'))


def load_parameter_maps():
    """
    load and returns the parameters for the histogram plots

    Returns
    -------
    parameters_map : dict
        the parameters to plot the parameters histograms
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return load_json(os.path.join(dir_path, 'data', 'parameters_map.json'))


def load_match_sequence(measurement_type: str, CX_overimposed: bool = False, CC_overimposed: bool = False):
    """
    load and returns the match sequence and filename mask for the given measurement type
    
    Returns
    -------
    parameters_map : dict
        the parameters to plot the parameters histograms
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = load_json(os.path.join(dir_path, 'data', 'match_sequences.json'))
    if CX_overimposed or CC_overimposed:
        return data[measurement_type[0]]
    else:
        return data[measurement_type]

def get_measurement_numbers(measurement_type: str, CC_overimposed: bool = False, small_ROIs: bool = False):
    """
    load and get the measurement numbers (i.e. number of ROIs) for the given measurement type
    
    Returns
    -------
    measurement_type : str
        the measurement type
    """
    if CC_overimposed:
        return 50
    elif small_ROIs:
        return 20
    else:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = load_json(os.path.join(dir_path, 'data', 'measurement_numbers.json'))
        return data[measurement_type]

def load_parameters_save():
    """
    load and get the parameters and the corresponding labels in the MM
    
    Returns
    -------
    measurement_type : str
        the measurement type
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return load_json(os.path.join(dir_path, 'data', 'parameters_save.json'))

def load_repetitions():
    """
    load and get the parameters and the corresponding labels in the MM
    
    Returns
    -------
    measurement_type : str
        the measurement type
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return load_json(os.path.join(dir_path, 'data', 'repetitions.json'))

def save_repetitions(repetitions):
    """
    load and get the parameters and the corresponding labels in the MM
    
    Returns
    -------
    measurement_type : str
        the measurement type
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data', 'repetitions.json'), 'w') as f:
        json.dump(repetitions, f)