import os
import pandas as pd
import pickle
import copy
import math
from penetration_depth.helpers import load_repetitions

def save_data_prism(measurements_types: list, path_data: str, wavelength: str, parameters: list, max_number_ROIs:int, number_ROIs: list,
                    CX_overimposed: bool = False, CC_overimposed: bool = False, small_ROIs: bool = False):
    """ -----------------------------------------------------------
    # save_data_prism is the master function saving the data to plug them in the prism software
    #
    # Parameters:
    #   measurements_types (list): list of the measurements types to be considered
    #   path_data (str): path to the data folder
    #   wavelength (str): wavelength of the measurements
    #   parameters (list): list of the parameters of interest
    ----------------------------------------------------------- """
    data_parameter, data_parameter_complete_thickness = get_data_df(measurements_types, path_data, wavelength, parameters, max_number_ROIs, number_ROIs,
                                                                    CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed, small_ROIs = small_ROIs)
    
    print()
    print(f"Creating the output file to plug in prism for {wavelength}...")

        
    results_path = os.path.join(path_data, 'results', 'prism_files')
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    results_path = os.path.join(results_path, wavelength)
    try:
        os.mkdir(results_path)
    except FileExistsError:
        pass
    
    # iterate over the parameters
    for parameter in parameters:
        # save the dfs into excel files for the top layer thickness...
        dfs = []
        for _, df in data_parameter[parameter].items():
            dfs.append(df)
        results_path_excel = os.path.join(results_path, parameter + '_prism.xlsx')
        concat = pd.concat(dfs, axis = 1)
        concat = concat.sort_index()
        concat.to_excel(results_path_excel)
        
        # ... and the complete thickness
        dfs = []
        for _, df in data_parameter_complete_thickness[parameter].items():
            dfs.append(df)
        results_path_excel = os.path.join(results_path, parameter + 'real_thickness_prism.xlsx')
        concat = pd.concat(dfs, axis = 1)
        concat = concat.sort_index()
        concat.to_excel(results_path_excel)
        
    print(f"Output file created for {wavelength}...")
    
    
def get_data_df(measurements_types: list, paths_data: str, wavelength, parameters, max_number_ROIs, number_ROIs,
                CX_overimposed: bool = False, CC_overimposed: bool = False, small_ROIs: bool = False):
    """ -----------------------------------------------------------
    # get the data as a dict of dataframes that can be saved and later plugged in the prism software
    #
    # Parameters:
    #   measurements_types (list): list of the measurements types to be considered
    #   paths_data (str): path to the data folder
    #   wavelength (str): wavelength of the measurements
    #   parameters (list): list of the parameters of interest
    ----------------------------------------------------------- """
    data_thickness = {}
    for measurements_type in measurements_types:
        if CX_overimposed or CC_overimposed:
            path = os.path.join(paths_data, 'results', measurements_type[0] + '_' + measurements_type[1], wavelength, 'combined_data_thickness.pickle')
        else:
            path = os.path.join(paths_data, 'results', measurements_type, wavelength, 'combined_data_thickness.pickle')
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        if CX_overimposed or CC_overimposed:
            data_thickness[tuple(measurements_type)] = data
        else:
            data_thickness[measurements_type] = data
        
    data_df = {}
    for measurements_type, vals in data_thickness.items():
        data_measurement_type = {}
        for parameter, val in vals.items():
            val = dict(val)
            if number_ROIs[0] is not None:
                for key, v in val.items():
                    if small_ROIs and not (CX_overimposed or CC_overimposed):
                        repetitions = load_repetitions()
                        max_nb = max_number_ROIs * number_ROIs[1][measurements_type + 'WM'] * repetitions[measurements_type]
                    else:
                        if measurements_type[1]== 'GM':
                            max_nb = number_ROIs[0] * max_number_ROIs
                        else: 
                            max_nb = number_ROIs[1] * max_number_ROIs
                            
                    if len(val[key]) < max_nb:
                        val[key] = val[key] + [math.nan] * (max_nb - len(val[key]))
                
            df = pd.DataFrame({ key: pd.Series(v) for key, v in val.items() }).T
            data_measurement_type[parameter] = df
        data_df[measurements_type] = data_measurement_type
        
    data_parameter = {}
    data_parameter_complete_thickness = {}
    for parameter in parameters:
        data_param = {}
        data_param_complete_thickness = {}
        for measurements_type in measurements_types:
            if CX_overimposed or CC_overimposed:
                data_param[tuple(measurements_type)] = data_df[tuple(measurements_type)][parameter]
                new_data_df = copy.deepcopy(data_df[tuple(measurements_type)][parameter])
            else:
                data_param[measurements_type] = data_df[measurements_type][parameter]
                new_data_df = copy.deepcopy(data_df[measurements_type][parameter])
            
            # add the value of the bottom layer thickness
            if measurements_type == '0_overimposition' or measurements_type == '45_overimposition' or measurements_type == '90_overimposition':
                new_data_df.index = 2 * new_data_df.index
            elif measurements_type == '100+x':
                new_data_df.index = 100 + new_data_df.index
            else:
                assert measurements_type == 'splitted' or CX_overimposed or CC_overimposed
                new_data_df.index = new_data_df.index
                
            if CX_overimposed or CC_overimposed:
                data_param_complete_thickness[tuple(measurements_type)] = new_data_df
            else:
                data_param_complete_thickness[measurements_type] = new_data_df
            
        data_parameter[parameter] = data_param
        data_parameter_complete_thickness[parameter] = data_param_complete_thickness
    
    return data_parameter, data_parameter_complete_thickness