from penetration_depth.mask_analysis import load_data
from penetration_depth.create_plots import generate_plots
from penetration_depth.create_output import create_output_pickle_master
from penetration_depth.save_data_prism import save_data_prism
from penetration_depth.helpers import load_parameters_save
import os
import shutil

def process_PD(path_data: str, measurements_types: list, wavelengths: list, parameters: list, 
               iq_size: int = 90, metric: str = "median", Flag: bool = False, CX_overimposed: bool = False):
    """ -----------------------------------------------------------
    # process the penetration depth data for all the wavelengths given as an input, and save the results, plots and pickle files
    #
    # Parameters:
    #   path_data (str): path to the data
    #   measurements_types (list): list of the different measurement types
    #   parameters_save (list): list of the parameters to save
    #   wavelengths (list): list of the wavelengths of the data
    #   parameters (list): list of the parameters of interest
    #   iq_size (int): size of the IQ (default: 95)
    #   metric (str): metric to use to compute the penetration depth (default: "median")
    #   Flag (bool): flag to display the progress bar (default: False)
    ----------------------------------------------------------- """
    parameters_save = list(load_parameters_save().keys())
    if CX_overimposed:
        move_annotations(path_data)
        
    # get the proportion of azimuth values from the IQ size 
    proportion_azimuth_values = 1 - ((100 - iq_size) * 2)/100
    assert proportion_azimuth_values > 0
    
    # iterate over the wavelengths
    for wavelength in wavelengths:
        print('Processing: ' + wavelength + '...')
        
        # load the data and save the raw data for the different ROIs
        data_measurement = load_data(path_data, measurements_types, wavelength, parameters_save, Flag = Flag, 
                                     iq_size = iq_size, CX_overimposed = CX_overimposed)
        
        # generate the plots
        _ = generate_plots(path_data, data_measurement, measurements_types, wavelength, metric = metric,
                                                        Flag = Flag, CX_overimposed = CX_overimposed)
        
        # create and save the pickle files
        _ = create_output_pickle_master(data_measurement, measurements_types, parameters, 
                                        path_data, wavelength, proportion_azimuth_values = proportion_azimuth_values, 
                                        Flag = Flag, CX_overimposed = CX_overimposed)
        
        # save the data in a "prism" format
        save_data_prism(measurements_types, path_data, wavelength, parameters, CX_overimposed = CX_overimposed)
        print('Processed: ' + wavelength + '\n')
        
    
def move_annotations(path_data_orig):
    path_data = os.path.join(path_data_orig, '90_overimposition')
    for folder in os.listdir(path_data):
        if folder != 'results':
            path_annotation = os.path.join(path_data, folder, 'annotation')

            path_annotation_GM = path_annotation.replace('measurements\90_overimposition', 'CC_GM')
            for file in os.listdir(path_annotation_GM):
                old_name = os.path.join(path_annotation_GM, file)
                new_name = old_name.replace('CC_GM', 'measurements\90_overimposition').replace('.tif', '_GM.tif')
                shutil.copy(old_name, new_name)

            path_annotation_WM = path_annotation.replace('measurements\90_overimposition', 'CC_WM')
            for file in os.listdir(path_annotation_WM):
                old_name = os.path.join(path_annotation_WM, file)
                new_name = old_name.replace('CC_WM', 'measurements\90_overimposition').replace('.tif', '_WM.tif')
                shutil.copy(old_name, new_name)