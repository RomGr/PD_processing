from penetration_depth.move_annotations import move_annotations
from penetration_depth.mask_analysis import load_data
from penetration_depth.create_plots import generate_plots
from penetration_depth.create_output import create_output_pickle_master
from penetration_depth.save_data_prism import save_data_prism
from penetration_depth.helpers import load_parameters_save
import random
random.seed(42)

def process_PD(path_data: str, measurements_types: list, wavelengths: list, parameters: list, 
               iq_size: int = 90, metric: str = "median", CX_overimposed: bool = False, 
               CC_overimposed: bool = False, small_ROIs: bool = False, max_number_ROIs: int = 15,
               Flag: bool = False):
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
    if CX_overimposed or CC_overimposed:
        print()
        print(f"Moving the annotations and creating the small ROIs...")
        ROIs_GM, ROIs_WM = move_annotations(path_data, max_number_ROIs, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed, small_ROIs = small_ROIs,
                                            Flag = Flag)
        print()
        print(f"Annotations and small ROIs processed...")
        print()
    else:
        ROIs_GM, ROIs_WM = None, None

    
    # iterate over the wavelengths
    for wavelength in wavelengths:
        print()
        print('Processing: ' + wavelength + '...')
        
        # load the data and save the raw data for the different ROIs
        data_measurement = load_data(path_data, measurements_types, wavelength, parameters_save, Flag = Flag, 
                                     iq_size = iq_size, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed,
                                     small_ROIs = small_ROIs)
        
        # generate the plots
        _ = generate_plots(path_data, data_measurement, measurements_types, wavelength, metric = metric,
                                            Flag = Flag, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed,
                                            small_ROIs = small_ROIs)
        
        # create and save the pickle files
        _ = create_output_pickle_master(data_measurement, measurements_types, parameters, 
                                        path_data, wavelength, Flag = Flag, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)
        
        # save the data in a "prism" format
        save_data_prism(measurements_types, path_data, wavelength, parameters, max_number_ROIs, [ROIs_GM, ROIs_WM],
                        CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)
        print()
        print('Processed: ' + wavelength + '\n')
        print()
    
    return data_measurement
        