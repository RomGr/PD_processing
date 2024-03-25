from penetration_depth.mask_analysis import load_data
from penetration_depth.create_plots import generate_plots
from penetration_depth.create_output import create_output_pickle_master
from penetration_depth.save_data_prism import save_data_prism
from penetration_depth.helpers import load_parameters_save
import os, shutil
import cv2, numpy as np
import random
random.seed(42)

def process_PD(path_data: str, measurements_types: list, wavelengths: list, parameters: list, 
               iq_size: int = 90, metric: str = "median", Flag: bool = False, CX_overimposed: bool = False, 
               CC_overimposed: bool = False, small_ROIs: bool = False, max_number_ROIs: int = 15):
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
        ROIs_GM, ROIs_WM = move_annotations(path_data, max_number_ROIs, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed, small_ROIs = small_ROIs)
    else:
        ROIs_GM, ROIs_WM = None, None
    # get the proportion of azimuth values from the IQ size 
    proportion_azimuth_values = 1 - ((100 - iq_size) * 2)/100
    assert proportion_azimuth_values > 0

    
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
                                        path_data, wavelength, proportion_azimuth_values = proportion_azimuth_values, 
                                        Flag = Flag, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)
        
        # save the data in a "prism" format
        save_data_prism(measurements_types, path_data, wavelength, parameters, max_number_ROIs, [ROIs_GM, ROIs_WM],
                        CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed)
        print()
        print('Processed: ' + wavelength + '\n')
        print()
    
    return data_measurement
        
    
def move_annotations(path_data_orig, max_number_ROIs, CX_overimposed: bool = False, CC_overimposed: bool = False, small_ROIs: bool = False):
    path_data = os.path.join(path_data_orig, '90_overimposition')
    for folder in os.listdir(path_data):
        if folder != 'results':
            path_annotation = os.path.join(path_data, folder, 'annotation')

            annotations_GM = []
            if not CC_overimposed:
                path_annotation_GM = path_annotation.replace('measurements\90_overimposition', 'CC_GM')
                for file in os.listdir(path_annotation_GM):
                    old_name = os.path.join(path_annotation_GM, file)
                    new_name = old_name.replace('CC_GM', 'measurements\90_overimposition').replace('.tif', '_GM.tif')
                    if '_5' in file:
                        pass
                    else:
                        shutil.copy(old_name, new_name)
                        annotations_GM.append(new_name)
                        
                if small_ROIs:
                    create_small_ROIs(annotations_GM, max_number_ROIs, CX_overimposed = CX_overimposed, WM = False)  
                
            annotations_WM = []
                
            path_annotation_WM = path_annotation.replace('measurements\90_overimposition', 'CC_WM')
            for file in os.listdir(path_annotation_WM):
                old_name = os.path.join(path_annotation_WM, file)
                new_name = old_name.replace('CC_WM', 'measurements\90_overimposition').replace('.tif', '_WM.tif')
                if '_3' in file and not CC_overimposed:
                    pass
                else:
                    shutil.copy(old_name, new_name)
                    annotations_WM.append(new_name)
            
            if small_ROIs:
                create_small_ROIs(annotations_WM, max_number_ROIs, CX_overimposed = CX_overimposed, WM = True)                    
             
            ROIs_GM = len(annotations_GM)
            ROIs_WM = len(annotations_WM)
    return ROIs_GM, ROIs_WM       
                    
def create_small_ROIs(annotations, max_number_ROIs, CX_overimposed: bool = False, WM: bool = True):
    
    size_ROI = 200
    all_ROIs = []

    path_folder_annotation = os.path.join(annotations[0].split('annotation')[0], 'annotation', 'small_ROIs')
    
    for annotation in annotations:
        image = cv2.imread(annotation, cv2.IMREAD_GRAYSCALE)
        ROIs = np.zeros(image.shape)
        ROIs, ROI_counter = selectROIs(ROIs, image, size_ROI, CX_overimposed = CX_overimposed)

        if ROI_counter > max_number_ROIs:
            numbers = list(range(1, ROI_counter + 1))
            random_elements = random.sample(numbers, max_number_ROIs)
        else:
            random_elements = range(1, ROI_counter + 1)

        for x in range(1, len(random_elements) + 1):
            if np.sum(ROIs == x) != 0:
                all_ROIs.append((ROIs == x) * 255)
    
    try:
        os.mkdir(path_folder_annotation)
    except FileExistsError:
        pass

    for idx, ROI in enumerate(all_ROIs):
        if WM:
            cv2.imwrite(os.path.join(path_folder_annotation, 'ROI_WM_' + str(idx) + '.tif'), ROI)
        else:
            cv2.imwrite(os.path.join(path_folder_annotation, 'ROI_GM_' + str(idx) + '.tif'), ROI)
                    
                    
def selectROIs(ROIs, image, size_ROI, CX_overimposed: bool = False):
    ROI_counter = 0
    for x in np.linspace(0,image.shape[0] - 1, image.shape[0]):
        for y in np.linspace(0,image.shape[1] - 1, image.shape[1]):
            x = int(x)
            y = int(y)

            distances_origin = {}
            if CX_overimposed:
                condition = 0
            else:
                condition = 255

            if image[int(x),int(y)] != condition and ROIs[int(x), int(y)] == 0:

                for idx in range(0,50):
                    for idy in range(0,50):
                        try:
                            
                            idx_tot = x + idx
                            idy_tot = y + idy
                            if image[int(idx_tot),int(idy_tot)] != condition and ROIs[int(idx_tot), int(idy_tot)] == 0:
                                distances_origin[tuple([idx_tot, idy_tot])] = np.linalg.norm(np.array([x,y]) - np.array([idx_tot,idy_tot]))


                            idx_tot = x + idx
                            idy_tot = y - idy
                            if image[int(idx_tot),int(idy_tot)] != condition and ROIs[int(idx_tot), int(idy_tot)] == 0:
                                distances_origin[tuple([idx_tot, idy_tot])] = np.linalg.norm(np.array([x,y]) - np.array([idx_tot,idy_tot]))

                            idx_tot = x - idx
                            idy_tot = y - idy
                            if image[int(idx_tot),int(idy_tot)] != condition and ROIs[int(idx_tot), int(idy_tot)] == 0:
                                distances_origin[tuple([idx_tot, idy_tot])] = np.linalg.norm(np.array([x,y]) - np.array([idx_tot,idy_tot]))

                            idx_tot = x - idx
                            idy_tot = y + idy
                            if image[int(idx_tot),int(idy_tot)] != condition and ROIs[int(idx_tot), int(idy_tot)] == 0:
                                distances_origin[tuple([idx_tot, idy_tot])] = np.linalg.norm(np.array([x,y]) - np.array([idx_tot,idy_tot]))

                        except:
                            pass
                            
                distances_origin_sorted = sorted(distances_origin.items(), key=lambda x:x[1])
                for dist in distances_origin_sorted[0:size_ROI]:
                    ROIs[dist[0][0], dist[0][1]] = ROI_counter + 1
                ROI_counter = ROI_counter + 1

    return ROIs, ROI_counter
