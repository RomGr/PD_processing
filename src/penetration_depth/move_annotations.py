import os, shutil
import numpy as np
import random
random.seed(42)
from PIL import Image
from tqdm import tqdm
from collections import Counter
from penetration_depth.helpers import save_repetitions

def move_annotations(path_data_orig, max_number_ROIs, CX_overimposed: bool = False, CC_overimposed: bool = False, small_ROIs: bool = False, measurements_types: list = None,
                     Flag: bool = False):
    
    if CX_overimposed or CC_overimposed:
        return move_annotations_CC_CX(path_data_orig, max_number_ROIs, CX_overimposed = CX_overimposed, CC_overimposed = CC_overimposed, small_ROIs = small_ROIs, Flag = Flag)
    else:
        return move_annotations_legacy(path_data_orig, max_number_ROIs, small_ROIs = small_ROIs, measurements_types = measurements_types, Flag = Flag)
    

def move_annotations_legacy(path_data_orig, max_number_ROIs, small_ROIs: bool = False, measurements_types: list = None, Flag: bool = False):
    
    ROIs_GM = {}
    ROIs_WM = {}
    
    repetitions = {}
    for measurement_type in measurements_types:
        
        if Flag:
            print()
            print(f"Creating small ROIs for {measurement_type}...")
            
        path_data = os.path.join(path_data_orig, measurement_type)
        
        repetitions[measurement_type] = create_annotation_json(path_data)
        
        for folder in (tqdm(os.listdir(path_data)) if Flag else os.listdir(path_data)):

            if folder != 'results':
                path_annotation = os.path.join(path_data, folder, 'annotation')
                annotations = []
                    
                for file in os.listdir(path_annotation):
                    if 'overimposed' in file or 'CC_' in file:
                        annotations.append(os.path.join(path_annotation, file))
                
                if small_ROIs:
                    create_small_ROIs(annotations, max_number_ROIs, WM = True)   
                    
                ROI_GM = 0
                ROI_WM = len(annotations)
        
        ROIs_GM[measurement_type + 'GM'] = ROI_GM
        ROIs_WM[measurement_type + 'WM'] = ROI_WM
        
        if Flag:
            print(f"Small ROIs created for {measurement_type}...")
            print()
                
    save_repetitions(repetitions)
    return ROIs_GM, ROIs_WM  



def move_annotations_CC_CX(path_data_orig, max_number_ROIs, CX_overimposed: bool = False, CC_overimposed: bool = False, small_ROIs: bool = False,
                     Flag: bool = False):
    

    path_data = os.path.join(path_data_orig, '90_overimposition')
    for folder in os.listdir(path_data):
        
        if Flag:
            print()
            print(f"Moving the annotations for {folder}...")
    
        if folder != 'results':
            path_annotation = os.path.join(path_data, folder, 'annotation')


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
                create_small_ROIs(annotations_WM, max_number_ROIs, WM = True)   
                
                
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
                    create_small_ROIs(annotations_GM, max_number_ROIs, WM = False)  
                
            ROIs_GM = len(annotations_GM)
            ROIs_WM = len(annotations_WM)
            
        if Flag:
            print(f"Small ROIs created for {folder}")
    return ROIs_GM, ROIs_WM       
                    
def create_small_ROIs(annotations, max_number_ROIs, WM: bool = True):
    
    size_ROI = 200
    all_ROIs = []
    
    path_folder_annotation = os.path.join(annotations[0].split('annotation')[0], 'annotation')
    path_small_ROIs = os.path.join(path_folder_annotation, 'small_ROIs')
    
    if WM:
        try:
            shutil.rmtree(path_small_ROIs)
        except FileNotFoundError:
            pass
        
        os.mkdir(path_small_ROIs)
    
    ROIs_combined = np.zeros([np.array(Image.open(annotations[0])).shape[0], np.array(Image.open(annotations[0])).shape[1], 3])
    
    for annotation in annotations:
        image = np.array(Image.open(annotation))
        ROIs = np.zeros(image.shape)
    
        ROIs, ROI_counter = selectROIs(ROIs, image, size_ROI)

        if ROI_counter > max_number_ROIs:
            numbers = list(range(1, ROI_counter + 1))
            random_elements = random.sample(numbers, max_number_ROIs)
        else:
            random_elements = range(1, ROI_counter + 1)

        for x in random_elements:
            if np.sum(ROIs == x) != 0:
                all_ROIs.append((ROIs == x) * 255)
                color = [random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)]
                for idx in range(ROIs_combined.shape[0]):
                    for idy in range(ROIs_combined.shape[1]):
                        if ROIs[idx, idy] == x:
                            ROIs_combined[idx, idy] = color
    
    
    for idx, ROI in enumerate(all_ROIs):
        if WM:
            Image.fromarray(ROI.astype(np.uint8)).save(os.path.join(path_small_ROIs, 'ROI_WM_' + str(idx) + '.tif'))
        else:
            Image.fromarray(ROI.astype(np.uint8)).save(os.path.join(path_small_ROIs, 'ROI_GM_' + str(idx) + '.tif'))
            
    if WM:
        Image.fromarray(ROIs_combined.astype(np.uint8)).save(os.path.join(path_folder_annotation, 'ROIs_combined_WM.png'))
    else:
        Image.fromarray(ROIs_combined.astype(np.uint8)).save(os.path.join(path_folder_annotation, 'ROIs_combined_GM.png'))
                    
                    
def selectROIs(ROIs, image, size_ROI):
    ROI_counter = 0
    for x in np.linspace(0,image.shape[0] - 1, image.shape[0]):
        for y in np.linspace(0,image.shape[1] - 1, image.shape[1]):
            x = int(x)
            y = int(y)

            distances_origin = {}
            condition = 255

            if image[int(x),int(y)] == condition and ROIs[int(x), int(y)] == 0:
                
                for idx in range(0,50):
                    for idy in range(0,50):
                        try:
                            
                            idx_tot = x + idx
                            idy_tot = y + idy
                            if image[int(idx_tot),int(idy_tot)] == condition and ROIs[int(idx_tot), int(idy_tot)] == 0:
                                distances_origin[tuple([idx_tot, idy_tot])] = np.linalg.norm(np.array([x,y]) - np.array([idx_tot,idy_tot]))


                            idx_tot = x + idx
                            idy_tot = y - idy
                            if image[int(idx_tot),int(idy_tot)] == condition and ROIs[int(idx_tot), int(idy_tot)] == 0:
                                distances_origin[tuple([idx_tot, idy_tot])] = np.linalg.norm(np.array([x,y]) - np.array([idx_tot,idy_tot]))

                            idx_tot = x - idx
                            idy_tot = y - idy
                            if image[int(idx_tot),int(idy_tot)] == condition and ROIs[int(idx_tot), int(idy_tot)] == 0:
                                distances_origin[tuple([idx_tot, idy_tot])] = np.linalg.norm(np.array([x,y]) - np.array([idx_tot,idy_tot]))

                            idx_tot = x - idx
                            idy_tot = y + idy
                            if image[int(idx_tot),int(idy_tot)] == condition and ROIs[int(idx_tot), int(idy_tot)] == 0:
                                distances_origin[tuple([idx_tot, idy_tot])] = np.linalg.norm(np.array([x,y]) - np.array([idx_tot,idy_tot]))

                        except:
                            pass
                            
                distances_origin_sorted = sorted(distances_origin.items(), key=lambda x:x[1])
                for dist in distances_origin_sorted[0:size_ROI]:
                    ROIs[dist[0][0], dist[0][1]] = ROI_counter + 1
                ROI_counter = ROI_counter + 1

    return ROIs, ROI_counter


def most_common_occurrence(lst):
    if not lst:
        return 0  # If the list is empty, there are no occurrences
    counts = Counter(lst)
    return max(counts.values())  # Return the count of the most common element

def create_annotation_json(path_data):
    thicknesses = []
    for folder in os.listdir(path_data):
        thicknesses.append(folder.split('um_')[0].split('-')[-1])
    return most_common_occurrence(thicknesses)