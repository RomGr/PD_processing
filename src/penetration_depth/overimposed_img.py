from PIL import Image
import numpy as np
import os
import traceback
import pickle
from tqdm import tqdm

def save_the_imgs(paths: list, results_path: str, wavelength: str, Flag: bool = False, CC_overimposed: bool = False,
                  measurement_type = None):
    """ -----------------------------------------------------------
    # save_the_imgs is the master function calling save_img for each of the measurement folders
    #
    # Parameters:
    #   paths (list): list of paths to the measurement folders
    #   results_path (str): path to the results folder
    #   wavelength (str): wavelength of the measurement
    #   Flag (bool): if True, tqdm is used to display a progress bar
    ----------------------------------------------------------- """
    for path in (tqdm(paths) if Flag else paths):
        try:
            save_img(path, results_path, wavelength, CC_overimposed = CC_overimposed, measurement_type = measurement_type)
        except FileNotFoundError :
            if wavelength != '550nm' and wavelength != '650nm':
                pass
            else:
                traceback.print_exc()
                    
                    
def save_img(path: str, results_path: str, wavelength: str, CC_overimposed: bool = False, measurement_type = None):
    """ -----------------------------------------------------------
    # save_img is the master function calling generate_pixel_image for one measurement folder
    #
    # Parameters:
    #   path (str): path to the measurement folder
    #   results_path (str): path to the results folder
    #   wavelength (str): wavelength of the measurement
    ----------------------------------------------------------- """
    if CC_overimposed:
        path_msk = os.path.join(path, 'annotation', 'small_ROIs')
    else:
        path_msk = os.path.join(path, 'annotation')
    paths_masks = os.listdir(path_msk)
    path_masks_rel = []
    
    # get the annotation masks
    for p in paths_masks:
        if '.tif' in p:
            if measurement_type is not None:
                if measurement_type[1] in p:
                    path_masks_rel.append(os.path.join(path_msk, p))
            else:
                path_masks_rel.append(os.path.join(path_msk, p))

    try:
        os.mkdir(os.path.join(results_path, 'imgs'))
    except FileExistsError:
        pass

    # load the intensity image...
    path_image = os.path.join(path, 'polarimetry', wavelength, path.split('\\')[-1] + '_' + wavelength + '_realsize.png')
    path_save = os.path.join(os.path.join(results_path, 'imgs', 'intensity', path.split('\\')[-1] + '.png'))
    try:
        os.mkdir(os.path.join(results_path, 'imgs', 'intensity'))
    except:
        pass
    # ...and generate the pixel image for it
    generate_pixel_image(path_image, path_masks_rel, path_save)

    # do the same for each of the polarimetric parameters
    for parameter in ['Depolarization', 'Linear retardance', 'Azimuth of optical axis']:
        path_image = os.path.join(path, 'polarimetry', wavelength, parameter + '_img.png')
        path_save = os.path.join(os.path.join(results_path, 'imgs', parameter, path.split('\\')[-1] + '.png'))
        try:
            os.mkdir(os.path.join(results_path, 'imgs', parameter))
        except:
            pass
        generate_pixel_image(path_image, path_masks_rel, path_save, val_replace = 0)


def generate_pixel_image(path_image: str, paths_masks: list, path_save: str, val_replace = 255):    
    """ -----------------------------------------------------------
    # creates the images overlying the masks and the polarimetric parameters
    #
    # Parameters:
    #   path_image (str): path to the image of interest
    #   paths_masks (list): list of paths to the masks
    #   path_save (str): path to save the image
    #   val_replace (int): value to replace the pixels of the mask with (default: 255 or black)
    ----------------------------------------------------------- """
    im = Image.open(path_image)
    imnp = np.array(im)
    
    masks = []
    for masked in paths_masks:
        masks.append(np.array(Image.open(masked)))
        
    mask_combined = np.zeros((imnp.shape[0], imnp.shape[1]))
    for mask in masks:
        mask_combined += mask
    mask = mask_combined
    
    
    to_add = []
    for idx_x, x in enumerate(imnp):
        min_idx_x = max(idx_x - 1, 0)
        max_idx_x = min(idx_x + 1, len(mask) - 1)
        
        for idx_y, _ in enumerate(x):
            min_idx_y = max(idx_y - 1, 0)
            max_idx_y = min(idx_y + 1, len(mask[0]) - 1)
            if mask[idx_x][idx_y] != 0 and mask[min_idx_x][idx_y] == 0:
                to_add.append([idx_x, idx_y])
            elif mask[idx_x][idx_y] != 0 and mask[max_idx_x][idx_y] == 0:
                to_add.append([idx_x, idx_y])
            elif mask[idx_x][idx_y] != 0 and mask[idx_x][min_idx_y] == 0:
                to_add.append([idx_x, idx_y])
            elif mask[idx_x][idx_y] != 0 and mask[idx_x][max_idx_y] == 0:
                to_add.append([idx_x, idx_y])
                
    for add in to_add:
        imnp[add[0]][add[1]] = val_replace
        imnp[add[0] - 1][add[1]] = val_replace
        imnp[add[0] + 1][add[1]] = val_replace
        imnp[add[0]][add[1] + 1] = val_replace
        imnp[add[0] - 1][add[1] + 1] = val_replace
        imnp[add[0] + 1][add[1] + 1] = val_replace
        imnp[add[0]][add[1] - 1] = val_replace
        imnp[add[0] - 1][add[1] - 1] = val_replace
        imnp[add[0] + 1][add[1] - 1] = val_replace
    
    Image.fromarray(imnp).save(path_save)