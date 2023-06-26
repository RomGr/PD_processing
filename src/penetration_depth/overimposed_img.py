from PIL import Image
import numpy as np
import os
import traceback

from tqdm import tqdm

def save_img(path, results_path, wavelength):
    paths_masks = os.listdir(os.path.join(path, 'annotation'))
    path_masks_rel = []
    for p in paths_masks:
        if '.tif' in p:
            path_masks_rel.append(os.path.join(path, 'annotation', p))
    try:
        os.mkdir(os.path.join(results_path, 'imgs'))
    except FileExistsError:
        pass

    path_image = os.path.join(path, 'polarimetry', wavelength, path.split('\\')[-1] + '_' + wavelength + '_realsize.png')
    path_save = os.path.join(os.path.join(results_path, 'imgs', 'intensity', path.split('\\')[-1] + '.png'))
    try:
        os.mkdir(os.path.join(results_path, 'imgs', 'intensity'))
    except:
        pass
    generate_pixel_image(path_image, path_masks_rel, path_save)

    for parameter in ['Depolarization', 'Linear retardance', 'Azimuth of optical axis']:
        path_image = os.path.join(path, 'polarimetry', wavelength, parameter + '_img.png')
        path_save = os.path.join(os.path.join(results_path, 'imgs', parameter, path.split('\\')[-1] + '.png'))
        try:
            os.mkdir(os.path.join(results_path, 'imgs', parameter))
        except:
            pass
        generate_pixel_image(path_image, path_masks_rel, path_save, val_replace = 0)

def save_the_imgs(paths: list, results_path: str, wavelength: str, Flag: bool = False):
    
    if Flag:
        for path in tqdm(paths):
            try:
                save_img(path, results_path, wavelength)
            except FileNotFoundError :
                if wavelength != '550nm' and wavelength != '650nm':
                    pass
                else:
                    traceback.print_exc()
    else:   
        for path in paths:
            try:
                save_img(path, results_path, wavelength)
            except FileNotFoundError :
                if wavelength != '550nm' and wavelength != '650nm':
                    pass
                else:
                    traceback.print_exc()


def generate_pixel_image(path_image, paths_masks, path_save, val_replace = 255):    
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