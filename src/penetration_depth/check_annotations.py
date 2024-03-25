import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import traceback

def find_blob(mask: np.array):
    """ -----------------------------------------------------------
    # function used to find the different blobs
    #
    # Parameters:
    #   mask (np.array): mask containing the annotations
    #
    # Returns:
    #   ndimage.label(imgf > threshold) (tuple): tuple containing an image with the blobs and the number of blobs
    ----------------------------------------------------------- """  
    blur_radius = 0
    threshold = 50
    img = mask
    # smooth the image (to remove small objects)
    imgf = ndimage.gaussian_filter(img, blur_radius)
    threshold = 50
    return ndimage.label(imgf > threshold) 


def get_blobs(img_split: list, kernel_size: int = 5):
    """ -----------------------------------------------------------
    # function used to get the blobs from the annotations and to create the combined mask combining all the annotation
    #
    # Parameters:
    #   img_split (list): list of the paths to the annotations
    #   kernel_size (int): size of the kernel used for the erosion
    #
    # Returns:
    #   mask_combined (np.array): combined annotation mask
    #   blobs (tuple): tuple containing an image with the blobs and the number of blobs
    ----------------------------------------------------------- """  
    imgs_splits = []
    for img in img_split:
        imgs_splits.append(np.array(Image.open(img)).astype(np.uint8))
        
    mask_combined = np.zeros(imgs_splits[0].shape)
    mask_combined_blob = np.zeros(imgs_splits[0].shape)
    
    for idx, img in enumerate(imgs_splits):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.erode(img, kernel) 
        mask_combined += (idx + 1) * (mask != 0)
        mask_combined_blob += mask
        
    try:
        assert len(np.unique(mask_combined_blob)) == 2
        blobs = None
    except:
        blobs = find_blob(mask_combined_blob)
    return mask_combined, blobs


def check_the_annotations(to_clean: list):
    """ -----------------------------------------------------------
    # check_the_annotations allows to check and re-do the annotations of the data to check that the annotations are not overlying one another
    #
    # Parameters:
    #   to_clean (list): list of tuples containing the path to the data to check and the number of expected blobs
    ----------------------------------------------------------- """
    for clean in to_clean:
        incorrect = True
        while incorrect:
            # get the blobs
            path, _, _ = clean
            imgs_split = []
            annotation_folder = os.path.join(path, 'annotation')
            for annotation in os.listdir(annotation_folder):
                if annotation.endswith('.tif'):
                    imgs_split.append(os.path.join(annotation_folder, annotation))
            try:
                _, blobs = get_blobs(imgs_split)
            except:
                traceback.print_exc()
                print(imgs_split, path)
                
            try:
                os.mkdir(os.path.join('./', 'tmp'))
            except:
                pass
            
            if blobs is None:
                incorrect = False
            else:
                # plot the blobs
                plt.imshow(blobs[0]) 
                plt.savefig(os.path.join('./', 'tmp', 'blobs_fig.png'))
                
                # show them and ask the user to re-do the annotations, and after this to press enter to re-process
                cv2.imshow("Blobs", np.asarray(cv2.imread(os.path.join('./', 'tmp', 'blobs_fig.png'))))
                print(clean)
                print("Number of expected blobs: {}, Number of blobs: {}".format(len(imgs_split), blobs[1]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if len(imgs_split) != blobs[1]:
                    input("Press enter to re-process...")
                else:
                    incorrect = False