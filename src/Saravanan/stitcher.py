import pdb
import glob
import cv2
import os
import numpy as np
from src.Saravanan.some_folder import stitch


class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        ####  Your Implementation here
        #### you can use functions, class_methods, whatever!! Examples are illustrated below. Remove them and implement yours.
        #### Just make sure to return final stitched image and all Homography matrices from here
        focal_length = {
            "I1": 700,
            "I2": 800,
            "I3": 400,
            "I4": 600,
            "I5" :700,
            "I6" : 1000
        }
        S = stitch.Stitch(all_images,focal_length[path[-2:]])
        S.stitch_images()
        # Collect all homographies calculated for pair of images and return
        homography_matrix_list = S.homography_matrix_list
        # Return Final panaroma
        stitched_image = S.result
        #####
        
        return stitched_image, homography_matrix_list 
