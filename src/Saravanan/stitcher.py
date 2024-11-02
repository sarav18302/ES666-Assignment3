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

        S = stitch.Stitch(all_images)
        S.stitch_images()



        



        # Collect all homographies calculated for pair of images and return
        homography_matrix_list =[]
        # Return Final panaroma
        stitched_image = S.result
        #####
        
        return stitched_image, homography_matrix_list 

    def convert_coordinates(self, pixel_x, pixel_y):
        global center, focal_length
    
        transformed_x = (focal_length * np.tan((pixel_x - center[0]) / focal_length)) + center[0]
        transformed_y = ((pixel_y - center[1]) / np.cos((pixel_x - center[0]) / focal_length)) + center[1]
    
        return transformed_x, transformed_y

    def project_onto_cylinder(self, initial_image):
        print("Projecting onto Cylinder")
        global image_width, image_height, center, focal_length
        image_height, image_width = initial_image.shape[:2]
        center = [image_width // 2, image_height // 2]
        focal_length = 1000
    
        # Creating a blank transformed image
        transformed_image = np.zeros(initial_image.shape, dtype=np.uint8)
    
        # Using meshgrid to generate all pixel coordinates
        x_coords, y_coords = np.meshgrid(np.arange(image_width), np.arange(image_height))
        grid_x = x_coords.flatten()
        grid_y = y_coords.flatten()
    
        # Finding corresponding coordinates of the transformed image in the initial image
        source_x, source_y = self.convert_coordinates(grid_x, grid_y)
    
        # Rounding off the coordinate values to get exact pixel values (top-left corner)
        source_tl_x = source_x.astype(int)
        source_tl_y = source_y.astype(int)
    
        # Finding valid transformed image points whose corresponding 
        # initial image points lie inside the initial image
        valid_indices = (source_tl_x >= 0) & (source_tl_x < (image_width - 1)) & \
                        (source_tl_y >= 0) & (source_tl_y < (image_height - 1))
    
        # Filtering out invalid points
        grid_x = grid_x[valid_indices]
        grid_y = grid_y[valid_indices]

        source_x = source_x[valid_indices]
        source_y = source_y[valid_indices]
    
        source_tl_x = source_tl_x[valid_indices]
        source_tl_y = source_tl_y[valid_indices]
    
        # Bilinear interpolation
        delta_x = source_x - source_tl_x
        delta_y = source_y - source_tl_y
    
        weight_top_left = (1.0 - delta_x) * (1.0 - delta_y)
        weight_top_right = delta_x * (1.0 - delta_y)
        weight_bottom_left = (1.0 - delta_x) * delta_y
        weight_bottom_right = delta_x * delta_y
    
        # Using advanced indexing for interpolation
        transformed_image[grid_y, grid_x, :] = (weight_top_left[:, None] * initial_image[source_tl_y, source_tl_x, :] +
                                                 weight_top_right[:, None] * initial_image[source_tl_y, source_tl_x + 1, :] +
                                                 weight_bottom_left[:, None] * initial_image[source_tl_y + 1, source_tl_x, :] +
                                                 weight_bottom_right[:, None] * initial_image[source_tl_y + 1, source_tl_x + 1, :])
    
        # Cropping out the black region from both sides (using symmetry)
        min_x = np.min(grid_x)
        transformed_image = transformed_image[:, min_x: -min_x, :]
    
        return transformed_image



 
    
