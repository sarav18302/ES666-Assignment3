import cv2
import numpy as np
import os
from src.Saravanan.some_folder import match
import math

class Stitch():
    def __init__(self,filenames,focal_length):
        # self.path = args
        # filenames = sorted([os.path.join(self.path, f) for f in os.listdir(self.path) if f.lower().endswith('.jpg')])
        print("Filenames :",*filenames)
        self.focal_length = focal_length
        self.imageSet = [cv2.imread(each) for each in filenames]
        self.images = [cv2.resize(each,(480,320)) for each in self.imageSet ]
        self.pimages = [self.project_onto_cylinder(each) for each in self.images]
        self.count = len(self.images)
        self.left_list, self.right_list = [],[]
        self.centerIdx = int(self.count/2)
        self.matcher_obj = match.matchers()
        self.homography_matrix_list =[]

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
        focal_length = self.focal_length
    
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
        
    
    def homogeneous_coordinate(self,coordinate):
        x = coordinate[0]/coordinate[2]
        y = coordinate[1]/coordinate[2]
        return x, y
    
    def backward_warping(self,SecImg,homography,window_size):
        height1,width1 = SecImg.shape[:2]
        size_x,size_y = window_size
        homography_inverse = np.linalg.inv(homography)
        print("Homography inverse: ", "\n", homography_inverse)
        warp = np.zeros((window_size[1], window_size[0], 3))
        
        for x in range(size_x):
            for y in range(size_y):
                point_xy = self.homogeneous_coordinate(np.dot(homography_inverse, [[x], [y], [1]]))
                point_x = int(point_xy[0])
                point_y = int(point_xy[1])

                if (point_x >= 0 and point_x < width1 and point_y >= 0 and point_y < height1):
                    warp[y, x, :] = SecImg[point_y, point_x, :]
        return warp
        
    def weighted_sum(self,img1,img2,alpha):
        return np.clip(img1*alpha+img2*(1-alpha),0,255).astype(img1.dtype)

    def blend_images(self,img1,img2,alpha):
        img1 = np.where(np.all(img1 == 0, axis=-1, keepdims=True), img2, img1)
        img2 = np.where(np.all(img2 == 0, axis=-1, keepdims=True), img1, img2)
        alpha = 0.5
        warp = self.weighted_sum(img2,img1,alpha)
        return warp
    
    def wrap(self,BaseImg,SecImg):
        
        Homography = self.matcher_obj.match(BaseImg,SecImg)
        print("Homography: ",Homography)
        self.homography_matrix_list.append(Homography)
        height1, width1 = SecImg.shape[:2]
        height2, width2 = BaseImg.shape[:2]

    # Performing forward Wrapping of corners of the image  
        up_left_cor = self.homogeneous_coordinate(np.dot(Homography, [[0],[0],[1]]))
        up_right_cor = self.homogeneous_coordinate(np.dot(Homography, [[width1],[0],[1]]))
        low_left_cor = self.homogeneous_coordinate(np.dot(Homography, [[0],[height1],[1]]))
        low_right_cor = self.homogeneous_coordinate(np.dot(Homography, [[width1],[height1],[1]]))
        transformed_corners =np.float32([up_left_cor,low_left_cor,low_right_cor,up_right_cor]).reshape(-1, 1, 2)
        
    # Concatenating the base Image corner coordinates with transformed secondary image coordinates
        all_corners = np.concatenate((transformed_corners, np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2))) 
        
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        Hnew = Ht.dot(Homography)
        homography = Hnew
    
        offset_x = math.floor(xmin)
        offset_y = math.floor(ymin)


        max_x = math.ceil(xmax)
        max_y = math.ceil(ymax)

        size_x = max_x - offset_x
        size_y = max_y - offset_y


        #Perform backward Warping
        window_size = [size_x,size_y]
        warp = self.backward_warping(SecImg,homography,window_size)
        print("Warping is completed.")

        img= np.zeros((window_size[1],window_size[0],3))
        img[t[1]:height2+t[1], t[0]:width2+t[0]] = BaseImg

        warp = self.blend_images(warp,img,alpha = 0.5)
        print("Image is stitched")
        warp= warp.astype(BaseImg.dtype)
        
        return warp

    def stitch_images(self):
        
        #Stitching Images from centerIdx to the end of list
        BaseImg = self.pimages[self.centerIdx]
        for SecImg in self.pimages[self.centerIdx+1:]:
            BaseImg = self.wrap(BaseImg,SecImg)
        rightStitch = BaseImg
        
        #Stitching Images from index 0 to centerIdx 
        BaseImg = self.pimages[self.centerIdx-1]
        for SecImg in self.pimages[0:self.centerIdx-1][::-1]:
            BaseImg = self.wrap(BaseImg,SecImg)
        leftStitch = BaseImg
        
        #Stitching leftStitch and rightStitch
        BaseImg = rightStitch
        SecImg = leftStitch
        self.result = self.wrap(BaseImg,SecImg)
