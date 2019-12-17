#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


bd_images_path = 'D:\\20190927'

def cv_read(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path, dtype = np.uint8), -1)
    return cv_img

#print(os.listdir(bd_images_path))   


# In[3]:


def grid_coord_generate(gshape, gsize):
    coord = np.zeros([gshape[0]*gshape[1], 3])
    coord[:,:2] = np.mgrid[0:gshape[0], 0:gshape[1]].T.reshape(-1,2)
    return coord


# In[4]:


def fill_con_circle(src0):

    circles0 = cv.HoughCircles(src0, cv.HOUGH_GRADIENT, dp = 1, minDist = 40.0, param1 = 200, param2 = 50, minRadius = 10, maxRadius = 60)

    if circles0 is None:
        print('failed')
        return None, None
    else:
        circles0 = np.squeeze(circles0)
        print('find', circles0.shape, 'circles')
    
    int_center_coord = circles0.astype('int')
    int_center_x = int_center_coord[:,0]
    int_center_y = int_center_coord[:,1]
    center_intensity = src0[int_center_y, int_center_x]

    indice = np.argsort(center_intensity)
    mean_intensity = np.mean(center_intensity)

    if mean_intensity < center_intensity[indice[-1]]:
        con_circle_center = circles0[indice[:-4:-1],:2]
    elif mean_intensity > center_intensity[indice[0]]:
        con_circle_center = circles0[indice[:3],:2]

    #print(int(np.mean(int_center_coord[:,2])/2))
    #print(center_intensity)
    #print(con_circle_center)
    '''
    plt.figure(figsize = (20, 10))
    plt.subplot(121)
    plt.imshow(src0, cmap = 'gray')
    plt.plot(circles0[:,0], circles0[:,1],'ro' )
    plt.plot(con_circle_center[:,0], con_circle_center[:,1], 'ys', markersize = 10 )
    plt.subplot(122)
    plt.imshow(src0, cmap = 'gray')
    plt.show()
    '''
    point_color = (mean_intensity)
    thickness = -1
    radii = int(np.mean(int_center_coord[:,2])/5*4)
    # center = (2319,2041)
    for center in con_circle_center:
        src0 = cv.circle(src0, (center[0],center[1]), radii, point_color, thickness)
    
    return src0, con_circle_center

#src0, con_circle_center = fill_con_circle(src0)


# In[14]:


def find_circles(src, patternSize, plot_flag = False):
    
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    #params.thresholdStep = 5
    #params.minThreshold = 10
    #params.maxThreshold = 250

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 100000
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    detector = cv.SimpleBlobDetector_create(params)
    
    #patternSize = (9,12)
    isFound, centers = cv.findCirclesGrid(src, patternSize, flags=cv.CALIB_CB_SYMMETRIC_GRID+cv.CALIB_CB_CLUSTERING, blobDetector = detector)

    if isFound:
        print('find', centers.shape[0], 'circles') 
    else:
        print('failed to find circles')

    cv.drawChessboardCorners(src, patternSize, centers, isFound)
    
    if plot_flag:
        plt.figure(figsize = (5,4))
        plt.imshow(src, cmap = 'gray')
        plt.show()
    
    return np.squeeze(centers)

#image_points = find_circles(src0)


# In[15]:


grid_shape = (12, 9)
grid_size = 6
grid_coord = grid_coord_generate(grid_shape, grid_size)

def read_img_find_circle(bd_images_path):

    image_list = os.listdir(bd_images_path)
    obj_points = []
    left_image_points = []
    right_image_points = []
    for i, image_name in enumerate(image_list):
        if image_name[-5:] == '0.bmp':
            src_image = cv_read(os.path.join(bd_images_path, image_name))
            position_flag = 1
        elif image_name[-5:] == '1.bmp':
            src_image = cv_read(os.path.join(bd_images_path, image_name))
            position_flag = 0
        else:
            print('please rename the images or check the list')
            exist()
            position_flag = -1
            
        print(position_flag)
        img_shape = src_image.shape
        
        src_image = 255-src_image
        src, con_circle_center = fill_con_circle(src_image)
        center_points = find_circles(src, grid_shape)
        
        if position_flag:
            obj_points.append(grid_coord)
            left_image_points.append(center_points)
        else:
            right_image_points.append(center_points)
    
    return obj_points, left_image_points, right_image_points
    
# obj_points, left_image_points, right_image_points = read_img_find_circle(bd_images_path)

# In[7]:

def calibrate(obj_points, left_image_points, right_image_points, img_shape):
    obj_points = np.array(obj_points).astype('float32')

    left_rt, left_camera_matrix, left_dist, left_R_vector, left_T_vector = cv.calibrateCamera(
        obj_points, left_image_points, img_shape, None, None)
    right_rt, right_camera_matrix, right_dist, right_R_vector, right_T_vector = cv.calibrateCamera(
        obj_points,right_image_points, img_shape, None, None)

    ret, M1, d1, M2, d2, R, T, E, F = cv.stereoCalibrate(
        obj_points, left_image_points, right_image_points,
        left_camera_matrix, left_dist, right_camera_matrix, right_dist,
        img_shape)

    print('Intrinsic_mtx_1:\n', M1)
    print('dist_1:\n', d1)
    print('Intrinsic_mtx_2:\n', M2)
    print('dist_2:\n', d2)
    print('R', R)
    print('T', T)
    print('E', E)
    print('F', F)
    return M1, d1, M2, d2, R, T, E, F

# M1, d1, M2, d2, R, T, E, F = calibrate(obj_points, left_image_points, right_image_points, img_shape)