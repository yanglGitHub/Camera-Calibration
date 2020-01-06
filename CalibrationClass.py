#!/usr/bin/env python
# coding: utf-8

import cv2
import glob
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pdb


# 解释judge_grid_orientation部分程序
# 
# 首先，确定哪个点是三角形的哪个顶点，角度对边点概念
# 
# 然后，根据最短边和x方向的夹角判断朝向
# 
# OriFlag = 1,2,3,4
# ![image.png](attachment:image.png)

class CameraCalibration(object):
    
    def __init__(self, filepath, patternSpace = 1.0, patternSize = (12, 9)):
        self.filepath = filepath
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.patternSize = patternSize
        self.objp = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
        self.objp[:, :2] = patternSpace*np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
        self.initial()
        
        
    def show_image(self, sequence = 2):
        first2images = glob.glob(self.filepath)[:sequence]
        for image in first2images:
            img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), -1)
            # print(image, type(img))
            plt.figure()
            plt.imshow(img, cmap = 'gray')
            plt.show()


    def initial(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        
        
    def calibrate_images(self):
        
        self.images_list = glob.glob(self.filepath)
        self.images_list.sort()
        
        for i, fname in enumerate(self.images_list):
        
            print(i)
            img = cv2.imdecode(np.fromfile(fname, dtype = np.uint8), -1)
            
            # judge the orientation of the grid pattern
            img, mean_radii, con_circle = self.fill_con_circle(img, False)
            # if no circles or not enough circles was found, then continue to next calibration image
            if img is None:
                continue
            # Find the chess board corners
            ret, corners = self.find_circles_grid(img, mean_radii)
            
            # the orientation of the grid pattern, these three concentric circles, is important
            # OriFlag = self.judge_grid_orientation(con_circle)
            OriFlag = 1            
            t = {1: lambda points: points, 
                 2: lambda points: points, 
                 3: lambda points: np.flip(np.reshape(points, (12,9,2)), (0,1)).reshape(108,2),
                 4: lambda points: np.flip(np.reshape(points, (12,9,2)), (0,1)).reshape(108,2),}
            
            if ret is True:
                self.objpoints.append(self.objp)
                # modify the order of these center points according to the orientation of concentric circles
                corners = t[OriFlag](corners)
                # If found, add object points, image points (after refining them)
                self.imgpoints.append(corners)
            else:
                continue
                
            img_shape = img.shape[::-1]
    
        # calibrate single camera
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_shape, None, None)
        
        # record the results
        self.camera_model = dict([('Intrinsic', self.M1), ('distortion', self.d1)])
        
        
    def clean_data(self, data):
        mean_value = np.mean(data)
        std_value = np.std(data)
        median_value = np.median(data)
        # print('mean_value: ',mean_value, '\n std_value: ', std_value, '\n median_value: ', median_value)
        index = abs(data-median_value) > 0.2*median_value
        '''
        if std_value/mean_value < 0.05:
            index = abs(data-mean_value) > 5*std_value
        else:
            index = abs(data-median_value) > std_value
        '''
        return index

    def fill_con_circle(self, src, plot_flag = False):

        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, dp = 1, minDist = 40.0, param1 = 200, param2 = 20, minRadius = 10, maxRadius = 60)
        if circles is None:
            print('failed')
            return None, None, None
        elif circles.shape[1] < 20:
            print('not enough circles')
            return None,None, None
        else:
            circles = np.squeeze(circles)
            print('find', circles.shape[0], 'circles')

        int_center_coord = circles.astype('int')
        int_center_x = int_center_coord[:,0]
        int_center_y = int_center_coord[:,1]
        center_intensity = src[int_center_y, int_center_x]

        # put the cooridinate, radii, center intensity into one matrix
        circles_stack = np.append(circles, np.reshape(center_intensity, (-1,1)), axis = 1)

        index = self.clean_data(circles_stack[:,2])
        circles_filter = circles_stack[~index, :]

        indice = np.argsort(circles_filter[:,3])
        mean_intensity = np.mean(circles_filter[:,3])
        median_radii = np.median(circles_filter[:,2])

        max2intensity = circles_filter[indice[-2], 3]

        if np.abs(max2intensity - np.median(circles_filter[:,3])) < 20:
            con_circle_center = circles_filter[indice[:3],:]
            white_back = False
        else:
            con_circle_center = circles_filter[indice[:-4:-1],:]
            white_back = True

        point_color = int(mean_intensity)
        thickness = -1
        radii = int(median_radii/5*3)
        # center = (2319,2041)

        for center in con_circle_center:
            src_filled = cv2.circle(src, (center[0],center[1]), radii, point_color, thickness)

        if plot_flag:
            plt.figure()
            plt.subplot(121)
            plt.imshow(src, cmap = 'gray')
            plt.plot(circles_filter[:,0], circles_filter[:,1],'ro' )
            plt.plot(con_circle_center[:,0], con_circle_center[:,1], 'ys', markersize = 10 )
            plt.subplot(122)
            plt.imshow(src_filled, cmap = 'gray')
            plt.show()

        if not(white_back):
            src_filled = 255-src_filled

        return src_filled, median_radii, con_circle_center
    
    def judge_grid_orientation(self, con_circle):
        pi = 3.14159
        d01 = np.sum((con_circle[0,:2] - con_circle[1,:2])**2)
        d12 = np.sum((con_circle[1,:2] - con_circle[2,:2])**2)
        d20 = np.sum((con_circle[2,:2] - con_circle[0,:2])**2)

        d = [d01, d12, d20]
        index = np.argsort(d)

        min_angle = (index[0]+2) % 3
        mid_angle = (index[1]+2) % 3
        max_angle = (index[2]+2) % 3

        # 图像的y与坐标系的y符号相反
        midLengthSide = con_circle[mid_angle, :2] - con_circle[max_angle, :2]
        midLengthSide2VerticalAngle = midLengthSide[0]/np.sqrt(np.sum(midLengthSide**2))
        if midLengthSide2VerticalAngle > np.cos(20/180*pi):
            OriFlag = 3
        elif midLengthSide2VerticalAngle < -np.cos(20/180*pi):
            OriFlag = 1
        elif np.abs(midLengthSide2VerticalAngle) < np.cos(20/180*pi):
            if midLengthSide[1] > 0:
                OriFlag = 4
            else:
                OriFlag = 2
        return OriFlag

    def find_circles_grid(self, src, mean_radii = None, plot_flag = False):

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.thresholdStep = 5
        params.minThreshold = 10
        params.maxThreshold = 250

        # Filter by Area.
        params.filterByArea = True
        if mean_radii is None:
            params.minArea = 200
        else:
            params.minArea = int(3.14*mean_radii/1.5*mean_radii/1.5)

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

        detector = cv2.SimpleBlobDetector_create(params)

        #patternSize = (9,12)
        isFound, centers = cv2.findCirclesGrid(src, self.patternSize, 
                                               flags=cv2.CALIB_CB_SYMMETRIC_GRID+cv2.CALIB_CB_CLUSTERING, blobDetector = detector)

        if isFound:
            print('find', centers.shape[0], 'circles') 
        else:
            print('failed to find circles')
            return None, None

        cv2.drawChessboardCorners(src, self.patternSize, centers, isFound)

        if plot_flag:
            plt.figure(figsize = (5,4))
            plt.imshow(src, cmap = 'gray')
            plt.show()

        return isFound, np.squeeze(centers)


class StereoCalibration(CameraCalibration):

    def initial(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []
        
    def calibrate_images(self):
        
        # rewritten the file address and rules into left and right
        images_right = glob.glob(self.filepath[:-4:1] + '1.bmp')
        images_left = glob.glob(self.filepath[:-4:1] + '0.bmp')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            print(i)
            img_l = cv2.imdecode(np.fromfile(images_left[i], np.uint8), -1)
            img_r = cv2.imdecode(np.fromfile(images_right[i], np.uint8), -1)

            # Find the chess board corners
            img_l, mean_radii_l, con_circles_l = self.fill_con_circle(img_l)
            img_r, mean_radii_r, con_circles_r = self.fill_con_circle(img_r)
            if img_l is None or img_r is None:
                continue
            
            ret_l, corners_l = self.find_circles_grid(img_l, mean_radii_l)
            ret_r, corners_r = self.find_circles_grid(img_r, mean_radii_r)
            
            # OriFlag_l = self.judge_grid_orientation(con_circles_l)            
            # OriFlag_r = self.judge_grid_orientation(con_circles_r)  
            OriFlag_l = 1
            OriFlag_r = 1
            
            t = {1: lambda points: points, 
                 2: lambda points: points, 
                 3: lambda points: np.flip(np.reshape(points, (12,9,2)), (0,1)).reshape(108,2),
                 4: lambda points: np.flip(np.reshape(points, (12,9,2)), (0,1)).reshape(108,2),}
                 
                        
            
            # If found, add object points, image points (after refining them)
            if ret_l is True and ret_r is True:            
                self.objpoints.append(self.objp)
                corners_l = t[OriFlag_l](corners_l)
                corners_r = t[OriFlag_r](corners_r)
                self.imgpoints_l.append(corners_l)
                self.imgpoints_r.append(corners_r)
            else:
                continue

            img_shape = img_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, img_shape):
        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(self.objpoints, self.imgpoints_l, self.imgpoints_r,
                                                              self.M1, self.d1, self.M2, self.d2, img_shape,
                                                             flags = flags, criteria = stereocalib_criteria)

        self.camera_model = dict([('ret', ret), ('M1', M1), ('M2', M2), ('dist1', d1), ('dist2', d2), ('R', R), ('T', T)])


if __name__ == '__main__':
    filepath = r'G:\2019122011275252\*.bmp'
    OneCamera = CameraCalibration(filepath)
    OneCamera.calibrate_images()
