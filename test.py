#!/usr/bin/env python
# coding: utf-8

from CalibrationClass import CameraCalibration, StereoCalibration
import cv2
import numpy as np
fileaddress = r'G:\2019122011275252\*.bmp'

OneCamera = CameraCalibration(fileaddress)
# OneCamera.show_image()
OneCamera.calibrate_images()
OneCamera.camera_model

# filePath = 'F:\Pictures\haha.jpg'
# src = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
# print(type(src))
# print(src.shape)