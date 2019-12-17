#!/usr/bin/env python
# coding: utf-8

from CalibrationClass import CameraCalibration, StereoCalibration
import cv2
import numpy as np
fileaddress = r'I:\2019\单目三维视频引伸计\20190610拉伸机测试\spbd图片\*0.bmp'

OneCamera = CameraCalibration(fileaddress)
OneCamera.calibrate_images()

# filePath = 'F:\Pictures\haha.jpg'
# src = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
# print(type(src))
# print(src.shape)