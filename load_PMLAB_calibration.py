#!/usr/bin/env python
# coding: utf-8


import cv2 as cv
import os
import numpy as np



def load_CameraMatrix():
    with open('G:/ML/clb_offline.clbpj', 'r') as f:
        lines = f.readlines()

    camera_matrix = {}
    for line in lines:
        if (line.split('\\')[0] == 'results') & (len(line.split('\\')) == 4):
            camera_matrix[line.split('\\')[1] + line.split('\\')[3].split('=')[0]] = float(line.split('\\')[3].split('=')[1])

    cameraMatrix1 = np.identity(3)
    cameraMatrix2 = np.identity(3)
    distCoeffs1 = np.array([0,0,0,0]).T
    distCoeffs2 = np.array([0,0,0,0]).T

    cameraMatrix1[0,0] = camera_matrix['1focusX']
    cameraMatrix1[1,1] = camera_matrix['1focusY']
    cameraMatrix1[0,2] = camera_matrix['1centerX']
    cameraMatrix1[1,2] = camera_matrix['1centerY']

    cameraMatrix2[0,0] = camera_matrix['2focusX']
    cameraMatrix2[1,1] = camera_matrix['2focusY']
    cameraMatrix2[0,2] = camera_matrix['2centerX']
    cameraMatrix2[1,2] = camera_matrix['2centerY']

    R = np.array([[camera_matrix['2r11'], camera_matrix['2r12'], camera_matrix['2r13']],
        [camera_matrix['2r21'], camera_matrix['2r22'], camera_matrix['2r23']],
        [camera_matrix['2r31'], camera_matrix['2r32'], camera_matrix['2r33']]])
    T = np.array([camera_matrix['2x'], camera_matrix['2y'], camera_matrix['2z']])

    imageSize = (2448, 2048)
    Data = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T)
    return Data



Data = load_CameraMatrix()

print(Data)