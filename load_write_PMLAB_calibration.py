#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import numpy as np

def write_to_PMLAB(camera_model, NewfilePath = None):
    M1 = np.squeeze(camera_model['M1'])
    M2 = np.squeeze(camera_model['M2'])
    dist1 = np.squeeze(camera_model['dist1'])
    dist2 = np.squeeze(camera_model['dist2'])
    R = np.squeeze(camera_model['R'])
    angle, jacobian = cv2.Rodrigues(R)
    angle = np.squeeze(angle)
    T = np.squeeze(camera_model['T'])
    
    with open('G:/ML/clb_offline.clbpj','r') as f:
        lines = f.readlines()
    
    replacedata = np.concatenate(([M1[0,2],M1[1,2],M1[0,0],M1[1,1],M1[0,1]], np.zeros(6), np.ravel(np.eye(3)), dist1, np.zeros(8-len(dist1)),
                                  [1], [M2[0,2],M2[1,2],M2[0,0],M2[1,1],M2[0,1]], angle, T, np.ravel(R), dist2, np.zeros(8-len(dist2))))
    k = 0
    for i, line in enumerate(lines):
        if i >= 20 and i <= 76:
            lines[i] = line.split('=')[0] + '=' + str(replacedata[k]) + '\n'
            k = k+1
    if NewfilePath is None:
        NewfilePath = 'new_clb.clbpj'

    with open(NewfilePath,'w') as f:
        f.writelines(lines) 

def load_from_PMLAB(fileaddress):
    with open(fileaddress, 'r') as f:
        lines = f.readlines()

    camera_matrix = {}
    for line in lines:
        if (line.split('\\')[0] == 'results') & (len(line.split('\\')) == 4):
            camera_matrix[line.split('\\')[1] + line.split('\\')[3].split('=')[0]] = float(line.split('\\')[3].split('=')[1])

    cameraMatrix1 = np.identity(3)
    cameraMatrix2 = np.identity(3)
    distCoeffs1 = [camera_matrix['1k1'], camera_matrix['1k2'], camera_matrix['1p1'], camera_matrix['1p2'],
                   camera_matrix['1k3'], camera_matrix['1k4'], camera_matrix['1k5'], camera_matrix['1k6']]
    distCoeffs2 = [camera_matrix['2k1'], camera_matrix['2k2'], camera_matrix['2p1'], camera_matrix['2p2'], 
                   camera_matrix['2k3'], camera_matrix['2k4'], camera_matrix['2k5'], camera_matrix['2k6']]

    cameraMatrix1[0,0] = camera_matrix['1focusX']
    cameraMatrix1[1,1] = camera_matrix['1focusY']
    cameraMatrix1[0,1] = camera_matrix['1focusS']
    cameraMatrix1[0,2] = camera_matrix['1centerX']
    cameraMatrix1[1,2] = camera_matrix['1centerY']

    cameraMatrix2[0,0] = camera_matrix['2focusX']
    cameraMatrix2[1,1] = camera_matrix['2focusY']
    cameraMatrix2[0,1] = camera_matrix['2focusS']
    cameraMatrix2[0,2] = camera_matrix['2centerX']
    cameraMatrix2[1,2] = camera_matrix['2centerY']
    
    
    R = np.array([[camera_matrix['2r11'], camera_matrix['2r12'], camera_matrix['2r13']],
        [camera_matrix['2r21'], camera_matrix['2r22'], camera_matrix['2r23']],
        [camera_matrix['2r31'], camera_matrix['2r32'], camera_matrix['2r33']]])
    T = np.array([camera_matrix['2x'], camera_matrix['2y'], camera_matrix['2z']])

    camera_model = {'M1':cameraMatrix1, 'M2':cameraMatrix2, 'dist1':distCoeffs1, 'dist2':distCoeffs2, 'R':R, 'T':T}
    return camera_model

if __name__ == '__main__':
    filepath = r'G:/ML/clb_offline.clbpj'
    camera_model = load_from_PMLAB(filepath)
    write_to_PMLAB(camera_model)