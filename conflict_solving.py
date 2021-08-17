import matplotlib.pyplot as plt
import cv2
import csv
import os
import pandas as pd
import numpy as np
import torch
from torch.backends import cudnn
import sys
import json

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes

from utils.inference_detectors import getPredictionsBBox
from utils.utils import invert_affine, postprocess_original, preprocess_single
from utils.printers import printBoarderROIS, printConflictsSolutionROIS, drawMatrix, printbboxesoutput

'''
This method is where a bounding box is evaluated as a possible conflict.

A possible conflict refers to a label that is very nearby a limit where the 
the original image was cut. 

label: it is an array with the 4 coordinates of the input bounding box

numOfRows: This variable establishes the number of rows in which the original image was split

numOfColumns: This variable establishes the number of columns in which the original image was split

widthCells: Every slice of the original image has a width, this variable matches that slice width

heightCells: Every slice of the original image has a height, this variable matches that slice height

margin: This variable is very important because establishes how far must be bounding box to be considered 
'''
def evaluateLabel(label,numOfRows, numOfColumns, widthCells, heightCells,margin):
  i = 1
  x1,y1,x2,y2 = label
  while i<=numOfRows:
    j=1
    while j<=numOfColumns:
      #cv2.rectangle(image, ((j*widthCells)-widthCells, (i*heightCells)-heightCells), ((j*widthCells), (i*heightCells)), (0, 0, 255), 2)
      upperLeft = (j*widthCells)-widthCells
      upper = (i*heightCells)-heightCells

      lowerRight = (j*widthCells)
      lower = (i*heightCells)
      if abs(x1-upperLeft)<=margin or abs(y1-upper)<=margin or abs(x2-lowerRight)<=margin or abs(y2-lower)<=margin:
        return True
      j = j+1
    i = i+1
  return False
'''
This method filters the bounding boxes that are in front of another bounding box with a certain distance

bordermatrix: This variable is a list of bounding boxes considered as a possible conflict

margin: this is the distance between a pair of bounding boxes considered a conflict.
This param is multiplied by 3 because the margin is the space between the bounding box and the line, 
also the line thickness and the space between the other bounding box and the line.
'''
def findingPairConflicts(bordermatrix, margin):
  pairs=[]
  conflictROIs = []
  for (xa1,ya1,xa2,ya2) in bordermatrix:##Label A
    
    for counter in range(len(bordermatrix)):##Label B for bboxCount in range(len(out[0]['rois']))
      (xb1,yb1,xb2,yb2)= bordermatrix[counter]
      if (xa1,ya1,xa2,ya2)!=(xb1,yb1,xb2,yb2) and ((xb1,yb1,xb2,yb2)!=(0,0,0,0)):##Making sure that we won't be evaluating the same label
        ##Getting the center of the label B
        cbx,cby = ((xb1 + xb2) / 2),((yb1 + yb2) / 2)
        distanceBtwABx = abs(xb1 - xa2)
        distanceBtwABy = abs(yb1 - ya2)
        if (distanceBtwABx <= (margin * 3)) and ((cby >= ya1) and (cby <= ya2)):
          ##Case where the labels are infront in the x axis
          pairs.append([(xa1,ya1,xa2,ya2),(xb1,yb1,xb2,yb2)])

          newy1 = ya1
          newy2 = ya2
          if ya1 <= yb1:
            newy1 = yb1
          if ya2 <= yb2:
            newy2 = yb2
          
          
          conflictROIs.append((xa1,newy1,xb2,newy2))
          bordermatrix[counter] = (0,0,0,0)  
          #break;
        elif (distanceBtwABy <= (margin * 3)) and ((cbx >= xa1) and (cbx <= xa2)):
          ##Case where the labels are infront in the y axis
          pairs.append([(xa1,ya1,xa2,ya2),(xb1,yb1,xb2,yb2)])

          newx1 = xa1
          newx2 = xa2
          if xa1 <= xb1:
            newx1 = xb1
          if xa2 <= xb2:
            newx2 = xb2

          conflictROIs.append((newx1,ya1,newx2,yb2))
          bordermatrix[counter] = (0,0,0,0)  
          #break;
  return pairs,conflictROIs

'''
This method is a very important one in our approach, this method receives the conflict zone ( that zone
that surrounds both bounding boxes of a conflict), makes a forward of EfficientDet with that conflict and
returns the prediction or predictions that solves the conflict.

conflictZone: this variable means that zone that surrounds both bounding boxes of a conflict and it is
a 2 point coordinates like (x1,y1,x2,y2) referring to the bounding box

bigOriginalImage: the original large scale image that were sliced

heightPieces: This remains the height of every slice taken from the original image

widthPieces: This remains the width of every slice taken from the original image

pathOfWeights: This variable refers to the path of where the weights file was stored for EfficientDet model.
'''
def solveConflict(conflictZone, bigOriginalImage, heightPieces, widthPieces, pathOfWeights,nms_threshold,iou_threshold,compound_coef,use_cuda):
  blank_image = np.zeros((heightPieces,widthPieces,3), np.uint8)
  xcenter, ycenter = (int((0+widthPieces)/2),int((0+heightPieces)/2))


  (x1,y1,x2,y2) = conflictZone
  w1 = x2-x1
  h1 = y2-y1
  roi1 = bigOriginalImage[y1:y2,x1:x2]
  blank_image[int(ycenter-h1/2):int(ycenter+h1/2),int(xcenter-w1/2):int(xcenter+w1/2)] = roi1

  output = getPredictionsBBox(blank_image,pathOfWeights,nms_threshold,iou_threshold,compound_coef,use_cuda)

  if len(output[0]['rois'])<=2:

    #printbboxesoutput(blank_image,output)

    #cv2.imwrite('/content/'+"ConflictSolved.jpg", blank_image)

    inferencesTranslated = []
    for ix1,iy1,ix2,iy2 in output[0]['rois']:
      (dx1,dy1,dx2,dy2) = (abs(int(xcenter-w1/2)-ix1),abs(int(ycenter-h1/2)-iy1),int(xcenter+w1/2)-ix2,int(ycenter+h1/2)-iy2)

      inferencesTranslated.append((int(x1+dx1),int(y1+dy1),int(x2-dx2),int(y2-dy2)))

    output[0]['rois'] = inferencesTranslated
    #print(output)
    return output
  return []
'''
This method received the ROI's from all slices and remove the conflict on it.

matrix: this variable is the matrix that contains all the bounding boxes per slice.

pairsCollected: this variable is an array that contains pairs of bounding boxes representing a 
conflict.
'''
def removeConflicts(matrix_to_edit,pairsCollected):
  matrix = matrix_to_edit
  newMatrix = []
  for rois in matrix:
    is_a_conflict = False
    #if column['rois'][i] in listSolutions[0]['rois'][0]:
    (x1,y1,x2,y2) = rois[2:]
    for i in range(len(pairsCollected)):
      pair = pairsCollected[i]
      if (x1,y1,x2,y2) in pair:
        is_a_conflict = True
        i = len(pairsCollected)
    if not is_a_conflict:    
      newMatrix.append(rois)
  return newMatrix
