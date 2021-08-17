import cv2
import os
import pandas as pd
import numpy as np
import csv
import glob
import sys
import yaml
from tqdm import tqdm
import argparse
from pycocotools.coco import COCO

from conflict_solving import *
from utils.inference_detectors import getPredictionsBBox
from utils.image_splitter import myImageSlicer
from utils.printers import *
from utils.utils import boolean_string

'''
This method generates the regions of interest (ROI) from the entire image,
it returns the conflict solving inferences and the inference without conflict solving.

Params:
pathOfWeights (str) -> The path of where the weights file was stored for EfficientDet model.

completeImage (str) ->  The image location path and format of it (.jpg or .png)

nms_threshold (float) -> confidence threshold to filter results.

Return:
bounding boxes format (bboxes) = [label,score,xmin,ymin,xmax,ymax]

rois (list<bboxes>) ->  All the bounding boxes produced by the network with out the conflict solving module.
rois_cf (list<bboxes>) ->  All the bounding boxes produced by the network with the conflict solving module activated.

'''
def generateROIS(pathOfWeights,completeImage,nms_threshold,iou_threshold,compound_coef,use_cuda):
    ### Important data from the original Image
    originalImage = cv2.imread(completeImage)
    heightOriginalImage = originalImage.shape[0]
    widthOriginalImage = originalImage.shape[1]
    #os.mkdir('/content/slices')
    cropped_images=myImageSlicer(1014,1140,completeImage)
    ### Important data from the pieces of images
    pieceImageSample = cropped_images[0][0]
    heightPieces = pieceImageSample.shape[0]
    widthPieces = pieceImageSample.shape[1]
    ##Getting the dimensions of the detection matrix
    columnsGrid = widthOriginalImage/widthPieces
    rowsGrid = heightOriginalImage/heightPieces
    rois = [] ##This matrix will contain the bunch of 
    #bbox coordinates(from slices images) traduced them to the original image (drone image)
    border = []

    ### This coming while loop will build the matrix containing the bbox coordinates
    #for i in tqdm(range(int(rowsGrid)),desc='Row of the image'):
    for i in range(int(rowsGrid)):
        for j in range(int(columnsGrid)):
            ## Obtaining the detection info from a single slice image
            out = getPredictionsBBox(cropped_images[i][j],pathOfWeights,nms_threshold,iou_threshold,compound_coef,use_cuda)
            ## Analizing the bunch of detections from the slice image
            for bboxCount in range(len(out[0]['rois'])):
                ## Translating the slice coordinates to the original image
                (x1, y1, x2, y2) = out[0]['rois'][bboxCount].astype(np.int)
                (nx1,ny1,nx2,ny2) = (x1+((j)*int(widthPieces)), y1+((i)*int(heightPieces)) , x2+((j)*int(widthPieces)), y2+((i*1)*int(heightPieces)))
                out[0]['rois'][bboxCount] = (int(nx1),int(ny1),int(nx2),int(ny2))
                ## Here we evaluate if a bounding box is a possible conflict
                #------------------------------
                isSuspiciousConflict = evaluateLabel((nx1,ny1,nx2,ny2),rowsGrid, columnsGrid,widthPieces,heightPieces,2)
                if isSuspiciousConflict:
                    border.append((int(nx1),int(ny1),int(nx2),int(ny2)))
                
                rois.append([out[0]['class_ids'][bboxCount]+1, float(out[0]['scores'][bboxCount]), nx1,ny1,nx2,ny2])
    
    pairsCollected,conflictsZones = findingPairConflicts(border, 2)  
    confResolution = []
    #for i in tqdm(range(len(conflictsZones)),desc='Solving conflicts'):
    for i in range(len(conflictsZones)):
        conflictZ = conflictsZones[i]
        out = solveConflict(conflictZ, originalImage, heightPieces, widthPieces, pathOfWeights,nms_threshold,iou_threshold,compound_coef,use_cuda)
        if len(out)>0:
            for bboxCount in range(len(out[0]['rois'])):
                (nx1,ny1,nx2,ny2) = out[0]['rois'][bboxCount]
                confResolution.append([out[0]['class_ids'][bboxCount]+1, float(out[0]['scores'][bboxCount]), nx1,ny1,nx2,ny2])   

    rois_cf = removeConflicts(rois,pairsCollected)
    rois_cf = rois_cf + confResolution
    return rois, rois_cf

def run_over_dataset(pathOfWeights,project_name,nms_threshold,iou_threshold,compound_coef,use_cuda,metric):
    params = yaml.safe_load(open(f'projects/{project_name}.yml'))
    obj_list = params['obj_list']
    SET_NAME = params['test_set']
    coco = COCO(f'datasets/{project_name}/annotations/instances_{SET_NAME}.json')
    ids = list(sorted(coco.imgs.keys()))
    for i in tqdm(range(len(ids)),desc='Images processed'):
        img_id = ids[i]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # image name for input image
        imgName = coco.loadImgs(img_id)[0]['file_name']
        num_objs = len(coco_annotation)
        imagePath = f'datasets/{project_name}/{SET_NAME}/{imgName}'
        rois, rois_cf = generateROIS(pathOfWeights,imagePath,nms_threshold,iou_threshold,compound_coef,use_cuda)
        print('############################################################')
        print(f'Image processed:{imgName}')
        print(f'Number of objects with NO conflict resolution:{len(rois)}')
        print(f'Number of objects with conflict resolution:{len(rois_cf)}')
        print('############################################################')

def get_args():
    """Get all expected parameters"""
    parser = argparse.ArgumentParser('EfficientDet Pytorch - Evaluate the model')
    parser.add_argument('-p', '--project', type=str, default="")
    parser.add_argument('-c', '--compound_coef', type=int, default=4)
    parser.add_argument('-w', '--weights', type=str, default="") 
    parser.add_argument('--nms_thres', type=float, default=0.4)
    parser.add_argument('--iou_thres', type=float, default=0.4)
    parser.add_argument('--use_cuda', type=boolean_string, default=False) 
    parser.add_argument('--metric', type=str, default="simple")

    args = parser.parse_args()
    return args

    #main method to be called
if __name__ == '__main__':
    opt = get_args()

    # main method to measure performance
    run_over_dataset(opt.weights,
                     opt.project,
                     opt.nms_thres,
                     opt.iou_thres,
                     opt.compound_coef,
                     opt.use_cuda,
                     opt.metric)