import json
import os
import numpy as np
import psutil
import csv

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess_original, boolean_string

from coco_to_coco_augmented import generate_COCO_Dataset_transformed


def get_predictions(imgs_path, 
                    set_name, 
                    image_ids, 
                    coco, 
                    model, 
                    conf_threshold, 
                    nms_threshold, 
                    input_sizes, 
                    compound_coef, 
                    use_cuda):
    '''
    Run the prediction of bounding boxes and store the results into a file

    Params
    :imgs_path (str) -> path of the images.
    :set_name (str) -> name of the set that is going to be used. E.g. test, val, or train.
    :image_ids (list<int>) -> ids of the images.
    :coco (pycocotools.coco) -> pycocotools for loading the images from the json.
    :model (EfficientDetBackbone) -> model to perform the predictions.
    :conf_threshold (float) -> confidence threshold to filter results.
    :nms_threshold (float) -> non-maximum supression to filter results.
    :input_sizes (list<int>) -> input sizes of the different architectures of EfficientDet.
    :compound_coef (int) -> compound coefficient that indicates the architecture used.
    :use_cuda (bool) -> use gpu or not.

    Return
    :(list<dict>) -> [{'image_id': 0,'category_id': 0,'score': 0.98,'bbox': [0,0,0,0]}, {...}]
    '''
    results = []
    use_float16 = False # by default do not use float 16

    # to transfor boxes
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # iterate over every image
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = imgs_path + image_info['file_name']

        # preprocess image and bounding boxes
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef])
        x = torch.from_numpy(framed_imgs[0])

        # use cuda and floating point precision
        if use_cuda:
            x = x.cuda()
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        # set the proper input for the model
        x = x.unsqueeze(0).permute(0, 3, 1, 2)

        # perform predictions
        features, regression, classification, anchors = model(x)

        # get results in the proper format
        preds = postprocess_original(x,
                                    anchors, 
                                    regression, 
                                    classification,
                                    regressBoxes, 
                                    clipBoxes,
                                    conf_threshold, 
                                    nms_threshold)

        # ommit this image if there are no results
        if not preds:
            continue

        # get boxes in the same size as the original image. E.g. original image is 1000x800 and model uses 512x512.
        preds = invert_affine(framed_metas, preds)[0]
        bbox_score = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # Translate from formats. In this: [x1,y1,x2,y2] -> [x1,y1,w,h]
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            # iterate over all bounding boxes
            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    # write output
    filepath = f'results/{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

    # return number of detections
    return len(results)

#Run the evaluation of the model using pycocotools
def eval_pycoco_tools(image_ids, coco_gt, pred_json_path, max_detect_list):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.maxDets = max_detect_list
    coco_eval.params.imgIds = image_ids
    
    # Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    coco_eval.evaluate()
    
    # Accumulate per image evaluation results and store the result in self.eval
    coco_eval.accumulate()
    
    # Compute and display summary metrics for evaluation results.
    coco_eval.summarize()
    
    #get precision and recall values from
    #   >>iou where the threshold is greater than 0.5
    #       from the values [0.5, 0.55, 0.6, ..., 0.95] --> a total of 10 values.
    #   >>I get all recall thresholds from the 101 interpolation of precision.
    #   >>category is the one from pineapple, actually there is only one category
    #       but, this software detect categories 0 and 1... where 0 is the SUPERCATEGORY IF IT EXISTS IN THE JSON. IF NOT, it is 0.
    #   >>area is related to 'all' from the values: [all, small, medium, large]
    #   >>get the highest max detections... from the values [0, 10, 100] or [10, 100, 1000] or ...
    iou_val = 1
    category_val = 0
    area_val = 0
    maxDet_val = 2
    
    #from the 101 precision vector, get the lowest precision
    precision_temp = coco_eval.eval["precision"][iou_val, :, category_val, area_val, maxDet_val]
    precision_result = precision_temp[np.where(precision_temp > 0)][-1]
    
    #get recall
    recall_result = coco_eval.eval["recall"][iou_val, category_val, area_val, maxDet_val]
    
    #print results
    return precision_result, recall_result


#Run the evaluation of the model using our code
def eval_fh(image_ids, coco_gt, pred_json_path, max_detect_list):
    return 0


def run_metrics(compound_coef, 
                nms_threshold, 
                confidence_threshold, 
                use_cuda,  
                project_name, 
                weights_path, 
                max_detect_list,
                input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536],
                metric_option='coco',
                set_to_use='test',
                augment_dataset=False,
                id_augmentation=0,
                num_of_workers=None,
                batch_size=None):    
    '''
    Method to perform the calculation of the metrics.
    '''


    #load default values, parameters and initialize
    #---------------------------------------------------------------------------------------------------------
    params = yaml.safe_load(open(f'projects/{project_name}.yml'))
    obj_list = params['obj_list']
    
    # load json
    SET_NAME = params[set_to_use + '_set']
    dataset_json = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    dataset_imgs_path = f'datasets/{params["project_name"]}/{SET_NAME}/'


    # augment the dataset before calculating metrics
    #==================================
    if augment_dataset:
        output_folder = f'datasets/{params["project_name"]}/{set_to_use}_{id_augmentation}/'
        gt_augmented_file = f'datasets/{params["project_name"]}/annotations/instances_{set_to_use}_{id_augmentation}.json'
        write_yml = True

        # delete file and folder if exist and then, create them
        if os.path.exists(output_folder):
            write_yml = False
            shutil.rmtree(output_folder)
        if os.path.exists(gt_augmented_file):
            os.remove(gt_augmented_file)
        while os.path.exists(output_folder):
            print("waiting")
            pass
        while os.path.exists(gt_augmented_file):
            print("waiting")
            pass

        # create directories
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # create new file with a fixed name
        new_set_name = f'transformed_set_{id_augmentation}
        if write_yml:
            with open(f'projects/{project_name}.yml', 'a') as my_file:
                my_file.write(f'{new_set_name}: {set_to_use}_{id_augmentation}\n')

        
        #---------
        generate_COCO_Dataset_transformed(output_folder,
                                            gt_augmented_file,
                                            obj_list,
                                            dataset_imgs_path[:len(dataset_imgs_path)-1],
                                            gt_augmented_file,
                                            aug_policy,
                                            num_of_workers,
                                            batch_size)
        #---------

        params = yaml.safe_load(open(f'projects/{project_name}.yml'))
        SET_NAME = params[new_set_name]
        dataset_json = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
        dataset_imgs_path = f'datasets/{params["project_name"]}/{SET_NAME}/'
    #==================================


    # load data set
    coco = COCO(dataset_json)
    image_ids = coco.getImgIds()
    
    # get the number of bboxes from the ground truth
    groundtruth_num = 0
    with open(dataset_json, "r") as read_file:
        data = json.load(read_file)
        groundtruth_num = len(data["annotations"])

    
    #insert file header of results
    if not os.path.exists('results/'):
        os.mkdir('results/')
    with open(f'results/{params["project_name"]}_results_d{compound_coef}.csv', "a") as myfile:
        my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
        my_writer.writerow(["groundtruth_num", "num_detections", "nms_threshold", "confidence_threshold", "precision", "recall", "f1_score"])
    #---------------------------------------------------------------------------------------------------------
       


    # get detections
    #---------------------------------------------------------------------------------------------------------
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                    ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda()

            
    #run the prediction of the bounding boxes and store results into a file
    predictions = get_predictions(dataset_imgs_path, 
                                SET_NAME, 
                                image_ids, 
                                coco, 
                                model, 
                                confidence_threshold, 
                                nms_threshold,
                                input_sizes, 
                                compound_coef, 
                                use_cuda)  
    #---------------------------------------------------------------------------------------------------------


    #evaluate model using the ground truth and the predicted bounding boxes
    if(predictions > 0):
        if metric_option == 'coco':
            p,r = eval_pycoco_tools(image_ids, coco, f'results/{SET_NAME}_bbox_results.json', max_detect_list)
        else:
            print('call to metrics our implementation')
        f1_result = (2.0 * p * r)/ (p + r)
    else:
        p = 0
        r = 0
        f1_result = 0
        
    print()
    print("===============================================================")
    print("Precision:" + str(p))
    print("Recall:" + str(r))
    print("===============================================================")
    
    #store results
    with open(f'results/{params["project_name"]}_results_d{compound_coef}.csv', "a") as myfile:
        my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
        my_writer.writerow([groundtruth_num, predictions, nms_threshold, confidence_threshold, p, r, f1_result])
    #--------------------------------------------


#LIMIT THE NUMBER OF CPU TO PROCESS THE JOB
def throttle_cpu(cpu_list):
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])


#main method to be called
if __name__ == '__main__':
    #throttle_cpu([28,29,30,31,32,33,34,35,36,37,38,39])
    
    #------------------------------------------------------------------------------------------------------------------------------    
    project_name = "apple_c1"
    weights_path = "logs/apple_c1/efficientdet-d0_trained_weights_semi_0.pth"
    compound_coef = 0
    nms_threshold = 0.5
    use_cuda = True
    confidence_threshold = 0.5
    max_detections = [10, 100, 1000]
    #metric_option = "coco"
    #set_to_use='test'
    #augment_dataset=False
    #id_augmentation=1
    #num_of_workers=None
    #batch_size=None
    #------------------------------------------------------------------------------------------------------------------------------

    run_metrics(compound_coef, 
                nms_threshold, 
                confidence_threshold,
                use_cuda,  
                project_name, 
                weights_path,  
                max_detections)

                '''
                run_metrics(compound_coef, 
                nms_threshold, 
                confidence_threshold, 
                use_cuda,  
                project_name, 
                weights_path, 
                max_detect_list,
                input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536],
                metric_option='coco',
                set_to_use='test',
                augment_dataset=False,
                id_augmentation=1,
                num_of_workers=None,
                batch_size=None):'''