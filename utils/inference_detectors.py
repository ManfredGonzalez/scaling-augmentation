import torch
from torch.backends import cudnn
import sys
import os
import json

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes

from utils.utils import invert_affine, postprocess_original, preprocess_single

'''
This method returns the bounding boxes and its metadata from a single input image

directoryOfImages: this is the directoy path where the image is stored

imageName: this is how the image is named includding its file format(.jpg,.png,...)
'''
def getPredictionsBBox(image,weights_path,nms_threshold,iou_threshold,compound_coef,use_cuda):
    compound_coef = 4
    force_input_size = None  # set None to use default size
    threshold = nms_threshold
    iou_threshold = 0.4

    
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ['pineapple']
    results = []
    predictions_boxes = []
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    #ori_imgs, framed_imgs, framed_metas = preprocess(image, max_size=input_size)
    ori_imgs, framed_imgs, framed_metas = preprocess_single(image, max_size=input_size)

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

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                                # replace this part with your project's anchor config
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess_original(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    return out

