import os
from coco_to_coco_augmented import generate_COCO_Dataset_transformed
from bbaug.policies import policies



ori = 8.0
dest = 9.0
scaling = 1.0/(dest/ori)
aug_policy = policies.policies_pineapple(scaling)

dataset = 'val'
generate_COCO_Dataset_transformed(
    f'datasets/pn_8mts_1_T2/{dataset}/', 
    f'datasets/pn_8mts_1_T2/annotations/instances_{dataset}.json',
    ['pineapple'],
    f'datasets/pn_8mts_1/{dataset}',
    f'datasets/pn_8mts_1/annotations/instances_{dataset}.json',
    aug_policy,
    0,
    2)

dataset = 'train'
generate_COCO_Dataset_transformed(
    f'datasets/pn_8mts_1_T2/{dataset}/',
    f'datasets/pn_8mts_1_T2/annotations/instances_{dataset}.json',
    ['pineapple'],
    f'datasets/pn_8mts_1/{dataset}',
    f'datasets/pn_8mts_1/annotations/instances_{dataset}.json',
    aug_policy,
    0,
    2)

dataset = 'test'
generate_COCO_Dataset_transformed(
    f'datasets/pn_8mts_1_T2/{dataset}/',
    f'datasets/pn_8mts_1_T2/annotations/instances_{dataset}.json',
    ['pineapple'],
    f'datasets/pn_8mts_1/{dataset}',
    f'datasets/pn_8mts_1/annotations/instances_{dataset}.json',
    aug_policy,
    0,
    2)