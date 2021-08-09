import os
from coco_to_coco_augmented import generate_COCO_Dataset_transformed
from bbaug.policies import policies



ori = 5
dest = 6
scaling = 1/(dest/ori)
aug_policy = policies.policies_pineapple(scaling)

dataset = 'val'
generate_COCO_Dataset_transformed(
    f'datasets/5m_pineapple_d4_5m_t6/{dataset}/',
    f'datasets/5m_pineapple_d4_5m_t6/annotations/instances_{dataset}.json',
    ['pineapple'],
    f'datasets/5m_pineapple_d4_5m/{dataset}',
    f'datasets/5m_pineapple_d4_5m/annotations/instances_{dataset}.json',
    aug_policy,
    0,
    2)

dataset = 'train'
generate_COCO_Dataset_transformed(
    f'datasets/5m_pineapple_d4_5m_t6/{dataset}/',
    f'datasets/5m_pineapple_d4_5m_t6/annotations/instances_{dataset}.json',
    ['pineapple'],
    f'datasets/5m_pineapple_d4_5m/{dataset}',
    f'datasets/5m_pineapple_d4_5m/annotations/instances_{dataset}.json',
    aug_policy,
    0,
    2)

dataset = 'test'
generate_COCO_Dataset_transformed(
    f'datasets/5m_pineapple_d4_5m_t6/{dataset}/',
    f'datasets/5m_pineapple_d4_5m_t6/annotations/instances_{dataset}.json',
    ['pineapple'],
    f'datasets/5m_pineapple_d4_5m/{dataset}',
    f'datasets/5m_pineapple_d4_5m/annotations/instances_{dataset}.json',
    aug_policy,
    0,
    2)