import numpy as np

def rand_box(weight, height, lam):
    cut_ratio = np.sqrt(1 - lam)
    cut_w = np.int(weight*cut_ratio)
    cut_h = np.int(height*cut_ratio)

    cx = np.random.randint(weight)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w//2, 0, weight)
    bby1 = np.clip(cy - cut_h//2, 0, height)
    bbx2 = np.clip(cx + cut_w//2, 0, weight)
    bby2 = np.clip(cy + cut_h//2, 0, height)

    return bbx1, bby1, bbx2, bby2

def efficientnet_params(model_name):
    dict_params = {
        "efficientnet-b1": {
            "input_res": 240,
            "ds_blocks": [15, 22],
            "fuse_linear": 1024,
            "last_linear": 1280
        },
        "efficientnet-b2": {
            "input_res": 260,
            "ds_blocks": [15, 22],
            "fuse_linear": 1128,
            "last_linear": 1408
        },
        "efficientnet-b3": {
            "input_res": 300,
            "ds_blocks": [17, 25],
            "fuse_linear": 1232,
            "last_linear": 1536
        },
        "efficientnet-b4": {
            "input_res": 380,
            "ds_blocks": [21, 31],
            "fuse_linear": 1432,
            "last_linear": 1792
        },
        "efficientnet-b5": {
            "input_res": 456,
            "ds_blocks": [26, 38],
            "fuse_linear": 1640,
            "last_linear": 2048
        },
        "efficientnet-b6": {
            "input_res": 528,
            "ds_blocks": [30, 44],
            "fuse_linear": 1840,
            "last_linear": 2304
        },
        "efficientnet-b7": {
            "input_res": 600,
            "ds_blocks": [37, 54],
            "fuse_linear": 2048,
            "last_linear": 2560
        }
    }

    params = dict_params[model_name]

    return params