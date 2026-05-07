import numpy as np

def curve_extraction(original_curves, output_dir):
    output = {
        "ori":{
            "curves": [],
            "dy_dx": [],
            "d2y_dx2": [],
            "feature_names": [],
            "features": [],
            "curves_tsne":[],
            "dy_dx_tsne": [],
            "d2y_dx2_tsne": [],
            "features_tsne": [],
        },
        "moving_avg":{
            "curves": [],
            "dy_dx": [],
            "d2y_dx2": [],
            "feature_names": [],
            "features": [],
            "curves_tsne":[],
            "dy_dx_tsne": [],
            "d2y_dx2_tsne": [],
            "features_tsne": [],
            "params": [],
            "rmse":[],
        },
        "sigmoid_5p":{
            "curves": [],
            "dy_dx": [],
            "d2y_dx2": [],
            "feature_names": [],
            "features": [],
            "curves_tsne":[],
            "dy_dx_tsne": [],
            "d2y_dx2_tsne": [],
            "features_tsne": [],
            "params": [],
            "rmse":[],
        },
    }