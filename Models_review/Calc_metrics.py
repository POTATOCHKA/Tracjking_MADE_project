import argparse
import os
from typing import List, Dict
import motmetrics as mm
import pickle

import pandas as pd
import tqdm
import numpy

PATH_TO_DATASETS = ''


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-for-calc', type=str, default='Data/VK_exp_1',
                        help='path data with ground truth and preds')
    opt = parser.parse_args()
    return opt


def get_list_gt_objects(opt: str) -> List:
    ground_truth_folder = os.path.basename(os.path.normpath(opt.data_for_calc)) + '_gt'
    path_to_gt_folder = os.path.join(opt.data_for_calc, ground_truth_folder)
    if os.path.exists(path_to_gt_folder):
        gt_files_paths = [os.path.join(path_to_gt_folder, f) for f in os.listdir(path_to_gt_folder) if
                          os.path.isfile(os.path.join(path_to_gt_folder, f))]
    else:
        print(f'No files is derictory {path_to_gt_folder}')
        return []

    return gt_files_paths


def get_models_predicts_files(opt: argparse, gt_files_paths: List) -> Dict:
    predicts_folder = os.path.basename(os.path.normpath(opt.data_for_calc)) + '_predicts'
    path_to_preds_folder = os.path.join(opt.data_for_calc, predicts_folder)
    models_names = [name for name in os.listdir(path_to_preds_folder)]
    model_to_pred_dict = {}
    for model_name in models_names:
        path_models_preds = os.path.join(path_to_preds_folder, model_name)
        model_preds_files = [f for f in os.listdir(path_models_preds) if
                             os.path.isfile(os.path.join(path_models_preds, f))]

        model_preds_files_paths = [os.path.join(path_models_preds, f) for f in os.listdir(path_models_preds) if
                                   os.path.isfile(os.path.join(path_models_preds, f))]
        gt_files_name = set(map(lambda x: os.path.basename(os.path.normpath(x)), gt_files_paths))
        if set(model_preds_files) == gt_files_name:
            model_to_pred_dict.update({model_name: {os.path.basename(path): path for path
                                                    in model_preds_files_paths}})
        else:
            print(f'mismatch preds and gt files, model_name : {model_name}')
    return model_to_pred_dict


def compute_one_frame(acc, GT_labels, GT_bboxes, preadict_labels, predict_bboxes, max_iou=0.75):
    dists = mm.distances.iou_matrix(GT_bboxes, predict_bboxes, max_iou=max_iou)
    acc.update(GT_labels, preadict_labels, dists)


def compute_one_file(model_name_acc, gt_file_path, pred_file_path):
    with open(gt_file_path, 'rb') as f:
        gt_dict = pickle.load(f)
    with open(pred_file_path, 'rb') as f:
        pred_dict = pickle.load(f)

    for img_n in gt_dict:
        ground_truth_labels = []
        ground_truth_bboxes = []
        strongsort_predict_labels = []
        strongsort_predict_bboxes = []
        for person in gt_dict[img_n]:
            ground_truth_labels.append(person['id'])
            ground_truth_bboxes.append(person['xywh'])
        for person in pred_dict[img_n]:
            strongsort_predict_labels.append(person['id'])
            strongsort_predict_bboxes.append(person['xywh'])
        compute_one_frame(model_name_acc, ground_truth_labels, ground_truth_bboxes, strongsort_predict_labels,
                          strongsort_predict_bboxes)


def calc_metrics(gt_files_paths, model_to_pred_dict):
    model_acc_list = []
    for model_name in model_to_pred_dict:
        model_name_acc = mm.MOTAccumulator(auto_id=True)
        for gt_file_path in tqdm.tqdm(gt_files_paths):
            pred_file_path = model_to_pred_dict[model_name][os.path.basename(gt_file_path)]
            compute_one_file(model_name_acc, gt_file_path, pred_file_path)
        model_acc_list.append(model_name_acc)
    try:
        mh = mm.metrics.create()
        summary = mh.compute_many(
            model_acc_list,
            metrics=['num_frames', 'num_matches', 'num_switches',
                     'num_false_positives', 'num_misses', 'mota',
                     'idf1', 'precision', 'recall'],
            names=list(model_to_pred_dict.keys()))
        summary = summary.astype('float32')
        summary = summary.round(3)
        return summary
    except Exception:
        mh = mm.metrics.create()
        summary = mh.compute_many(
            model_acc_list,
            metrics=['num_frames', 'num_matches', 'num_switches',
                     'num_false_positives', 'num_misses', 'mota',
                     'precision', 'recall'],
            names=list(model_to_pred_dict.keys()))
        summary = summary.astype('float32')
        summary = summary.round(3)
        return summary


def main(opt):
    gt_files_paths = get_list_gt_objects(opt)
    model_to_pred_dict = get_models_predicts_files(opt, gt_files_paths)
    metrics = calc_metrics(gt_files_paths, model_to_pred_dict)
    print(metrics.T)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
