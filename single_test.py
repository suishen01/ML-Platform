import os
import warnings
import argparse
from Utils.csvread import CsvReader
from MachineLearningModels.model import Model
import numpy as np
from MachineLearningModels.ridge import Ridge
from MachineLearningModels.adaboost import AdaBoost
from MachineLearningModels.gradientboost import GradientBoost
from MachineLearningModels.randomforest import RandomForest
from MachineLearningModels.ann import NeuralNetwork
from MachineLearningModels.pca import PCA
import pandas as pd

def init_arg_parser():
    parser = argparse.ArgumentParser(description="Single model prediction")

    parser.add_argument('-train', '--trainingdata', help="Get the training data", required=True)
    parser.add_argument('-in', '--input', help="Inputs for the ML model", required=True)
    parser.add_argument('-out', '--output', help="Outputs for the ML model", required=True)
    parser.add_argument('-m', '--modeltype', help="Model type", required=True)
    parser.add_argument('-t', '--type', help="Prediction type, c for classification, r for regression", required=True)
    parser.add_argument('-mp', '--modelpath', help="Model path", required=True)
    parser.add_argument('-rp', '--resultspath', help="Results path", required=True)

    return parser

def read_list(path):
    return open(path, 'r').read().splitlines()

def load_model(path, model_type, type, feature_headers, label_headers):
    if model_type == 'Ridge':
        model = Ridge()
        model.load(path, type)
    elif model_type == 'AdaBoost':
        model = AdaBoost()
        model.load(path, type)
    elif model_type == 'GradientBoost':
        model = GradientBoost()
        model.load(path, type)
    elif model_type == 'RandomForest':
        model = RandomForest()
        model.load(path, type)
    elif model_type == 'NeuralNetwork':
        model = NeuralNetwork(feature_headers, label_headers)
        model.load(path, type)
    else:
        print(model_type, ' is not implemented yet')
        model = None
    return model

if __name__ == "__main__":
    parser = init_arg_parser()
    args = parser.parse_args()

    csvreader = CsvReader()
    data = csvreader.read(args.trainingdata)

    feature_headers = read_list(args.input)
    label_headers = read_list(args.output)

    if args.type == 'c':
        type = 'classifier'
    else:
        type = 'regressor'

    model_type = args.modeltype

    modelpath = args.modelpath

    labels = data[label_headers].copy()
    features = data[feature_headers].copy()

    model = None
    if '[' in model_type:
        model_type = model_type.replace('[','')
        model_type = model_type.replace(']','')
        sub_models = model_type.split(',')
        sub_models_list = []
        index = 0
        for sub_model_type in sub_models:
            sub_model_type = sub_model_type.lstrip()
            sub_model_type = sub_model_type.rstrip()
            if index == 0:
                tmp_features = PCA.fit(features)
            else:
                model = load_model(modelpath, model_type, type, feature_headers, label_headers)
            index = 1
    else:
        model = load_model(modelpath, model_type, type, feature_headers, label_headers)

    predictions = model.predict(features)
    #print(predictions)
    dict = {}

    index = 0
    for label_header in label_headers:
        dict[label_header] = predictions
        index += 1

    df = pd.DataFrame(dict)

    df.to_csv(args.resultspath)
