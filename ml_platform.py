import os
import warnings
import argparse
from Utils.csvread import CsvReader
from MachineLearningModels.ridge import Ridge
from MachineLearningModels.adaboost import AdaBoost
from MachineLearningModels.gradientboost import GradientBoost
from MachineLearningModels.randomforest import RandomForest
from MachineLearningModels.ann import NeuralNetwork
from MachineLearningModels.pca import PCA
import numpy as np

def init_arg_parser():
    parser = argparse.ArgumentParser(description="Automatically generates score table.")

    parser.add_argument('-train', '--trainingdata', help="Get the training data", required=True)
    parser.add_argument('-in', '--input', help="Inputs for the ML model", required=True)
    parser.add_argument('-out', '--output', help="Outputs for the ML model", required=True)
    parser.add_argument('-t', '--type', help="Prediction type, c for classification, r for regression", required=True)

    parser.add_argument('-vr', '--validationratio',type=float, default=0.9, help='Validation Ratio (default: 0.7)',required=False)
    parser.add_argument('-val', '--validation', help='Validation dataset',required=False)
    parser.add_argument('-c','--config', help='Configuration file',required=False)
    parser.add_argument('-m','--models', help='Involved ML models', required=False)
    parser.add_argument('-r','--reports', help='Produced results', required=False)
    return parser

def read_dict(path):
    return eval(open(path, 'r').read())

def read_list(path):
    return open(path, 'r').read().splitlines()

def build_model(model_type, prediction_type, configs, feature_headers, label_headers):
    if model_type == 'Ridge':
        model = Ridge(alpha=configs['alpha'], type=prediction_type)
    elif model_type == 'AdaBoost':
        model = AdaBoost(n_estimators=configs['n_estimators'], type=prediction_type)
    elif model_type == 'GradientBoost':
        model = GradientBoost(n_estimators=configs['n_estimators'], type=prediction_type)
    elif model_type == 'RandomForest':
        model = RandomForest(n_estimators=configs['n_estimators'], type=prediction_type)
    elif model_type == 'NeuralNetwork':
        model = NeuralNetwork(feature_headers, label_headers, epochs=configs['epochs'], batch_size=configs['batch_size'], type=prediction_type)
    elif model_type == 'PCA+Ridge':
        model = PCA(n_components=configs['n_components'], type=prediction_type)
    else:
        print(model_type, ' is not implemented yet')
        model = None

    return model

def produce_report(model, reports, test_labels, predictions, label_headers):
    dict = {}
    for report in reports:
        if report == 'Accuracy':
            dict['Accuracy'] = model.getAccuracy(test_labels, predictions)
        elif report == 'ConfusionMatrix':
            model.getConfusionMatrix(test_labels, predictions, label_headers)
    return dict


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

    if args.validationratio:
        vr = args.validationratio
    else:
        vr = 0.9

    if args.validation:
        validation_data = csvreader.read(args.validation)
        vr = 1
    else:
        validation_data = None

    if args.config:
        config_path = args.config
    else:
        config_path = 'configs.txt'

    configs = read_dict(config_path)

    if args.models:
        models_path = args.models
    else:
        models_path = 'models.txt'

    models = read_list(models_path)

    if args.reports:
        reports_path = args.reports
    else:
        reports_path = 'reports.txt'

    reports = read_list(reports_path)

    labels = data[label_headers].copy()
    features = data[feature_headers].copy()

    train_features, test_features = np.split(features, [int(vr*len(features))])
    train_labels, test_labels = np.split(labels, [int(vr*len(labels))])

    if validation_data:
        test_labels = validation_data[labels].copy()
        test_features = validation_data[features].copy()

    models_list = []

    for model_type in models:
        if model_type in configs.keys():
            model = build_model(model_type, type, configs[model_type], feature_headers, label_headers)
            if model:
                models_list.append(model)
        else:
            print('No valid configurations for model ', model_type)

    results = []
    for model in models_list:
        model.fit(train_features, train_labels)
        model.save()
        predictions = model.predict(test_features)
        result = produce_report(model, reports, test_labels, predictions, label_headers)
        results.append(result)

    index = 0
    with open('results.txt', 'w') as f:
        for item in results:
            f.write(models[index])
            f.write(':\n')
            f.write("%s\n" % item)
            index += 1
