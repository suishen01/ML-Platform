import os
import warnings
import argparse
from Utils.csvread import CsvReader
from DataPreprocessor.dataSQL import SQL
from MachineLearningModels.ridge import Ridge
from MachineLearningModels.adaboost import AdaBoost
from MachineLearningModels.gradientboost import GradientBoost
from MachineLearningModels.randomforest import RandomForest
from MachineLearningModels.svm import KernelSVM
from MachineLearningModels.ann import NeuralNetwork
from MachineLearningModels.cnn import ConvolutionalNeuralNetwork
from MachineLearningModels.lstm import LSTMModel
from MachineLearningModels.pca import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_arg_parser():
    parser = argparse.ArgumentParser(description="Automatically generates score table.")

    parser.add_argument('-train', '--trainingdata', help="Get the training data", required=True)
    parser.add_argument('-in', '--input', help="Inputs for the ML model", required=True)
    parser.add_argument('-out', '--output', help="Outputs for the ML model", required=True)
    parser.add_argument('-t', '--type', help="Prediction type, c for classification, r for regression", required=True)

    parser.add_argument('-vr', '--validationratio',type=float, default=0.9, help='Validation Ratio (default: 0.7)',required=False)
    parser.add_argument('-index', '--index', help="Index array for plotting", required=False)
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
    elif model_type == 'KernelSVM':
        model = KernelSVM(kernel=configs['kernel'], degree=configs['degree'])
    elif model_type == 'NeuralNetwork':
        model = NeuralNetwork(feature_headers, label_headers, epochs=configs['epochs'], batch_size=configs['batch_size'], type=prediction_type)
    elif model_type == 'ConvolutionalNeuralNetwork':
        model = ConvolutionalNeuralNetwork(height=configs['height'], width=configs['width'], dimension=configs['dimension'], classes=configs['classes'], epochs=configs['epochs'], batch_size=configs['batch_size'])
    elif model_type == 'LSTM':
        model = LSTMModel(feature_headers, label_headers, epochs=configs['epochs'], batch_size=configs['batch_size'], type=prediction_type, lookback=configs['lookback'], num_of_cells=configs['num_of_cells'])
    elif model_type == 'PCA':
        model = PCA(n_components=configs['n_components'], type=prediction_type)
    else:
        print(model_type, ' is not implemented yet')
        model = None

    return model

def produce_report(model, reports, test_labels, predictions, label_headers, indexarray, figpath):
    dict = {}
    for report in reports:
        if report == 'Accuracy':
            dict['Accuracy'] = model.getAccuracy(test_labels, predictions)
        elif report == 'ConfusionMatrix':
            model.getConfusionMatrix(test_labels, predictions, label_headers)
        elif report == 'RSquare':
            dict['RSquare'] = model.getRSquare(test_labels, predictions)
        elif report == 'MSE':
            dict['MSE'] = model.getMSE(test_labels, predictions)
        elif report == 'MAPE':
            dict['MAPE'] = model.getMAPE(test_labels, predictions)
        elif report == 'RMSE':
            dict['RMSE'] = model.getRMSE(test_labels, predictions)
        elif report == 'FeatureImportance':
            dict['FeatureImportance'] = model.featureImportance()
        elif report == 'Plot':
            if model.type == 'classifier':
                print('No plotting for classification')
            else:
                df = pd.DataFrame(data=predictions.flatten())
                test_labels = test_labels.reset_index(drop=True)
                plt.plot(df)
                plt.plot(test_labels, 'r')
                plt.savefig(figpath)
    return dict


if __name__ == "__main__":
    parser = init_arg_parser()
    args = parser.parse_args()

    csvreader = CsvReader()
    alldata = csvreader.read(args.trainingdata)

    feature_headers = read_list(args.input)
    label_headers = read_list(args.output)

    if args.type == 'c':
        type = 'classifier'
    else:
        type = 'regressor'

    alldata = alldata.fillna(0)
    data = alldata

    if args.validationratio:
        vr = args.validationratio
    else:
        vr = 0.6

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

    if args.index:
        indexarrays = read_list(args.index)
        indexarray = indexarrays[0]
    else:
        print('Please specify the index array')

    if args.reports:
        reports_path = args.reports
    else:
        reports_path = 'reports.txt'

    reports = read_list(reports_path)
    labels = data[label_headers].copy()
    features = data[feature_headers].copy()
    indices = data[indexarray].copy()

    train_features, test_features = np.split(features, [int(vr*len(features))])
    train_labels, test_labels = np.split(labels, [int(vr*len(labels))])
    train_indices, test_indices = np.split(indices, [int(vr*len(indices))])

    if validation_data:
        test_labels = validation_data[labels].copy()
        test_features = validation_data[features].copy()

    models_list = []

    for model_type in models:
        if '[' in model_type:
            model_type = model_type.replace('[','')
            model_type = model_type.replace(']','')
            sub_models = model_type.split(',')
            sub_models_list = []
            for sub_model_type in sub_models:
                sub_model_type = sub_model_type.lstrip()
                sub_model_type = sub_model_type.rstrip()
                if sub_model_type in configs.keys():
                    sub_model = build_model(sub_model_type, type, configs[sub_model_type], feature_headers, label_headers)
                    if sub_model:
                        sub_models_list.append(sub_model)
                else:
                    print('No valid configurations for sub model ', model_type)
            models_list.append(sub_models_list)
        else:
            if model_type in configs.keys():
                model = build_model(model_type, type, configs[model_type], feature_headers, label_headers)
                if model:
                    models_list.append(model)
            else:
                print('No valid configurations for model ', model_type)

    results = []
    modelindex = 0
    for model in models_list:
        if isinstance(model, list):
            tmp_train_features = model[0].fit(train_features, train_labels)
            model[1].fit(tmp_train_features, train_labels)
            model[1].save()
            tmp_test_features = model[0].fit(test_features, None)
            predictions = model[1].predict(tmp_test_features)
            figpath = str(modelindex) + '.png'
            result = produce_report(model[1], reports, test_labels, predictions, label_headers, test_indices, figpath)
            results.append(result)
        else:
            dict = {}
            model.fit(train_features, train_labels)
            model.save()
            predictions = model.predict(test_features)
            figpath = str(modelindex) + '.png'
            result = produce_report(model, reports, test_labels, predictions, label_headers, test_indices, figpath)
            results.append(result)
        modelindex = modelindex + 1

    index = 0
    result_path = 'results.txt'
    with open(result_path, 'w') as f:
        for item in results:
            f.write(models[index])
            f.write(':\n')
            f.write("%s\n" % item)
            index += 1
