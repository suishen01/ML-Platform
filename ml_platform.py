import os
import warnings
import argparse
from Utils.csvread import CsvReader
from DataPreprocessor.dataSQL import SQL
from MachineLearningModels.ridge import Ridge
from MachineLearningModels.decisiontree import DecisionTree
from MachineLearningModels.adaboost import AdaBoost
from MachineLearningModels.gradientboost import GradientBoost
from MachineLearningModels.randomforest import RandomForest
from MachineLearningModels.svm import KernelSVM
from MachineLearningModels.ann import NeuralNetwork
from MachineLearningModels.cnn import ConvolutionalNeuralNetwork
from MachineLearningModels.lstm import LSTMModel
from MachineLearningModels.pca import PCA
from MachineLearningModels.lasso import Lasso
from MachineLearningModels.pls import PLS
from MachineLearningModels.elasticnet import ElasticNet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_arg_parser():
    parser = argparse.ArgumentParser(description="Automatically generates score table.")

    parser.add_argument('-train', '--trainingdata', help="Get the training data", required=True)
    parser.add_argument('-in', '--input', help="Inputs for the ML model", required=True)
    parser.add_argument('-out', '--output', help="Outputs for the ML model", required=True)
    parser.add_argument('-t', '--type', help="Prediction type, c for classification, r for regression", required=True)
    parser.add_argument('-index', '--index', help="Index array for plotting", required=True)

    parser.add_argument('-pp', '--predictionpath', help="Predictions path", required=False)
    parser.add_argument('-kfold', '--kfold', help="Enable kfold validation",type=int, default=1, required=False)
    parser.add_argument('-rp', '--resultpath', help="Results path", required=False)
    parser.add_argument('-fp', '--figurepath', help="Figures path", required=False)
    parser.add_argument('-tr', '--testratio',type=float, default=0.7, help='Test Ratio (default: 0.7)',required=False)
    parser.add_argument('-test', '--test', help='Validation dataset',required=False)
    parser.add_argument('-c','--config', help='Configuration file',required=False)
    parser.add_argument('-m','--models', help='Involved ML models', required=False)
    parser.add_argument('-r','--reports', help='Produced results', required=False)
    parser.add_argument('-origin','--origin', help='For classification accuracy', required=False)
    parser.add_argument('-hmr','--hitmissratio', help='For classification accuracy', required=False)
    return parser

def read_dict(path):
    return eval(open(path, 'r').read())

def read_list(path):
    return open(path, 'r').read().splitlines()

def build_model(model_type, prediction_type, configs, feature_headers, label_headers):
    if model_type == 'Ridge':
        model = Ridge(label_headers=label_headers, alpha=configs['alpha'], type=prediction_type)
    elif model_type == 'DecisionTree':
        model = DecisionTree(label_headers=label_headers, max_depth=configs['max_depth'], type=prediction_type)
    elif model_type == 'AdaBoost':
        model = AdaBoost(label_headers=label_headers, n_estimators=configs['n_estimators'], type=prediction_type)
    elif model_type == 'GradientBoost':
        model = GradientBoost(label_headers=label_headers, n_estimators=configs['n_estimators'], type=prediction_type)
    elif model_type == 'RandomForest':
        model = RandomForest(label_headers=label_headers, n_estimators=configs['n_estimators'], type=prediction_type)
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
    elif model_type == 'PLS':
        model = PLS(n_components=configs['n_components'], type=prediction_type)
    elif model_type == 'Lasso':
        model = Lasso(label_headers=label_headers, alpha=configs['alpha'], type=prediction_type)
    elif model_type == 'ElasticNet':
        model = ElasticNet(label_headers=label_headers, l1_ratio=configs['l1_ratio'], type=prediction_type)
    else:
        print(model_type, ' is not implemented yet')
        model = None

    return model

def produce_report(model, reports, test_labels, predictions, feature_headers, label_headers, indexarray, figpath, origin=None, hitmissr=None):
    dict = {}
    dataframe = None
    for report in reports:
        if report == 'Accuracy':
            kwargs = {'origin':origin, 'hitmissr':hitmissr}
            dict['Accuracy'] = model.getAccuracy(test_labels, predictions, {k: v for k, v in kwargs.items() if v is not None})

        elif report == 'ConfusionMatrix':
            model.getConfusionMatrix(test_labels, predictions, label_headers)
        elif report == 'ROC':
            model.getROC(test_labels, predictions, label_headers)
        elif report == 'RSquare':
            dict['RSquare'] = model.getRSquare(test_labels, predictions)
        elif report == 'MSE':
            dict['MSE'] = model.getMSE(test_labels, predictions)
        elif report == 'MAPE':
            dict['MAPE'] = model.getMAPE(test_labels, predictions)
        elif report == 'RMSE':
            dict['RMSE'] = model.getRMSE(test_labels, predictions)
        elif report == 'FeatureImportance':
            fis = model.featureImportance()
            i = 0
            for fi in fis:
                dict['FeatureImportance_'+feature_headers[i]] = fi
                i = i + 1
        elif report == 'Plot':
            if model.type == 'classifier':
                print('No plotting for classification')
            else:
                df = pd.DataFrame(data=predictions.flatten())
                test_labels = test_labels.reset_index(drop=True)
                plt.figure()
                p, = plt.plot(df, label='prediction')
                a, = plt.plot(test_labels, label='actual')
                if indexarray.shape[0] > 50:
                    for i in indexarray.index:
                        mod10 = int(indexarray.shape[0]/10)
                        if i % (int(indexarray.shape[0]/mod10)) != 0:
                            indexarray[i] = ''
                plt.xticks(df.index, indexarray, rotation='vertical')
                plt.legend(handles=[p, a])
                plt.savefig(figpath)
    df = pd.DataFrame(dict.items(), columns=['key', 'value'])
    df = df.sort_values(by=['key'])
    df = df.set_index('key')
    df = df.T
    return df


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
    #data = data.sample(frac=1).reset_index(drop=True)

    if args.testratio:
        vr = args.testratio
    else:
        vr = 0.7

    if args.test:
        test_data = csvreader.read(args.test)
        vr = 1
    else:
        test_data = None

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
        indexarray = None

    if args.reports:
        reports_path = args.reports
    else:
        reports_path = 'reports.txt'

    if args.predictionpath:
        predictionpath = args.predictionpath
    else:
        predictionpath = 'predictions'

    if args.resultpath:
        resultpath = args.resultpath
    else:
        resultpath = 'results'

    if args.figurepath:
        figurepath = args.figurepath
    else:
        figurepath = ''

    reports = read_list(reports_path)


    if indexarray:
        indices = data[indexarray].copy()
    else:
        indices = pd.DataFrame(list(range(0, labels.shape[0])), columns=['index'])

    if test_data is not None:
        train_data = data
        test_labels = test_data[label_headers].copy()
        test_features = test_data[feature_headers].copy()
        test_indices = test_data[indexarray].copy()
    else:
        train_data, test_data = np.split(data, [int(vr*len(data))])
        train_features = train_data[feature_headers].copy()
        train_labels = train_data[label_headers].copy()
        test_labels = test_data[label_headers].copy()
        test_features = test_data[feature_headers].copy()
        test_indices = test_data[indexarray].copy()

    if args.kfold > 1:
        shuffled = train_data.sample(frac=1)
        kfolddata = np.array_split(shuffled, args.kfold)
    else:
        kfolddata = [train_data]

    final_model_list = []
    final_score_list = []
    i = 0
    while i < len(models):
        final_model_list.append(0)
        final_score_list.append(-1)
        i += 1

    counter = 0

    for folddata in kfolddata:
        counter = counter + 1

        if args.kfold > 1:
            folddata = folddata.reset_index(drop=True)
            print("Fold " + str(counter) + "/" + str(args.kfold) + " started: ")

        labels = folddata[label_headers].copy()
        features = folddata[feature_headers].copy()

        train_features, tmp_test_features = np.split(features, [int(0.7*len(features))])
        train_labels, tmp_test_labels = np.split(labels, [int(0.7*len(labels))])

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
                        if (sub_model_type == 'Lasso' or sub_model_type == 'PLS' or sub_model_type == 'ElasticNet' or model_type == 'KernelSVM') and type == 'classifier':
                            print('No classification for Lasso, PLS, ElasticNet, KernelSVM')
                        else:
                            sub_model = build_model(sub_model_type, type, configs[sub_model_type], feature_headers, label_headers)
                            if sub_model:
                                sub_models_list.append(sub_model)
                    else:
                        print('No valid configurations for sub model ', model_type)
                models_list.append(sub_models_list)
            else:
                if model_type in configs.keys():
                    if (model_type == 'Lasso' or model_type == 'PLS' or model_type == 'ElasticNet' or model_type == 'KernelSVM') and type == 'classifier':
                        print('No classification for Lasso, PLS, ElasticNet, KernelSVM')
                    else:
                        model = build_model(model_type, type, configs[model_type], feature_headers, label_headers)
                        if model:
                            models_list.append(model)
                else:
                    print('No valid configurations for model ', model_type)

        results = []
        modelindex = 0
        
        kwargs = {'origin':args.origin, 'hitmissr':args.hitmissratio}
        for model in models_list:
            if isinstance(model, list):
                first_train_features = model[0].fit(train_features, train_labels)
                model[1].fit(first_train_features, train_labels)
                model[1].save()
                first_test_features = model[0].fit(tmp_test_features, None)
                predictions = model[1].predict(first_test_features)
                tmp_accuracy = model[1].getAccuracy(tmp_test_labels, predictions, {k: v for k, v in kwargs.items() if v is not None})
                if tmp_accuracy >= final_score_list[modelindex]:
                    final_score_list[modelindex] = tmp_accuracy
                    final_model_list[modelindex] = model
            else:
                dict = {}
                model.fit(train_features, train_labels)
                model.save()
                predictions = model.predict(tmp_test_features)
                tmp_accuracy = model.getAccuracy(tmp_test_labels, predictions, {k: v for k, v in kwargs.items() if v is not None})
                if tmp_accuracy >= final_score_list[modelindex]:
                    final_score_list[modelindex] = tmp_accuracy
                    final_model_list[modelindex] = model
            modelindex = modelindex + 1

    import copy
    report_indices = copy.deepcopy(test_indices)
    modelindex = 0
    for model in final_model_list:
        if isinstance(model, list):
            first_train_features = model[0].fit(train_features, train_labels)
            model[1].fit(first_train_features, train_labels)
            model[1].save()
            first_test_features = model[0].fit(test_features, None)
            predictions = model[1].predict(first_test_features)
            figpath = figurepath + '_' + str(modelindex) + '.png'
            resultdf = produce_report(model[1], reports, test_labels, predictions, feature_headers, label_headers, report_indices, figpath, args.origin, args.hitmissratio)
            resultdf.to_csv(resultpath + '_' + str(modelindex) + '.csv', index=False)
            predictions = pd.DataFrame(data=predictions.flatten())
            test_labels = test_labels.reset_index(drop=True)
            tmp_df = pd.concat([test_indices, predictions, test_labels], axis=1)
            tmp_df = tmp_df.rename(columns={0:'predictions', label_headers[0]:'actual'})
            tmp_df.to_csv(predictionpath + '_' + str(modelindex) + '.csv', index=False)
        else:
            dict = {}
            model.fit(train_features, train_labels)
            model.save()
            predictions = model.predict(test_features)
            figpath = figurepath + '_' + str(modelindex) + '.png'
            resultdf = produce_report(model, reports, test_labels, predictions, feature_headers, label_headers, report_indices, figpath)
            resultdf.to_csv(resultpath + '_' + str(modelindex) + '.csv', index=False)
            predictions = pd.DataFrame(data=predictions.flatten())
            test_labels = test_labels.reset_index(drop=True)
            tmp_df = pd.concat([test_indices, predictions, test_labels], axis=1)
            tmp_df = tmp_df.rename(columns={0:'predictions', label_headers[0]:'actual'})
            tmp_df.to_csv(predictionpath + '_' + str(modelindex) + '.csv', index=False)
        modelindex = modelindex + 1
