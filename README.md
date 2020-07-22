# ML-Platform
A platform integrated with Keras, sklearn, and many data process/result evaluation tools.

This framework consists three parts, data preprocessor, model generator and single prediction tester

## Data Precessor
Currently, we only provides data cleaner, which can replace 'NaN' values in your dataset based on your strategy. 
There are three strategies,

- ignore: remove all the rows containing 'NaN' data
- mean_col
- mean_row
- 

An example to call this tool is 
python3 data_clean.py -train data.csv -s ignore -o new_data.csv

Where, 
data.csv is the file contains your original data
ignore is the strategy
new_data.csv is the new file which will store your cleaned data

## Model Generator
In model generator, it generates a series of models based on user's configurations. Each model can produce a report to show its performance.
In this step, we may need to include some files, which are discussed below.

### -m model.txt (required)
model.txt file defines what models we what to generate. An example is included in this repository called model.txt.
Currently, we support the following models.

| Models  |
| ------------- |
| Ridge  |
| DecisionTree  |
| AdaBoost  |
| GradientBoost  |
| RandomForest  |
| KernelSVM |
| NeuralNetwork  |
| ConvolutionalNeuralNetwork |
| LSTM |
| PCA |
| PLS |
| Lasso |
| Elasticnet |

(Note: PCA is a dimension reduction technique, it needs to combine with the other models. An example to use PCA and Ridge is [PCA, Ridge] )


### -c configs.txt (required)

This file defines the specific configurations of each model. There is a default configs.txt file in this repository, but users are welcome to customise their own models.


### -train trainingdata.csv (required)

The training data to train the ML models. The dataset is expected to have at least two columns (for one input and one output) and the corresponding headers.


### -in input.txt/-out output.txt (required)

These two files specify the headers of the inputs/outputs. The headers need to be included in the trainingdata file. An example of training dataset (data.csv), input.txt and output.txt are included in the repository.


### -r reports.txt (required)

The performance such as plotting(prediction vs actual plot), confusion matrix, R-square score, etc. An example is provided in report.txt.
Currently we have the following reports:

| Report  |
| ------------- |
| Accuracy |
| ConfusionMatrix |
| ROC |
| RSquare |
| MSE |
| MAPE |
| RMSE |
| FeatureImportance |
| Plot |

(Note: some reports are only avaliable under certain inference types, for example, ConfusionMatrix will only be produced in classification.)

### -index index.txt (optional)

Similar to the input.txt/output.txt, this file has the header of a column in the training dataset. However, this file identifies the index column, which is only needed when you are making a plot for regression. The index will become the x-axis labels for the plotting. An example is provided in indexarray.txt.


### -test test_data.txt (optional)

A model needs to be validated to avoid overfitting. The users are welcomed to provide their own validation dataset (this dataset is required to have the same format and columns as the training dataset). If there is no validation dataset, the training dataset will be splited automatically by 7(for training):3(for validating).


### other arguments

-t (required)

The inference type, r for regression and c for classification

-tr (optional)

test ratio, default is 0.7, representing a 7:3 split for train/test.

-hmr (optional)

hitmissratio for calculating accuracy for regression type inference.

-origin (optional)

origin point for calculating accuracy for regression type inference.


### An example to call this tool is
python3 ml_platform.py -train data.csv -in input.txt -out output.txt -t c



## Single Test
Once the models are generated, the users can make predictions use single test

An example to call this tool is
python3 single_test.py -train data.csv -in input.txt -out output.txt -m NeuralNetwork -t c -mp nn_model.h5 -rp prediction.csv

Where,

data.csv is the file contains your original data

input.txt is a list of input feature names

output.txt is a list of output label names

NeuralNetwork is the model type

nn_model.h5 is the model's location

prediction.csv is the output path

c is prediction type, c for classification and r for regression
