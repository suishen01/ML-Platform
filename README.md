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
python3 data_cleaner.py -train data.csv -s ignore -o new_data.csv

Where, 
data.csv is the file contains your original data
ignore is the strategy
new_data.csv is the new file which will store your cleaned data

## Model Generator
In model generator, it generates a series of models based on user's configurations. Each model can produce a report to show its performance
Currently, we only supports five models, the models and their corresponding configurations are shown below

| Models  | Configurations |
| ------------- | ------------- |
| Ridge  | alpha  |
| AdaBoost  | n_estimator  |
| GradientBoost  | n_estimator  |
| RandomForest  | n_estimator  |
| NeuralNetwork  | epochs, batch_size  |
| PCA+Ridge  | n_componenets  |

Additional to the models and configurations, the users are also requried to specify the training dataset, features, labels, and 
the report contents. 
Currently, the report can produce accuracy rates and confusion matrices.

An example to call this tool is
python3 ml_platform.py -train data.csv -in input.txt -out output.txt -t c

Where,
data.csv is the file contains your original data
input.txt is a list of input feature names
output.txt is a list of output label names
c is prediction type, c for classification and r for regression

The results and reports will generate automatically in results.txt and reports.txt
The models will be saved automatically.
More advanced settings can be found in ml_platform.py

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
