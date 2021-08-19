# Neural_Network_Charity_Analysis
creating a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

## Project Overview
The purpose of this project is to design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset.

## Resources:
Data: charity_data.csv
Tools: Jupyter Notebook, SciKit Learn, Pandas, Tensor Flow

## Analysis
In this analysis, we used data preprocessing techniques to clean the dataset including binning categorical data and one-hot encoding. The data was then split into training and testing data to build a binary classification model using a neural network/deep learning model. After building an initial model, we attempted to optimize the model to increase its performance on the test data by tuning specific parameters such as removing noisy variables from feature dataset, create binning to reduce feature data noise, adding neurons and layers to the model, and testing additional activation functions within the model to increase performance.


### Data preprocessing
- **Target Variable**
The target variable for our model is the `IS_SUCCESSFUL` column variable because it is a dependent variable in this dataset for we can build a prediction model.

- **Feature Variables**
The independent variables that can contribute to the prediction of the `IS_SUCCESSFUL` column variable. They are included as feature variables because they may have an affect on a binary outcomes. The columns are defined as the following:
    - `APPLICATION_TYPE` — Alphabet Soup application type
    - `AFFILIATION` — Affiliated sector of industry
    - `CLASSIFICATION` — Government organization classification
    - `USE_CASE` — Use case for funding
    - `ORGANIZATION` — Organization type
    - `STATUS` — Active status
    - `INCOME_AMT` — Income classification
    - `SPECIAL_CONSIDERATIONS` — Special consideration for application
    - `ASK_AMT` — Funding amount requested

- **Unused Variables from dataset**
The `EIN` and `NAME` columns were not included in the analysis because they represent independent identifier variables for the dataaset and will not contribute to the prediction modeling.

- **Splitting and scaling**
The dataset was split into training and testing data using the `train_test_split` module. `StandardScaler()` was used to scale the feature data for training and testing. The `random_state` parameter was set to 78 for all modeling tests.

### Compiling, Training, and Evaluating our model
To evalutate our models we used `binary_crossentropy` for loss and `adam` optimizer evaluated by accuracy score based on the test data.

#### Initial Model
We started by basic parameters for a deep learning model to gauge a starting point for optimizing. We used the ReLU activation function for both initial dense layers which is ideal for looking at positive nonlinear input data for classification. We used the sigmoid function for the output layer as it outputs a probability of 0 to 1. We used a high level of neurons so the model will train quuickly within 50 epochs.

**Compiling**
- Model type: Neural Network built using Tensorflow Keras Sequential model
- Layers | Nodes | Activation Function: 
    - Input layer (Dense layer #1) |  80 neurons | ReLU
    - Dense layer #2 |  30 neurons | ReLU
    - Output layer | 1 neuron | Sigmoid

**Picture 1.1: First Deep Learning Model**
**** insert initial keras model ****


**Training**
- Number of epochs: 50

**Evaluation**
- Model Accuracy: 73.1%
- Model loss: 0.552

**Analysis**
This model **FAILED** to reach our goal accuracy threshold of 75%. 

**Next steps**
In the next model, we are going to use the Keras optimizer to give us an idea of what the best model design is for the current feature variables.


#### First model optimization
For our first model improvement we used an Bayesian optimization function to tune our model using twenty models that test number of layers, nodes, and test the use of four different activation functions for our hidden layers (sigmoid, tanh, ReLU, and Leaky ReLU) to find a "better" model than our original. The function then picks the parameters for its highest scoring results based on accuracy. You can see the optimized output in Picture 2.1.

**Picture 2.1: Keras model optimizer function results**
**** insert keras tuner image here ****

We then set these as our model parameters to test against our initial model performance.

**Compiling**
- Model type: Neural Network built using Tensorflow Keras Sequential model
- Layers | Nodes | Activation Function: 
    - Input layer (Dense layer #1) |  5 neurons | Leaky ReLU
    - Dense layer #2 |  3 neurons | Leaky ReLU
    - Dense layer #3 |  3 neurons | Leaky ReLU
    - Dense layer #4 |  9 neurons | Leaky ReLU
    - Dense layer #5 |  9 neurons | Leaky ReLU
    - Output layer | 1 neuron | Sigmoid

**Picture 2.1: Optimized model v1**
**** insert optimized model v1 ****


**Training**
We increased the number of training epochs signifcantly to test if it would help improve the model further.
- Number of epochs: 150

**Evaluation**
- Model Accuracy: 73.0%
- Model loss: 0.551

**Analysis**
This model **FAILED** to reach our goal accuracy threshold of 75%. It also failed to improve on the first model as the accuracy and loss results were almost identical.

**Next steps**
In our next model we attempted a single layer neural network and remove some of the feature variables to test to see if limiting the number of inputs can help improve the model. We removed the following feature variables in addition to the "identifying" variables:
 - `STATUS`: classifier that defines if the loan is active or not, so most likely not needed for prediction
 - `SPECIAL_CONSIDERATIONS`: identify unique cases, but may not have bearing on success
 - `INCOME_AMT`: Showing the value counts, there is a significant number of "0"s implying the data may not be comprehensive and could add unneccessary variance

#### Second model optimization
For this optimization test, we removed some feature variables deemed less important to the prediction model based on their description. We also reverted our activation function back to the basic ReLU function in order to set similar parameters to our initial model. Additionally, we reduced the overall complexity of the model layers as the the first optimization attempt did not improve the model's accuracy.

**Compiling**
- Model type: Neural Network built using Tensorflow Keras Sequential model
- Layers | Nodes | Activation Function: 
    - Input layer (Dense layer #1) |  3 neurons | ReLU
    - Output layer | 1 neuron | Sigmoid

**Picture 3.1: Optimized model v2**
**** insert optimized model v2 ****


**Training**
We reduced the number of training epochs to 50 to mirror our first attempt. After a couple models we also noticed a lack of improvement in the models after 50 or so epochs.
- Number of epochs: 50

**Evaluation**
- Model Accuracy: 72.3%
- Model loss: 0.569

**Analysis**
This model **FAILED** to reach our goal accuracy threshold of 75%. It also failed to improve on the first model as the accuracy and loss results were worse than our first two models. This may imply our model is not complex enough and may need additional layers. Also, it is possible that the reduction in too many feature variables decreased the model's prediction accuracy.

**Next steps**
Our next steps, were to reintroduce the `INCOME_AMT` feature variable to the dataset as it may provide some boost to accuracy even if the data may be not fully correct. We also added an additional dense layer to the model and attempted to clean up some of the feature variable datasets through binning.


#### Third optimization model
For this optimization test, we reducedthe variance in `USE_CASE` feature variable to bucket less frequent values. We also manually created logarithmicesque bins for the `ASK_AMT` feature variable as the numbers we're skewed towards lower value loans and only a few extremely high value loans existed. We also added additional neurons (more than our first model) and an additional layer with a different activation function than the first dense layer (from the previous optimization attempt) because our simple model from our last optimization attempt did not improve performance.

**Compiling**
- Model type: Neural Network built using Tensorflow Keras Sequential model
- Layers | Nodes | Activation Function: 
    - Input layer (Dense layer #1) |  120 neurons | Leaky ReLU
    - Dense layer #2 |  60 neurons | ReLU
    - Output layer | 1 neuron | Sigmoid

**Picture 4.1: Optimized model v3**
**** insert optimized model v3 ****


**Training**
We again ramped up the training epochs to identify a threshold where improvements were starting to wane.
- Number of epochs: 150

**Evaluation**
- Model Accuracy: 72.7%
- Model loss: 0.570

**Analysis**
This model **FAILED** to reach our goal accuracy threshold of 75%. It also failed to improve on the first model as the accuracy and loss results were worse than our first two models. This may imply our the ReLU activation function is not as effective as the Leaky ReLU function as it performed worse as the second layer activation function (which differs from our first optimization attempt). Since we continued to leave out two of our initial feature variables, those may also have a larger effect on the prediction model than we thought initially.

**Next steps**
Reduce number of changes between iterations to understand granular effects of changing parameters and input features.

#### Summary of deep learning optimizations
Our current optimiation efforts failed to *significantly improve* performance from our initial model. There was a slight improvement in one of our models, but not significant. This may be due to the tuning approach of testing mutliple changes within each iteration, therefore failing to specifically identify which changes had the biggest effect on the model's performance. Overall, the Keras Sequential deep learning model was able to perform with an accuracy in the range of 72-74% consistently and has not improved with several changes to the model.

#### Recommendations for continued optimization
- Try to change only one parameter to optimize at a time to more easily identify exactly how each change affects the model accuracy instead of changing multiple metrics at once.
- Attempt different model type like a Random Forest Classifier as they are designed to handle robust datasets and reduce overfitting issues.
- Use a binning tool instead of manual binning which are subject ot human bias
- Use a boosting technique such as oversampling to increase the number of data points in the higher value loan amount range as the dataset is skewed towards smaller loans