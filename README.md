# Bank Churn Prediction

This project aims to predict customer churn in a bank using a machine learning model. Customer churn occurs when customers stop using a company's services. Understanding and predicting churn helps banks to take preemptive actions to retain customers.

## Dataset

The dataset used for this project is `Customer-Churn-Records.csv`. It contains various features related to the customer's demographics and banking activities.

## Project Steps

1. **Data Loading and Exploration**: Load and explore the dataset.
2. **Data Cleaning**: Handle missing values and remove duplicates.
3. **Feature Engineering**: Convert categorical variables to numerical using one-hot encoding.
4. **Data Scaling**: Normalize the data using MinMaxScaler.
5. **Model Building**: Build a neural network using TensorFlow Keras.
6. **Model Training and Evaluation**: Train the model and evaluate its performance.
7. **Visualization**: Plot the training and validation accuracy and loss.

## Code Walkthrough

### Data Loading and Exploration

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_frame = pd.read_csv("/content/Customer-Churn-Records.csv")

data_frame.head()
data_frame.shape
data_frame.isnull().sum()
data_frame.duplicated().sum()
