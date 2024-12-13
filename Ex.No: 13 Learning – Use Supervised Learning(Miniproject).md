# Ex.No: 13 Learning – Use Supervised Learning  
### DATE:
4.11.2024
### REGISTER NUMBER : 212222040073
### AIM: 
To write a program to train the Big mart sales analysis and prediction
###  Algorithm:
1.Start the program.

2.Import required Python libraries, including NumPy, Pandas, Google Colab, Gradio, and various scikit-learn modules.

3.Mount Google Drive using Google Colab's 'drive.mount()' method to access the data file located in Google Drive.

4.Install the Gradio library using 'pip install gradio'.

5.Load the diabetes dataset from a CSV file ('big market.csv') using Pandas.

6.Separate the target variable ('Outcome') from the input features and Scale the input features using the StandardScaler from scikit-learn.

7.Create a multi-layer perceptron (MLP) classifier model using scikit-learn's 'MLPClassifier'.

8.Train the model using the training data (x_train and y_train).

9.Define a function named 'Big Market Analysis' that takes input parameters for various features and Use the trained machine learning model to predict the outcome based on the input features.

10.Create a Gradio interface using 'gr.Interface' and Specify the function to be used to make predictions based on user inputs.

11.Launch the Gradio web application, enabling sharing, to allow users to input their data and get predictions regarding diabetes risk.

12.Stop the program.
### Program:
import kagglehub
elahehkazemian_big_mart_sales_prediction_path = kagglehub.dataset_download('elahehkazemian/big-mart-sales-prediction')

print('Data source import complete.')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Loading the Data
df = pd.read_csv('/content/Train.csv')
df_ = pd.read_csv('//content/Test.csv')
df.head()
# data processing
df.isnull().sum()
df= df.drop(['Outlet_Size', 'Item_Weight'], axis=1, errors='ignore')
df.head(2)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
df['Item_Identifier'] = label_encoder.fit_transform(df['Item_Identifier'])
label_encoder = LabelEncoder()
df['Item_Fat_Content'] = label_encoder.fit_transform(df['Item_Fat_Content'])
label_encoder = LabelEncoder()
df['Item_Type'] = label_encoder.fit_transform(df['Item_Type'])
label_encoder = LabelEncoder()
df['Outlet_Identifier'] = label_encoder.fit_transform(df['Outlet_Identifier'])
label_encoder = LabelEncoder()
df['Outlet_Location_Type'] = label_encoder.fit_transform(df['Outlet_Location_Type'])
label_encoder = LabelEncoder()
df['Outlet_Type'] = label_encoder.fit_transform(df['Outlet_Type'])
# Exploratory Data Analysis
df.head()
plt.figure(figsize=(10, 6))
sns.histplot(df['Item_Fat_Content'], bins=30, kde=True)
plt.title('Distribution of Item_Fat_Content')
plt.xlabel('Item_Fat_Content')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=df)
plt.title('Item Outlet Sales by Outlet Type')
plt.xlabel('Outlet Type')
plt.ylabel('Item Outlet Sales')
plt.xticks(rotation=45)
plt.show()
# Correlation Analysis
numeric_df = df.select_dtypes(include=[np.number])


plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
df.head(1)
# Building a Sales Predictor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
X= df[['Item_Identifier', 'Item_Fat_Content', 'Item_MRP', 'Outlet_Establishment_Year']]
y = df['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Display predictions
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions.head())

### Output:

![Screenshot 2024-11-12 082953](https://github.com/user-attachments/assets/17accb49-1fa1-4f8b-990d-54423992fb6a)
![Screenshot 2024-11-12 082944](https://github.com/user-attachments/assets/3609a5ae-b539-45f4-9993-b714703e441b)
![Screenshot 2024-11-12 082937](https://github.com/user-attachments/assets/c7192f82-3182-4424-8fc5-862e5048186e)
![Screenshot 2024-11-12 082929](https://github.com/user-attachments/assets/74ed4afe-013d-4b1d-bfb8-4544ee97a56e)
![Screenshot 2024-11-12 082923](https://github.com/user-attachments/assets/51195a8a-cbc5-4e63-9832-64931bca9fa8)
![Screenshot 2024-11-12 082916](https://github.com/user-attachments/assets/e17daafc-a48e-4077-a702-6b48f5e933e2)
![Screenshot 2024-11-12 082910](https://github.com/user-attachments/assets/43cabf34-de6a-41bb-a5ea-edcd193c8438)
![Screenshot 2024-11-12 082724](https://github.com/user-attachments/assets/f807e519-92ac-4966-8d43-323a97392b80)
![Screenshot 2024-11-12 083036](https://github.com/user-attachments/assets/e31637e5-a018-4395-9dd8-aa9c2d95b3a8)
![Screenshot 2024-11-12 083031](https://github.com/user-attachments/assets/ad313211-9102-44ba-86b0-b36102a091e6)
![Screenshot 2024-11-12 083025](https://github.com/user-attachments/assets/1742c32e-bc1d-4f08-9ce4-f17e7485ccc6)
![Screenshot 2024-11-12 083020](https://github.com/user-attachments/assets/e2bab541-4b8b-46e1-8df9-37ab3466f88c)
![Screenshot 2024-11-12 083012](https://github.com/user-attachments/assets/c342f7db-b970-4d2a-a120-4ec689f4c3f8)
![Screenshot 2024-11-12 083005](https://github.com/user-attachments/assets/152e0e81-808e-4f2e-adf0-74d56f42f831)



### Result:
Thus the system was trained successfully and the prediction was carried out.
