

# Ex.No: 13 Learning – Use Supervised Learning(Miniproject)
## DATE:     
09/10/2024
## REGISTER NUMBER : 
212222040073
## AIM: 
To write a program to train the classifier for Diebetes.
##  Algorithm:
Step 1 : Start the program.
Step 2 : Import the necessary packages, such as NumPy and Pandas.
Step 3 : Install and import Gradio for creating the user interface.
Step 4 : Load the diabetes dataset using Pandas.
Step 5 : Split the dataset into input features (`x`) and target labels (`y`).
Step 6 : Split the data into training and testing sets using `train_test_split`.
Step 7 : Standardize the training and testing data using the `StandardScaler`.
Step 8 : Instantiate the `MLPClassifier` model with 1000 iterations and train the model on the training data.
Step 9 : Print the model's accuracy on both the training and testing sets.
Step 10 : Define a function `diabetes` to take input values for diabetes features and predict the outcome using the trained model.
Step 11 : Create an interactive Gradio interface for the diabetes detection system with text inputs for each feature and output as either "YES" or "NO" for the prediction.
Step 12 : Launch the Gradio interface and share it for interaction.
Step 13 : Stop the program.
## Program:
```
#import packages
import numpy as np
import pandas as pd
```
```
pip install gradio
```
```
import gradio as gr
```
```
import pandas as pd
```
```
#get the data
data = pd.read_csv('/content/diabetes.csv')
data.head()
```
```
print(data.columns)
```
```
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
```
```
#split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y)
```
```
#scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
```
```
#instatiate model
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))
```
```
print(data.columns)
```
```
#create a function for gradio
def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"
```
```
outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```
## Output:
![Screenshot 2024-10-12 125413](https://github.com/user-attachments/assets/a07fe78b-43f6-4e03-ad83-7040b191f6de)

![Screenshot 2024-10-12 125552](https://github.com/user-attachments/assets/5eb8b5db-05df-4340-8a1c-a4b91208bb34)


## Result:
Thus the system was trained successfully and the prediction was carried out.
