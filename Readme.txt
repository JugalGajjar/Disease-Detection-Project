THIS PROJECT IS WRITTEN & COMPILED BY JUGAL GAJJAR.

Description:
------------
This is a Machine Learning (ML) project in python created to assist doctors in detecting following diseases:
- Diabetes
- Heart Disease
- Breast Cancer
- Parkinsons Disease
- Chronic Kidney Disease
On the basis of selected disease, a data entry form will be presented in which user can fill the details and predict whether the patient has that particular disease or not. This project can also be scaled up by adding more disease ML/DL models.


Required Libraries:
-------------------
For GUI console:
- tkinter
- numpy
- pickle

For Machine Learning/Deep Learning models:
- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- keras
- sklearn


Steps to Use:
-------------
- To use Projet with Existing diseases
1. Open Terminal/Command Prompt in the directory where GUI_Console.py is stored
2. Run command (in Windows): python GUI_Console.py
   OR
   Run command (in UNIX Systems): python3 GUI_Console.py
3. Select disease from Dropdown Menu
4. Click "Open Detector" button
5. Enter the details in form and Press "Predict" button
(Result will be displayed on a new window)

- To add new disease in Project
1. Store Dataset in "Datasets" folder
2. Create the model and store it in "Models" folder
3. In GUI_Console.py, add Display name in dropdown list
4. Add conditional statement for new disease as other diseases are added
5. Create a function for GUI window for new disease and add a nested function in it to predict the Disease
(Function code snippet can be taken from previously added functions)


Advantages:
-----------
- This project is easily scalable to include more number of diseases.
- Models used in this project have an average accuracy of more than 85% which brings out pretty accurate results.
- Simple and easy to use GUI console for entering data and predicting disease.
