Diabetes Prediction Project
This project involves building machine learning models to predict whether a person has diabetes based on certain medical features. The dataset used contains information about various health parameters and the target variable is whether the person has diabetes (1) or not (0). Multiple classification models, including Logistic Regression, Random Forest, Support Vector Machine (SVM), and Decision Trees, were trained and evaluated to predict the outcome.

Files
diabetes.csv: The dataset containing medical attributes and diabetes outcomes.
README.md: This file providing project details.
Installation Requirements
To run this project, you need the following libraries installed:

bash
Copiar código
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Dataset
The dataset diabetes.csv contains the following columns:

Pregnancies: Number of pregnancies.
Glucose: Plasma glucose concentration.
BloodPressure: Diastolic blood pressure (mm Hg).
SkinThickness: Triceps skinfold thickness (mm).
Insulin: 2-hour serum insulin (mu U/ml).
BMI: Body mass index.
DiabetesPedigreeFunction: A function which calculates the likelihood of diabetes based on family history.
Age: Age of the patient.
Outcome: 1 if the person has diabetes, 0 otherwise (target variable).
Data Preprocessing
Missing Values: The dataset contains no missing values.
Data Inspection: Data consists of 768 rows and 9 columns.
Feature Analysis: Summary statistics and correlation heatmaps were created to understand the relationships between features.
Data Visualization
Countplot of Outcome: A countplot to show the distribution of diabetes outcomes (0 vs. 1).
Distribution of Pregnancies: A distribution plot and box plot were used to visualize the distribution of the 'Pregnancies' feature.
Pairplot: A pairplot showing relationships between all features.
Heatmap: A heatmap to visualize the correlation matrix between features.
Model Building
The project uses several machine learning models to predict diabetes:

1. Logistic Regression
Training: The data is split into training and testing sets (80% training, 20% testing).
Performance: The model achieved an accuracy of 79.22% on the test set.
2. Support Vector Machine (SVM)
Model: A linear kernel SVM model is used for classification.
3. Random Forest Classifier
Training: Random Forest model with 200 trees is trained and evaluated.
Feature Importance: Feature importances are calculated and plotted to identify the most influential features.
4. Decision Tree Classifier
Model: A decision tree model is used for classification, providing a clear interpretation of the decision-making process.
5. XGBoost
Model: An XGBoost model is also applied, though it is not fully explored in the current implementation.
Model Evaluation
The evaluation is done using metrics such as accuracy, precision, recall, F1-score, confusion matrix, and classification report.

Example Results:
Logistic Regression: Accuracy: 79.22%, with a confusion matrix:
True negatives: 87
False positives: 9
False negatives: 23
True positives: 35
Classification Report: The model performed well with a balanced precision and recall for both classes (0 and 1).

Usage
To use this project:

Ensure all necessary Python libraries are installed.
Load the dataset (diabetes.csv).
Run the script to preprocess the data, build models, and evaluate them.
Modify the test_data to make predictions for new input values.
python
Copiar código
test_data = (8, 183, 64, 0, 0, 23.3, 0.672, 32)
test_data_np = np.array(test_data)
test_data_rs = test_data_np.reshape(1, -1)
prediction = logreg.predict(test_data_rs)
print(prediction)  # Output: 1 (diabetic)
Conclusion
This project demonstrates the use of machine learning models for diabetes prediction. By analyzing features such as glucose levels, age, BMI, and others, we can predict whether a patient has diabetes. Logistic Regression and Random Forest models both achieve an accuracy of 79.22%, which can be further improved by tuning hyperparameters or using more advanced models like XGBoost.



