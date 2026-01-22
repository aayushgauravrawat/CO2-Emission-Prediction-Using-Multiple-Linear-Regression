# CO2-Emission-Prediction-Using-Multiple-Linear-Regression
Predicting vehicle CO₂ emissions using multiple linear regression with feature selection, multicollinearity handling, and model interpretation.
The goal of this project is to build a multiple linear regression model to predict vehicle CO₂ emissions based on engine characteristics and fuel consumption parameters.

The workflow followed in this project includes:

Data Preprocessing
  Removed categorical and non-numeric variables
  Selected only continuous numerical features for regression

Feature Selection
  Performed correlation analysis with the target variable (CO₂ emissions)
  Identified fuel consumption and engine variables as dominant predictors
  Handled multicollinearity among fuel-related features

Model Training
  Applied multiple linear regression using scikit-learn
  Split data into training (80%) and testing (20%) sets
  
Model Evaluation
  Evaluated using R² score and Mean Absolute Error
  Checked generalization by comparing train and test performance
  Interpreted learned regression coefficients

Based on correlation analysis and redundancy handling, the following features were selected:
Engine Size
Number of Cylinders
City Fuel Consumption

The learned regression equation is:
CO2
=71.97+11.80(EngineSize)+5.91(Cylinders)+8.34(FuelConsumptionCity)

| Metric                    | Value          |
| ------------------------- | -------------- |
| Train R²                  | 0.859          |
| Test R²                   | **0.877**      |
| Mean Absolute Error (MAE) | **16.76 g/km** |



This project demonstrates a complete regression pipeline including preprocessing, feature selection, multicollinearity handling, model training, evaluation, and interpretation.
The resulting model is accurate, stable, and physically meaningful, making it suitable for practical emission prediction tasks.

Technologies Used:
Python
NumPy
Pandas
scikit-learn
Matplotlib

Author
Aayush Gaurav Rawat
BTech Computer Science Engineering
MIT MANIPAL(2029 BATCH)
