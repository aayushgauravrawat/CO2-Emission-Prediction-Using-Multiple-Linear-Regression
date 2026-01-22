import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df=pd.read_csv('FuelConsumptionCo2.csv')
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)
y= df['CO2EMISSIONS']
corr=df.corr()
print(corr['CO2EMISSIONS'].sort_values(ascending=False))
print(df[['FUELCONSUMPTION_CITY',
    'FUELCONSUMPTION_HWY',
    'FUELCONSUMPTION_COMB_MPG']].corr())
df=df.drop(['FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB'],axis=1)
X=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
multiple_regression = LinearRegression()
multiple_regression.fit(X_train, y_train)
y_pred=multiple_regression.predict(X_test)
print("R2 score:",r2_score(y_test,y_pred))
print("Train R2:", r2_score(y_train, multiple_regression.predict(X_train)))
print("Test  R2:", r2_score(y_test,  multiple_regression.predict(X_test)))
print("Actual CO2:", y_test.values[:5])
print("Predicted CO2:", y_pred[:5])
print("Intercept:", multiple_regression.intercept_)
print("Coefficients:", multiple_regression.coef_)
print("Features:", X.columns)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))