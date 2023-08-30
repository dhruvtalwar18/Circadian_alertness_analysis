import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from numpy import cumsum
from math import pi
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler 
from sklearn.metrics import mean_absolute_error
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler 

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    '''
    Load the data from the file path specified.
    Parameters:

    file_path (str): The path to the data file.
    Returns:

    DataFrame: The loaded data as a pandas DataFrame.

    '''
    raw_data = pd.read_csv(file_path)
    raw_data['dst'] = raw_data['dst'].astype('object')
    raw_data['weekend'] = raw_data['weekend'].astype('object')
    return raw_data


def preprocess_data(raw_data):
    '''
    Normalize the given data by applying one-hot encoding to categorical columns
    and standard scaling to numerical columns.
    Reduce skewness using log and square transformation.

    Parameters:
    data (DataFrame): The data to be normalized.
    dataskew (DataFrame): Dropped all categorical columns
    categorical_attributes (list): A list of categorical column names.
    numerical_attributes (list): A list of numerical column names.

    Returns:
    DataFrame: The normalized and transformed data as a pandas DataFrame.
  
  '''
    categorical_attributes = list(raw_data.select_dtypes(include=['object','bool']).columns)
    numerical_attributes = list(raw_data.select_dtypes(include=['float64', 'int64']).columns)
    data = raw_data
    data = data.drop(['family_id', 'username'], axis = 1)
    # transform non-numeric parameter to one-hot encoding 
    data = pd.get_dummies(data, dummy_na=False)


    data_skew = data.drop(['meal_type_breakfast_High Carb', 'meal_type_breakfast_High Fat', 'meal_type_breakfast_High Fibre', 'meal_type_breakfast_High Protein',
                      'meal_type_breakfast_MCB', 'meal_type_breakfast_OGTT', 'meal_type_breakfast_UK Average',
                     'sex_F', 'sex_M', 'zygosity_DZ', 'zygosity_MZ', 'zygosity_NT', 'sunrise_hr', 'dst_False',	'dst_True'	,'weekend_False'	,'weekend_True'], axis=1)


    logTr1 = ColumnTransformer(transformers=[("lg1", FunctionTransformer(np.log1p), ["M10VALUE_daybefore"])])
    logTr2 = ColumnTransformer(transformers=[("lg2", FunctionTransformer(np.log1p), ["L5VALUE"])])
    logTr3 = ColumnTransformer(transformers=[("lg3", FunctionTransformer(np.log1p), ["bmi"])])
    df_M10 = pd.DataFrame(logTr1.fit_transform(data_skew))
    df_L5 = pd.DataFrame(logTr2.fit_transform(data_skew))
    df_bmi = pd.DataFrame(logTr3.fit_transform(data_skew))

    final_data = data
    final_data_col = final_data.columns.to_list()



    return df_M10,  df_L5 , df_bmi, final_data, final_data_col

def make_transformation(raw_data, df_M10,  df_L5 , df_bmi):
    '''
    Updates the transformed column to the train_data dataframe.

    '''
    train_data = raw_data
    train_data = train_data.drop(['family_id', 'username'], axis = 1)
    train_data['M10VALUE_daybefore'] = df_M10 
    train_data['L5VALUE'] = df_L5
    train_data['bmi'] = df_bmi
    X = train_data.drop("Morning Alertness",axis=1)
    y = train_data['Morning Alertness']
    ct = make_column_transformer(
        (StandardScaler(),['se_pcen','spt_pcen','sleepoffset_hr_pcen','L5VALUE','L5TIME_num','M10VALUE_daybefore','M10TIME_num_daybefore','meal_log_iauc_breakfast','meal_offset_to_breakfast_hr','age','bmi','sunrise_hr',]), #turn all values from 0 to 1
        (OneHotEncoder(handle_unknown="ignore"), ["meal_type_breakfast","sex",'zygosity','dst','weekend'])
    )

    return ct, X, y


def split_data(X, y, ct):

    '''
    Split the data in 80:20 ratio for training testing respectively.

    Parameters:
    X_train (DataFrame): The input features for training.
    y_train (Series): The target variable for training.

    Returns:
    X_train_normal: X_train after standard scalar transformation
    X_test_normal: X_test after standard scalar transformation

    '''

    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)
    X_train_normal = pd.DataFrame(ct.fit_transform(x_train))
    X_test_normal = pd.DataFrame(ct.transform(x_test))

    col = final_data_col[1:]
    col_dict = dict(zip(X_train_normal.columns, col))
    X_train_normal = X_train_normal.rename(columns=col_dict)
    X_test_normal = X_test_normal.rename(columns=col_dict)

    return X_train_normal, y_train,  X_test_normal, y_test


def train_models(X_train_normal, y_train, X_test_normal, y_test, save_path):
    '''
    Train several regression models on the given training data.
    Use a list of trained models to make predictions on new test data.

    Parameters:
    X_test (DataFrame): The input features for prediction.
    trained_models (list): A list of tuples containing (model_type, model_instance).


    Returns:
    list: A list of tuples containing model names and corresponding trained models.
    y_pred (array-like): Predicted labels for the input data.
    Train and Test R2 scores

    '''

 
    models = {
        'SVR':SVR(),
        'XGBRegressor':XGBRegressor(),
        'Ridge':Ridge(),
        'ElasticNet':ElasticNet(),
        'SGDRegressor':SGDRegressor(),
        'BayesianRidge':BayesianRidge(),
        'LinearRegression':LinearRegression(),
        'RandomForestRegressor':RandomForestRegressor()
    }

    model_results = []
    model_names = []
    for name,model in models.items():
        a = model.fit(X_train_normal,y_train)
        predicted = a.predict(X_test_normal)
        score = np.sqrt(mean_squared_error(y_test, predicted))
        model_results.append(score)
        model_names.append(name)
    df_results = pd.DataFrame([model_names,model_results])
    df_results = df_results.transpose()
    df_results = df_results.rename(columns={0:'Model',1:'RMSE'}).sort_values(by='RMSE',ascending=False)

    print(df_results)

    MLR = XGBRegressor()
    MLR.fit(X_train_normal,y_train)
    y_predict = MLR.predict(X_test_normal)
    df = y_test.to_frame()
    df = df.rename(columns={'Morning Alertness': 'y_test'})
    df['y_predict'] = y_predict
    print(df)
    df.to_csv(save_path)

    train_predictions = MLR.predict(X_train_normal)
    print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
    test_predictions = MLR.predict(X_test_normal)
    print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))



file_path = '../data/data.csv'
save_path = '../res/predicted_result_new.csv'
raw_data = load_data(file_path)
df_M10,  df_L5 , df_bmi, final_data, final_data_col = preprocess_data(raw_data)
ct, X, y = make_transformation(raw_data, df_M10,  df_L5 , df_bmi)
X_train_normal, y_train,  X_test_normal, y_test = split_data(X, y, ct)
train_models(X_train_normal, y_train, X_test_normal, y_test, save_path)