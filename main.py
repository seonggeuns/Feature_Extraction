import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from GPEE import GPFE
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('/Users/choeseong-geun/Maps_lab/liver_data/indian_liver_patient_clean.csv')
df=pd.get_dummies(df)
df = df[[col for col in df.columns if col != 'Decision'] + ['Decision']]
unit = {'Age' : None,
        'Total_Bilirubin' : 'mg/dL',
        'Direct_Bilirubin' : 'mg/dL',
        'Alkaline_Phosphotase' : 'IU/L',
        'Alamine_Aminotransferase' : 'IU/L',
        'Aspartate_Aminotransferase' : 'IU/L',
        'Total_Protiens' : 'g/dL',
        'Albumin' : 'g/dL',
        'Albumin_and_Globulin_Ratio' : None,
        'Gender_Female' : None,
        'Gender_Male' : None
        }


ML_model = {'ML': 'classification', 'model': DecisionTreeClassifier(random_state=0, max_depth=3),'classification_evaluate': accuracy_score,'regression_evaluate':None}

GP_config = {'population_size': 10, 'chromosome_size': 15, 'max_generation': 500}

GPFE(data=df,
     test_split_portion=0.3,
     ML_model=ML_model,
     GP_config=GP_config,
     unit=unit,
     kfold=5 # None이면 training accuracy 구함
     )



"""
1 for liver disease; 2 for no liver disease
unit = {'Age' : None,
        'Total_Bilirubin' : 'mg/dL',
        'Direct_Bilirubin' : 'mg/dL',
        'Alkaline_Phosphotase' : 'IU/L',
        'Alamine_Aminotransferase' : 'IU/L',
        'Aspartate_Aminotransferase' : 'IU/L',
        'Total_Protiens' : 'g/dL',
        'Albumin' : 'g/dL',
        'Albumin_and_Globulin_Ratio' : None,
        'Gender_Female' : None,
        'Gender_Male' : None
        }
        
"""


"""
unit = {'culmen_length_mm' : 'mm',
        'culmen_depth_mm' : 'mm',
        'flipper_length_mm' : 'mm',
        'body_mass_g' : 'g',
        'island_Biscoe' : None,
        'island_Dream' : None,
        'island_Torgersen' : None,
        'sex_male' : None
        }

"""

"""
unit = {'age' : None,
        'anaemia' : None,
        'creatinine_phosphokinase' : 'mcg/L',
        'diabetes' : None,
        'ejection_fraction' : 'percentage',
        'high_blood_pressure' : None,
        'platelets' : 'kiloplatelets/mL',
        'serum_creatinine' : 'mg/dL',
        'serum_sodium' : 'mEq/L',
        'sex' : None,
        'smoking' : None,
        'time' : None
        }
"""