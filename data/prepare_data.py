"""Build the three hierarchy levels from a single raw ED activity table.

The raw table is one row per (hospital, day). This module derives:

* **hospital level** -- the raw rows, minus the covariates that are only
  meaningful once aggregated (AQI, waiting time, temperature, mortality);
* **regional level** -- summed counts and averaged environmental covariates
  per (day, RHA);
* **national level** -- summed counts, averaged waiting time and mortality
  per day, with holidays taken from the ``holidays`` package.

See ``docs/DATA.md`` for the expected column schema. 
"""
import pandas as pd
import holidays

def process_data(data):
    # Preparing data
    data['time'] = pd.to_datetime(data['time'])
    data['month'] = data['time'].apply(lambda x: x.month)
    data['weekday'] = data['time'].apply(lambda x: x.weekday())
    
    # New feature: hospital open/closed
    # Assumption: if M1 == 0, hospital ED is closed that day
    data['open'] = (data['M1'] != 0).astype(int)
    
    # ---------------------------- Hospital level ----------------------------
    hospital_level = data.copy()
    hospital_level.drop(columns=['AQI', 'waiting_time', 'tmpc', 'mortality'], inplace=True)
    
    # ------------------------------ RHA level -------------------------------
    # Drop what we consider specific to the hospital level and irrelevant regionally
    rha_level_i=data.copy()
    rha_level_i.drop(columns=['D0', 'locality', 'SUB', 'SUMC', 'SUP', 'SUPCT', 'age_range', 
                              'patient_access', 'health24_access', 'PHC_access',
                              'emergency_doc_access', 'hospital_doc_access', 'mortality'],
                     inplace=True) 
     
    rha_level = rha_level_i.groupby(['time', 'RHA']).agg({
    'waiting_time': 'mean',
    'M1': 'sum',  
    'M2': 'sum',
    'M3': 'sum',
    'M4': 'sum',
    'M5': 'sum',
    'M6': 'sum',
    'M7': 'sum',
    'M8': 'sum',
    'AQI': 'mean',
    'is_holiday': 'sum',
    'tmpc': 'mean',
    'open': 'sum' }).reset_index()

    rha_level['month']=rha_level['time'].apply(lambda x: x.month)
    rha_level['weekday']=rha_level['time'].apply(lambda x: x.weekday())
  
    # ---------------------------- National level ----------------------------
    # Portugal holidays as representative district of RHA LVT holidays
    years = sorted(data['time'].dt.year.unique().tolist())
    pt_holidays = holidays.Portugal(years=years)
    pt_holidays_dates = set(pt_holidays.keys())

    national_level_i = data.copy()
    national_level_i.drop(columns=['D0','RHA', 'locality', 'SUB', 'SUMC', 'SUP', 'SUPCT', 
                                   'age_range', 'patient_access', 'health24_access', 
                                   'PHC_access','emergency_doc_access', 
                                   'hospital_doc_access', 'AQI', 'tmpc'], inplace=True)
    
    national_level = national_level_i.groupby('time').agg({
    'waiting_time': 'mean',
    'M1': 'sum',  
    'M2': 'sum',
    'M3': 'sum',
    'M4': 'sum',
    'M5': 'sum',
    'M6': 'sum',
    'M7': 'sum',
    'M8': 'sum',
    'mortality': 'mean',
    'open': 'sum' }).reset_index()
    
    national_level['month'] = national_level['time'].apply(lambda x: x.month)
    national_level['weekday'] = national_level['time'].apply(lambda x: x.weekday())    
    national_level['is_holiday'] = 0
    for idx in national_level.index:
        current_date = national_level.loc[idx, 'time'].date()
        if current_date in pt_holidays_dates:
            national_level.loc[idx, 'is_holiday'] = 1
  
    
    # Convert specific columns to numeric types
    numeric_cols = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'patient_access', 
                    'health24_access', 'PHC_access', 'emergency_doc_access', 
                    'hospital_doc_access', 'is_holiday', 'weekday', 'AQI', 'month', 
                    'waiting_time']
    hospital_numeric_cols = [col for col in numeric_cols if col in hospital_level.columns]
    rha_numeric_cols = [col for col in numeric_cols if col in rha_level.columns]
    national_numeric_cols = [col for col in numeric_cols if col in national_level.columns]

    hospital_level[hospital_numeric_cols] = hospital_level[hospital_numeric_cols].apply(pd.to_numeric)
    rha_level[rha_numeric_cols] = rha_level[rha_numeric_cols].apply(pd.to_numeric, errors='coerce')
    national_level[national_numeric_cols] = national_level[national_numeric_cols].apply(pd.to_numeric, errors='coerce')

    
    # List of actual categorical columns
    categorical_cols = ['RHA', 'D0', 'locality', 'age_range', 'time']
    hospital_categorical_cols = [col for col in categorical_cols if col in hospital_level.columns]
    rha_categorical_cols = [col for col in categorical_cols if col in rha_level.columns]
    national_categorical_cols = [col for col in categorical_cols if col in national_level.columns]

    # Convert categorical columns to category codes
    for col in hospital_categorical_cols:
        hospital_level[col] = hospital_level[col].astype('category').cat.codes
    for col in rha_categorical_cols:
        rha_level[col] = rha_level[col].astype('category').cat.codes
    for col in national_categorical_cols:
        national_level[col] = national_level[col].astype('category').cat.codes
   
    return hospital_level, rha_level, national_level
    
