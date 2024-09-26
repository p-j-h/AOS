import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def add_noise(column):
    noise = np.random.normal(0, 0.05, len(column))
    return column + noise

def normalize_data(df, columns):
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

# 1. 데이터 로딩 및 전처리
base_path = 'X:\\Dropbox\\003 Tactile\\chem\\data\\total'
wine_types = range(1, 11)
sensor_types = range(1, 5)
sheets = range(1, 6)

data_list = []

custom_columns = ['sec', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

for wine in wine_types:
    for sensor in sensor_types:
        file_name = os.path.join(base_path, f"wine{wine}_sensor{sensor}.xlsx")
        for sheet in sheets:
            df = pd.read_excel(file_name, sheet_name=sheet-1)
            df.columns = custom_columns
            #df = df.drop(df.index[0]).reset_index(drop=True)
            #df['sec'] = pd.to_timedelta(df['sec'].apply(lambda x: str(float(x))) + 's')
            try:
                df['sec'] = df['sec'].apply(lambda x: '{:.6f}'.format(float(x)))
                df['sec'] = pd.to_timedelta(df['sec'] + 's')
            except ValueError as e:
                problem_rows = df[df['sec'].apply(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isnumeric())]
                print(problem_rows)
                print(df['sec'].unique())
                raise e

            #df['sec'] = pd.to_timedelta(df['sec'].astype(float).astype(str)).dt.total_seconds()
            
            df['wine'] = wine
            df['sensor'] = sensor
            df['trial'] = sheet
            data_list.append(df)

# 2. 리샘플링 및 잘라내기
min_duration = min([(df['sec'].max() - df['sec'].min()) for df in data_list])
resampled_data_list = []
for df in data_list:
    df = df.set_index('sec').resample('0.5S').ffill().reset_index()
    df['sec'] = df['sec'] - df['sec'].min()
    df = df[df['sec'] <= min_duration]
    resampled_data_list.append(df)

data_list = resampled_data_list
# 정규화
sensor_columns = custom_columns[1:13] # 속성 이름 리스트에서 센서 이름만 추출
for df in data_list:
    df = normalize_data(df, sensor_columns)


final_df = data_list

# 4. 저장
final_df.to_csv('X:\\Dropbox\\003 Tactile\\chem\\data\\final\\final_data1234.csv', index=False)


# 5. 시각화
selected_data = final_df[(final_df['wine'] == 1) & (final_df['sensor'] == 1)]
plt.plot(range(len(selected_data['1'])), selected_data['1'])
plt.xlabel('Time (sec)')
plt.show()

selected_data = final_df[(final_df['wine'] == 1) & (final_df['sensor'] == 2)]
plt.plot(range(len(selected_data['1'])), selected_data['1'])
plt.xlabel('Time (sec)')
plt.show()

selected_data = final_df[(final_df['wine'] == 1) & (final_df['sensor'] == 3)]
plt.plot(range(len(selected_data['1'])), selected_data['1'])
plt.xlabel('Time (sec)')
plt.show()

plt.show()