import pandas as pd
from sklearn.model_selection import train_test_split

input_data_df = pd.read_csv("data/time_prediction_input.csv", sep='\t', dtype={'masking': str, 'intervention_model': str})

input_data_df['start_year'] = input_data_df['start_date'].str[-4:]
input_data_df['completion_year'] = input_data_df['completion_date'].str[-4:]
# print(input_data_df)
# time_start = '1960'
time_start = '2005'
time_end = '2025'
print("日期范围:")
print("开始日期晚于:", time_start)
print("结束日期早于:", time_end)
input_data_df = input_data_df[input_data_df['completion_year'] < time_end]
input_data_df = input_data_df[input_data_df['start_year'] >= time_start]


# shape of train_df: (77815, 10)
# shape of test_df: (36466, 10)
print("划分方式:")

time = '2019'
train_df = input_data_df[input_data_df['completion_year'] < time]
test_df = input_data_df[input_data_df['start_year'] >= time]
print(f"按时间{time}划分")

# train_df, test_df = train_test_split(input_data_df, test_size=0.2, random_state=0)
# print("随机划分")

train_df.drop(['start_year', 'completion_year'], axis=1, inplace=False).to_csv("data/time_prediction_train.csv", sep='\t', index=False)
test_df.drop(['start_year', 'completion_year'], axis=1, inplace=False).to_csv("data/time_prediction_test.csv", sep='\t', index=False)

print(train_df.shape)
print(test_df.shape)
