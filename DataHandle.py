import pandas as pd
import numpy as np
np.random.seed(0)

data = pd.read_csv("/Users/dhr/PycharmProjects/studygpt/content/air/atis_intents.csv",header=None)
data.head()
print(data)

data.columns = ['intent','text']
data['intent'].unique()
print(data['intent'].unique())

data['intent'].nunique()
print(data['intent'].nunique())

data['intent'] = data['intent'].str.replace('#','_')
print(data['intent'].unique())

data['intent'] = data['intent'].str.replace('atis_','')
data['intent'].unique()
print(data['intent'].unique())

print(data['intent'].value_counts())

labels = ['flight','ground_service','airfare','abbreviation','flight_time']

data = data[data["intent"].isin(labels)]
print(data['intent'].value_counts())

sample_data = data.groupby('intent').apply(lambda x: x.sample(n=40)).reset_index(drop = True)
print(sample_data)
sample_data.intent.value_counts()
print(sample_data.value_counts())
sample_data.to_csv("sample_data.csv",index=False)
sample_data = sample_data[['text','intent']]
sample_data.head()
sample_data['text'] = sample_data['text'].str.strip()
sample_data['intent'] = sample_data['intent'].str.strip()
sample_data['text'] = sample_data['text'] + "\n\nIntent:\n\n"
# sample_data['text'] = "Classify text into on the intent: flight, ground_service, airline, aircraft, flight_time. Text: "+sample_data['text'] + "\n\nIntent:\n\n"
sample_data['intent'] = " "+sample_data['intent'] + " END"
sample_data.head()

print(sample_data['text'][0])

print(sample_data['intent'][0])

sample_data.columns = ['prompt','completion']

sample_data.to_json("intent_sample.jsonl", orient='records', lines=True)