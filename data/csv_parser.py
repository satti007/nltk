
import numpy as np
import pandas as pd

seperator = '\n__________________________\n'

df = pd.read_csv('hi/BBC_Articles/news.csv', sep='	')
total = df.shape[0]

labels = np.unique(df['label'])

label2file = {}
for label in labels:
    label2file[label] = open(label + '.txt', 'w')
    label2file[label].write(seperator[1:])

for i in range(total):
    text = df['text'][i]
    label = df['label'][i]
    label2file[label].write(str(text))
    label2file[label].write(seperator)

for label in label2file:
    label2file[label].close()
