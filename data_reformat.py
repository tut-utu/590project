import json
import scipy.io
import ast
import pandas as pd
# ast.literal_eval(a)

dataset_path = 'data/flickr8k/dataset.json'
print('BasicDataProvider: reading %s' % (dataset_path, )) 
dataset = json.load(open(dataset_path, 'r'))

f = open('flickr8k_dataset.txt', 'w')
f.write("filename\timage_id\tcaption_id\tcaptions\tsplit\n")
a = []
for i in dataset['images']:
    for n,j in enumerate(i['sentids']):

        f.write(i['filename']+ "\t" + str(i['imgid']) + "\t" + str(j) + "\t" +
         str(i['sentences'][n]['tokens'])+"\t"+i['split']+ "\n")
f.close()

df = pd.read_csv('flickr8k_dataset.txt', delimiter='\t')
df[df['split']=='train'].to_csv('flickr8k_training_dataset.txt',index=False,sep='\t')