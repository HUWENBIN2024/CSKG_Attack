# This file is for converting atomic18-formed data to atomic20-formed data
import pandas as pd
import ast

# only need head 
df = pd.read_csv('./sampleTrainRatio0.01.tsv', sep='\t')
df['IsHead'] = df['EventType'].apply(lambda x: x == 'head')
df = df[df.IsHead == True]
df = df.drop(columns=['IsHead'])

# processing
out = pd.DataFrame(columns=['head_event','relation', 'tail_event'])
for i, row in df.iterrows():
    relations = ast.literal_eval(row['Relations'])
    for ral in relations:
        out.loc[len(out.index)] = [row['Event'], ral, 'dummy']

# write into tsv
out.to_csv('generation_data.tsv', sep="\t", index=False)
