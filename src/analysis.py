import glob 
import pandas as pd
import random
import os
from tqdm import tqdm

all_data = ''
for d in ['baby', 'clothing', 'elec', 'sport']:
    log_files = glob.glob(f'./log/*{d}*.log')
    data_test = {}
    data_val = {}
    for log in tqdm(log_files, total=len(log_files), desc="Processing"):
        if d not in log:
            continue
        
        with open(log, 'r') as f:
            data = f.read().strip().split('\n')
        
        for i, l in enumerate(data):
            if 'INFO test result: ' in l:
                """
                recall@5: 0.0338    recall@10: 0.0536    recall@20: 0.0843    recall@50: 0.1459    ndcg@5: 0.0226    ndcg@10: 0.0291    ndcg@20: 0.0369    ndcg@50: 0.0492    precision@5: 0.0071    precision@10: 0.0056    precision@20: 0.0045    precision@50: 0.0031    map@5: 0.0187    map@10: 0.0213    map@20: 0.0234    map@50: 0.0253    ,
                recall@5: 0.0329    recall@10: 0.0544    recall@20: 0.0859    recall@50: 0.1485    ndcg@5: 0.0218    ndcg@10: 0.0289    ndcg@20: 0.0370    ndcg@50: 0.0496    precision@5: 0.0073    precision@10: 0.0061    precision@20: 0.0048    precision@50: 0.0033    map@5: 0.0176    map@10: 0.0204    map@20: 0.0226    map@50: 0.0246    
                """
                l = data[i+1]
                l = l.replace('recall', '').replace('ndcg', '').replace('precision', '').replace('map', '').replace(',', '').split('@')[1:]
                try:
                    l = [float(i.split(':')[-1]) for i in l]
                except:
                    continue
                if len(l) == 16:
                    if data_test.get(log) == None or (data_test.get(log) != None and sum(data_test[log]) < sum(l)):
                        data_test[log] = l
            
        if data_test.get(log) == None:
            print('NULL', log)
            os.system(f'rm {log}')
            continue
        
    index = []
    for i in ['recall', 'ndcg', 'precision', 'map']:
        for k in [5, 10, 20, 50]:
            index.append(f'{i}@{k}')
            
    data_test = pd.DataFrame(data_test, index=index).T
    data_test = data_test[['recall@5', 'precision@5', 'map@5', 'ndcg@5']]
    data_test["sum"] =data_test.apply(lambda x:x.sum(), axis =1)
    data_test = data_test.sort_values('sum', ascending=False)
    data_test.drop(columns=['sum'], inplace=True)
    
    data_test.to_csv('analysis.txt', sep='\t')
    with open('analysis.txt', 'r') as f:
        all_data += f.read() + '\n\n'

with open('analysis.txt', 'w') as f:
    f.write(all_data)
