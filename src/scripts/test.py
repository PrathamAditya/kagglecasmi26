import pandas as pd

a = pd.read_csv('data/test/local_rank_refill_0334.csv').set_index('inchikey14')
b = pd.read_csv('data/test/local_rank_refill_0320.csv').set_index('inchikey14')
c = pd.read_csv('data/test/local_rank_refill_0315.csv').set_index('inchikey14')

common = a.index.intersection(b.index).intersection(c.index)
for k in common:
    ranks = [a.loc[k, 'rank'], b.loc[k, 'rank'], c.loc[k, 'rank']]
    if 1 in ranks and not all(r == 1 for r in ranks):
        print(k, '0334:', a.loc[k, 'rank'], a.loc[k, 'top1_smiles'],
                  '| 0320:', b.loc[k, 'rank'], b.loc[k, 'top1_smiles'],
                  '| 0315:', c.loc[k, 'rank'], c.loc[k, 'top1_smiles'])