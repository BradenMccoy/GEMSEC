import pandas as pd
data = pd.read_csv('bdata.20130222.mhci.txt',delimiter='\t')
data = data[data['species']=='human']
#test_sample.to_csv('blind_test_sample.csv')

def k_fold_cv(data,train_frac,k):
    train_sample = data.sample(frac=train_frac)
    test_sample = data[~data.index.isin(train_sample.index)]
    test_sample.to_csv('test_sample.csv')
    n = int(len(train_sample)/k)
    train_sets={}
    for i in range(k):
        train_sets['set'+str(i+1)]=train_sample.iloc[i*n:(i+1)*n,:]
        train_sets['set'+str(i+1)].to_csv('train_split_set_'+str(i+1)+'.csv')
    return(train_sets,test_sample)

# example of function use
train,test = k_fold_cv(data,0.8,5)
print(train)
print(test)
