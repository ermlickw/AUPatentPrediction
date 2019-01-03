from factory import *
import pickle
import pandas as pd
import numpy as np

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[Xtr['response']==grp_ids]
    else:
        D = Xtr
    D = D.where(D<min_tfidf, 0)
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


df = pd.read_csv('CSVs/GrantData.csv', nrows=1)
alldf = pd.DataFrame(np.load('TrainTestPreparedData/train.npy'))
alldf['response'] = pd.DataFrame(np.load('TrainTestPreparedData/train_label.npy'))

titlemodel = pickle.load(open("TFIDFvectorizers/"+"title-test",'rb'))
abstractmodel = pickle.load(open("TFIDFvectorizers/"+"abstract-test",'rb'))
claimsmodel = pickle.load(open("TFIDFvectorizers/"+"claims-test",'rb'))
descriptionmodel = pickle.load(open("TFIDFvectorizers/"+"descript-test",'rb'))

titlematrix = titlemodel.transform(df['title'])
titledf = pd.DataFrame(data=titlematrix.toarray())

abstractmatrix = abstractmodel.transform(df['abstract'])
abstractdf = pd.DataFrame(data=abstractmatrix.toarray())

descriptionmatrix = descriptionmodel.transform(df['description'])
descriptiondf = pd.DataFrame(data=descriptionmatrix.toarray())

claimsmatrix = claimsmodel.transform(df['claims'])
claimsdf = pd.DataFrame(data=claimsmatrix.toarray())

finaldf = pd.concat([titledf,
                abstractdf,
                descriptiondf,
                claimsdf], axis=1)

features = titlemodel.get_feature_names() +\
        abstractmodel.get_feature_names() +\
     descriptionmodel.get_feature_names() +\
     claimsmodel.get_feature_names() + ['response']


# outdf = top_tfidf_feats(np.array(finaldf.values[0]),features)
# print(outdf)
# outer = top_mean_feats(alldf, features, '28', 0.1,20)
# print(outer.iloc[1:,:])

units = set(alldf['response'])

for unit in units:
    words = top_mean_feats(alldf,features,unit,0.1,100)
    words = words.iloc[1:,0]
    pickle.dump(words, open('TopWords/'+unit, 'wb'))

# show the words in input that match these top words ranked by tfidf score
#clicking on one of the shown AU shows the full list of words/phrases
#to incorporate to move app to that TC


#improve classifications by doing level heiarcies
