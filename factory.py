'''
Patent AU Prediction Project | 12/28/2018
Billy Ermlick
********************************************************************************

********************************************************************************
'''
#import libraries
import os
import sys
import time
import nltk
import sys
import seaborn as sns
import pandas as pd
import re
import numpy as np
import operator
from collections import defaultdict
import matplotlib.pyplot as plt

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import text
from numpy import loadtxt
import itertools
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
import string

def tokenize(txt):
    """
    Tokenizer that uses porter stemmer to stemm all words
    :param text:
    :return:
    """
    txt = re.sub(r'\d+', '', txt) #remove numbers
    txt = "".join(c for c in txt if c not in string.punctuation) #remove punctuation
    otherstops = ['claim','claims', 'ref', 'refer','disclosur','includ','provid','form','crossrefer', 'particularli',
    'use','benefit','second', 'devic', 'incorpor', 'relat','background','present','prioriti','field','patent',
    'applic','crossref','invent', 'art', 'disclos',
    'file','technic', 'config', 'configur', 'machin', 'addit', 'response', 'signatur', 'realt','comprising', 'comprise', 'comprises',
    'responses', 'method', 'remov','make', 'specif','product','apparatus','afterward', 'alon', 'alreadi',
    'alway', 'anoth', 'anyon', 'anyth', 'anywher', 'apparatu', 'becam', 'becom', 'besid', 'cri', 'describ', 'els',
     'elsewher', 'empti', 'everi','everyon', 'everyth', 'everywher', 'fifti', 'formerli', 'forti', 'henc', 'hereaft',
      'herebi', 'howev', 'hundr', 'inde', 'oper','latterli', 'mani', 'meanwhil', 'moreov', 'mostli', 'nobodi', 'noon', 'noth',
       'nowher', 'otherwis', 'perhap', 'pleas','respons', 'seriou', 'sever', 'sinc', 'sincer', 'sixti', 'someon',
       'someth', 'sometim', 'somewher', 'thenc', 'compon', 'thereaft', 'therebi', 'therefor', 'thu', 'togeth', 'twelv', 'twenti',
        'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev','anywh', 'el', 'elsewh', 'everywh',
         'ind', 'otherwi', 'plea', 'print', 'block', 'figure', 'success', 'embodiments','implementat','respon', 'somewh','related']
    stop = text.ENGLISH_STOP_WORDS.union(string.punctuation).union(otherstops).union(stopwords.words('english'))
    tokens = [i for i in word_tokenize(txt.lower()) if i not in stop]
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(item).strip() for item in tokens if (stemmer.stem(item) not in stop and len(stemmer.stem(item))>1)]
    # print(stemmed)
    return stemmed

def makenewprediction(df):
    print(df)
    #convert to lowercase
    df=df.copy()
    df.iloc[6:10] = df.iloc[6:10].apply(lambda x: x.lower())
    #keep only first 50 words
    df['description'] =  ' '.join(df['description'].split()[:50])
    df['title'] = ' '.join(df['title'].split()[:50])
    df['abstract'] = ' '.join(df['abstract'].split()[:50])
    df['claims'] = ' '.join(df['claims'].split()[:50])

    #load vectorizers
    titlemodel = pickle.load(open("TFIDFvectorizers/title-test",'rb'))
    abstractmodel = pickle.load(open("TFIDFvectorizers/abstract-test",'rb'))
    claimsmodel = pickle.load(open("TFIDFvectorizers/claims-test",'rb'))
    descriptionmodel = pickle.load(open("TFIDFvectorizers/descript-test",'rb'))
    # PCAmodel = pickle.load(open("PCAmodel",'rb'))


    titlematrix = titlemodel.transform([df['title']])
    titledf = pd.DataFrame(data=titlematrix.toarray())

    abstractmatrix = abstractmodel.transform([df['abstract']])
    abstractdf = pd.DataFrame(data=abstractmatrix.toarray())

    descriptionmatrix = descriptionmodel.transform([df['description']])
    descriptiondf = pd.DataFrame(data=descriptionmatrix.toarray())

    claimsmatrix = claimsmodel.transform([df['claims']])
    claimsdf = pd.DataFrame(data=claimsmatrix.toarray())

    finaldf = pd.concat([titledf,
                    abstractdf,
                    descriptiondf,
                    claimsdf], axis=1)

    #if reduction wanted
    # PCAmodel = pickle.load(open("Classifiers/"+"PCAmodel",'rb'))
    # finaldf = pd.DataFrame(PCAmodel.transform(finaldf))
    #make prediction
    clf = pickle.load(open("Classifiers/"+"LogisticRegression",'rb'))
    prediction = clf.predict(finaldf)
    print(prediction)

def print_cm(cm, classes,
              normalize=False,
              title='Confusion matrix',
              cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    fig1=plt.figure(figsize=(30,30))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    fig1.savefig("CMs/"+title,dpi=100)

def preprocess_dataframe(df, numbtrainrows):
    '''
    This function cleans the dataset and represent each document by a feature vector
    The data is first cleaned, then tokenized into a TFIDF representation
    The TFIDF representations are then reduced using LSA or PCA
    The results are then saved to the directory
    '''

    #convert to lowercase
    df.iloc[:,6:9] = df.iloc[:,6:9].apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    #keep only first 50 words
    df['description'] = df['description'].apply(lambda x: ' '.join(x.split()[:50]))
    df['title'] = df['title'].apply(lambda x: ' '.join(x.split()[:50]))
    df['abstract'] = df['abstract'].apply(lambda x: ' '.join(x.split()[:50]))
    df['claims'] = df['claims'].apply(lambda x: ' '.join(x.split()[:50]))
    df.dropna(how='any')

    response_vector = df['art_unit']

    #create tfidf matrix
    def create_tfidfmatrix(inputcolumn, docs, name):
        #prep model
        n_grams = 3
        feature_model = TfidfVectorizer(
            ngram_range=(1, n_grams),
            lowercase=True,
            strip_accents='ascii',
            decode_error='replace',
            tokenizer=tokenize,
            norm='l2',
            min_df=1,
            max_features=2000
        )

        feature_matrix_transform =feature_model.fit_transform(inputcolumn.tolist())
        pickle.dump(feature_model, open('TFIDFvectorizers/'+name+'-test', 'wb'))
        feature_df = pd.DataFrame(feature_matrix_transform.toarray(), index=docs.tolist(), columns=feature_model.get_feature_names())
        return feature_df

    #assign matrix for each predictor
    title_tfidf_df = create_tfidfmatrix(df['title'], df['app_number'], 'title')
    abstract_tfidf_df = create_tfidfmatrix(df['abstract'], df['app_number'], 'abstract')
    description_tfidf_df = create_tfidfmatrix(df['description'], df['app_number'], 'descript')
    claims_tfidf_df = create_tfidfmatrix(df['claims'], df['app_number'], 'claims')

    #combine tfidfs created for each column
    df_feature_vector = pd.concat([title_tfidf_df,
                    abstract_tfidf_df,
                    description_tfidf_df,
                    claims_tfidf_df], axis=1)
    #memory drop
    title_tfidf_df = None
    abstract_tfidf_df = None
    description_tfidf_df = None
    claims_tfidf_df = None

    #feature reduction:

    #SVD instead -latent semantic analysis
    # SVDtrunc = TruncatedSVD(n_components=100)
    # df_feature_vector = SVDtrunc.fit_transform(df_feature_vector)
    # pickle.dump(SVDtrunc, open('SVDmodel', 'wb'))

    #PCA on feature_matrix
    # pca = PCA(n_components=100)
    # df_feature_vector = pca.fit_transform(df_feature_vector)
    # pickle.dump(pca, open('Classifiers/PCAmodel', 'wb'))

    #assign to train and test vectors and labels
    df_feature_vector = pd.DataFrame(df_feature_vector)
    train_feature_vector = df_feature_vector.iloc[:numbtrainrows,:]
    test_feature_vector = df_feature_vector.iloc[numbtrainrows:,:]
    df_feature_vector =None
    train_response_vector = response_vector.iloc[:numbtrainrows]
    test_response_vector = response_vector.iloc[numbtrainrows:]
    response_vector = None

    #save the processed dataset
    np.save('TrainTestPreparedData/train.npy',train_feature_vector)
    np.save('TrainTestPreparedData/train_label.npy',train_response_vector)
    np.save('TrainTestPreparedData/test.npy',test_feature_vector)
    np.save('TrainTestPreparedData/test_label.npy',test_response_vector)

    return   train_feature_vector, train_response_vector, test_feature_vector, test_response_vector

def train_model(classifier, params,
                feature_vector_train, train_y,
                feature_vector_valid, valid_y,
                model, labels,
                load_models):
    '''
    This function trains or loads the models. For training, a 5-fold gridsearch is used
    to find the optimal hyperparameters and the resutls are outputted and saved under
    classifiers
    '''

    #if not loading models then train and save them
    if load_models == False:
        cross_val = KFold(n_splits=5, shuffle=True)
        clf = GridSearchCV(classifier, params, cv=cross_val.split(feature_vector_train), n_jobs=1)
        clf.fit(feature_vector_train, train_y)
        print('Grid Search Completed', clf.best_estimator_, clf.best_score_)

        #save model:
        pickle.dump(clf.best_estimator_, open("Classifiers/"+model, 'wb'))
    else: #load model
        clf = pickle.load(open("Classifiers/"+model,'rb'))


    predictions = clf.predict(feature_vector_valid)
    #create performance metrics
    acc = metrics.accuracy_score(valid_y, predictions)
    prec = metrics.precision_score(valid_y, predictions, average='macro')
    recall = metrics.recall_score(valid_y, predictions, average='macro')
    cr = metrics.classification_report(valid_y,predictions)
    cm = metrics.confusion_matrix(valid_y,predictions, labels=labels)
    f1 = metrics.f1_score(valid_y,predictions, average='macro')

    #print out performance metrics
    print (model,"|   Accuracy:", acc, "|  Macro-averaged Precision:", prec, "| Macro-Averaged F1 Score: ", f1)
    print("Classification Report: \n",cr)
    # print("Confusion_Matrix: \n",cm)
    print_cm(cm, labels,True,model)
    return acc,prec, recall, cr, cm, f1


def main(load_data,load_models):
    '''
    This is the main function.
    Data is read in from the directory and subsampled based on the Experiment.
    Feature vectors are either created or loaded.
    Classifiers are either trained or loaded.
    Predicted results on the test set are provided.
	'''
    #build feature vectors if missing or specified by user above
    if load_data==False or not(os.path.isfile('TrainTestPreparedData/train.npy') or
                                os.path.isfile('TrainTestPreparedData/train_label.npy') or
                                os.path.isfile('TrainTestPreparedData/test.npy') or
                                os.path.isfile('TrainTestPreparedData/test_label.npy')):
        #open files
        df = pd.read_csv('CSVs/GrantData.csv', nrows=70000)
        print(len(df))

        # select TCs/AUs of interest
        df['art_unit'] = df['art_unit'].apply(lambda x: (str(x)[:2]).strip())
        labels = list(set(df['art_unit'].apply(lambda x: (x[:2]).strip())))
        # df= df[df['art_unit']==['17', '21', '24','26','28''36'],:]

        train_feature_vector, train_response_vector, test_feature_vector, test_response_vector = preprocess_dataframe(df,round(len(df)/10)*9)
        then=time.time()
        print("Feature and Response vectors CREATED in ",round(then-now,2), "seconds")

    else: #load the feature vectors if it is in memory already
        train_feature_vector = np.load('TrainTestPreparedData/train.npy')
        train_response_vector = np.load('TrainTestPreparedData/train_label.npy')
        test_feature_vector = np.load('TrainTestPreparedData/test.npy')
        test_response_vector = np.load('TrainTestPreparedData/test_label.npy')
        then=time.time()
        labels = list(set(test_response_vector))
        print("Feature and Response Vectors LOADED in",round(then-now,2), "seconds")

    #set classifiers and hyperparameters to be searched
    classifiers = {
            # 'Baseline': [DummyClassifier(strategy="stratified"), {}],
            'LogisticRegression': [LogisticRegression(solver='lbfgs', multi_class='multinomial'), {}],
            # 'KNN': [KNeighborsClassifier(),{ 'n_neighbors': np.arange(1,4,10)}],
            # 'LDA': [LinearDiscriminantAnalysis(solver='svd'), {}],
            # 'Bayes': [GaussianNB(), {}], #
            # 'SGD': [SGDClassifier(n_iter=8, penalty='elasticnet'), {'alpha':  10**-6*np.arange(1, 15, 2),'l1_ratio': np.arange(0.1, 0.3, 0.05)}],
            # 'Passive Aggressive': [PassiveAggressiveClassifier(loss='hinge'), {}],
            # 'Perceptron': [Perceptron(), {'alpha': np.arange(0.00001, 0.001, 0.00001)}], #
        }

    #train model for all classifiers and output results
    macroprecision = {}
    for model in classifiers.keys():
        acc, prec,recall, cr, cm, f1 = train_model(classifiers[model][0], classifiers[model][1],
                                                            train_feature_vector, train_response_vector,
                                                            test_feature_vector, test_response_vector,
                                                            model,labels,
                                                            load_models)
        then=time.time()
        macroprecision.update({model:[round(f1,2), round(prec,2),round(recall,2), round(acc,2)]})
        print(model, "finished in ", round(then-now,2)/60, "minutes")

    #print final results
    print('\n FINAL RESULTS | MACRO-F1  | MACRO-PRECISION | MACRO-RECALL | ACC')
    d_view = [ (v,k) for k,v in macroprecision.items() ]
    d_view.sort(reverse=True)
    for v,k in d_view:
        print(k,v)

    #make predictions
    makenewprediction(df.iloc[2,:])

if __name__ == '__main__':
    now=time.time()
    #preprocess data and create feature vectors OR load created data
    load_data = False
    load_models = False
    main(load_data, load_models)
    then=time.time()
    print("script finished in ",round(then-now,2)/60, "minutes")
