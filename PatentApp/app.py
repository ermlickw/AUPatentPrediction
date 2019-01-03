from flask import Flask, request, render_template
from scipy.sparse import hstack
from flask_basicauth import BasicAuth
from os import environ, path
import re, string, pickle
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os
import sys

topfolder = r"C:\Users\BillyErmlick\Desktop\Workspace\FirmWork\ArtUnitPredictions"
sys.path.insert(0, topfolder)
from factory import tokenize
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'
app = Flask(__name__)
app.config.from_object(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = 'admin'

basic_auth = BasicAuth(app)

@app.route('/')
@basic_auth.required
def home():
    return render_template('query.html')


@app.route('/query', methods=['POST', 'GET'])
@basic_auth.required
def submit_query():
    title = None
    abstract = None
    description = None
    claims = None

    if request.method == 'POST':
        try:
            title = request.form['title']
        except KeyError:
            return render_template('query.html', error=KeyError)

        try:
            abstract = request.form['abstract']
        except KeyError:
            return render_template('query.html', error=KeyError)

        try:
            description = request.form['description']
        except KeyError:
            return render_template('query.html', error=KeyError)

        try:
            claims = request.form['claims']
        except KeyError:
            return render_template('query.html', error=KeyError)

        #preprocessing
        df=dict()
        df['title'] = title
        df['abstract'] = abstract
        df['description'] = description
        df['claims'] = claims
        #first 50 words each
        df['description'] =  ' '.join(df['description'].split()[:50])
        df['title'] = ' '.join(df['title'].split()[:50])
        df['abstract'] = ' '.join(df['abstract'].split()[:50])
        df['claims'] = ' '.join(df['claims'].split()[:50])

        #load vectorizers
        titlemodel = pickle.load(open(topfolder + "/TFIDFvectorizers/"+"title-test",'rb'))
        abstractmodel = pickle.load(open(topfolder + "/TFIDFvectorizers/"+"abstract-test",'rb'))
        claimsmodel = pickle.load(open(topfolder + "/TFIDFvectorizers/"+"claims-test",'rb'))
        descriptionmodel = pickle.load(open(topfolder + "/TFIDFvectorizers/"+"descript-test",'rb'))

        #fit models to submission
        titlematrix = titlemodel.transform([df['title']])
        titledf = pd.DataFrame(data=titlematrix.toarray())

        abstractmatrix = abstractmodel.transform([df['abstract']])
        abstractdf = pd.DataFrame(data=abstractmatrix.toarray())

        descriptionmatrix = descriptionmodel.transform([df['description']])
        descriptiondf = pd.DataFrame(data=descriptionmatrix.toarray())

        claimsmatrix = claimsmodel.transform([df['claims']])
        claimsdf = pd.DataFrame(data=claimsmatrix.toarray())

        finaldf = pd.concat([titledf, abstractdf,
                        descriptiondf, claimsdf], axis=1)

        #if reduction wanted
        # PCAmodel = pickle.load(open(topfolder + "/Classifiers/"+"PCAmodel",'rb'))
        # finaldf = pd.DataFrame(PCAmodel.transform(finaldf))

        #load model
        clf = pickle.load(open(topfolder + "/Classifiers/"+"LogisticRegression",'rb'))


        #get matched key tokens
        enteredtokens = []
        if df['title']:
            enteredtokens= enteredtokens + list(titlemodel.fit([df['title']]).vocabulary_.keys())
        if df['abstract']:
            enteredtokens= enteredtokens + list(abstractmodel.fit([df['abstract']]).vocabulary_.keys())
        if df['description']:
            enteredtokens= enteredtokens + list(descriptionmodel.fit([df['description']]).vocabulary_.keys())
        if df['claims']:
            enteredtokens= enteredtokens + list(claimsmodel.fit([df['claims']]).vocabulary_.keys())
        enteredtokens=set(enteredtokens)

        #get words in submission that match top words in each TC/AU
        words=dict()
        for root, dirs, files in os.walk(topfolder+"/TopWords/"):
            for file in files:
                data = list(pickle.load(open(topfolder+'/TopWords/'+str(file),'rb')))
                words[file] = ', '.join(set(data)&set(enteredtokens))
                print(words[file])

        #make predictions and record results
        group = clf.predict(finaldf)
        probs = clf.predict_proba(finaldf)
        groups = clf.classes_
        results = dict()
        for i, clas in enumerate(groups):
            try:
                results[clas] = [probs[0][i],words[clas]]
            except:
                results[clas] = [probs[0][i], []]

        return render_template('results.html', group=group, results=sorted(results.items()),
                               title=title, abstract=abstract, description=description, claims=claims)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(environ.get("PORT", 8000)))
