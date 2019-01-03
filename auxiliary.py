import matplotlib.pyplot as plt
import re
import string
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
import itertools


def tokenize(txt):
    """
    Tokenizer and stemmer for all of the data fields
    """
    otherstops = ['claim','claims', 'embodi', 'exampl','unit','plural', 'ref', 'refer','disclosur','includ','provid','form','crossrefer',
     'particularli','use','benefit','second', 'devic', 'incorpor', 'relat','background','present','prioriti','field','patent',
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

    txt = re.sub(r'\d+', '', txt) #remove numbers
    txt = "".join(c for c in txt if c not in string.punctuation) #remove punctuation
    stop = text.ENGLISH_STOP_WORDS.union(string.punctuation).union(otherstops).union(stopwords.words('english'))
    tokens = [i for i in word_tokenize(txt.lower()) if i not in stop]
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(item).strip() for item in tokens if (stemmer.stem(item) not in stop and len(stemmer.stem(item))>1)]
    # print(stemmed)
    return stemmed




def print_cm(cm, classes,
              normalize=False,
              title='Confusion matrix',
              cmap=plt.cm.Blues):
    """
    This function prints and plots a confusion matrix.
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
