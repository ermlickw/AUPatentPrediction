'''
Patent AU Prediction Project 
Billy Ermlick

This script was used to parse the data from the downloaded xml content into CSV files
for the training and testing data.
'''
import time
import sys
import os
import dataset
from bs4 import BeautifulSoup
from datafreeze import freeze
from xml.dom.minidom import parse
import xml.dom.minidom
import re
import dataset
import zipfile


def unzipdownloads(DownloadLocation, IPGlocation):
    for root, dirs, files in os.walk(DownloadLocation): #walk it
        for file in files:
            if file.endswith(".zip"):
                zip_ref = zipfile.ZipFile(os.path.join(os.path.realpath(DownloadLocation),file)) # create zipfile object
                zip_ref.extractall(os.path.realpath(IPGlocation)) # extract file to dir
                zip_ref.close() # close file
                print('unzipped '+ file)
                # os.remove(file) # delete zipped file
                # print('deleted '+ file + 'from download folder')


def getxmlfiles(datafolderholder, xmlfolder):
    '''
    This will take all files in datafolderholder and create individual xml files
    in datasubfolder with patent application number as title
    '''
    data_to_write = ""
    docnumb = "error"
    potentaldoc = ""
    copydocno = True
    datafolder = os.path.realpath(datafolderholder) #find directory where data is stored

    for root, dirs, files in os.walk(datafolder): #walk it
        print(files)
        for file in files:
            data_to_write = ""
            docnumb = "error"
            potentaldoc = ""
            copydocno = True
            i=0
            if file.endswith(".xml"):
                print(file)
                with open(os.path.join(datafolder, file)) as cluster:
                    for line in cluster:
                        data_to_write = data_to_write + '\n' + line
                        #get document number
                        if re.match(r'<doc-number>',line.strip()) and copydocno:

                            s = line.strip()[line.strip().find("<doc-number>")+12:line.strip().find("</doc-number>")].strip()
                            if (len("".join(s.split())) == 8) and (s[:2]==str(12) or
                                                                    s[:2]==str(13) or s[:2]==str(14) or
                                                                    s[:2]==str(15) or s[:2]==str(16) or
                                                                    s[:2]==str(17) or s[:2]==str(11) or
                                                                    s[:2]==str(10) or s[:2]==str('D0')):
                                docnumb = s
                                copydocno = False

                        #write to individual file
                        if re.match(r'</us-patent-grant>',line.strip()):
                            with open(os.path.join(os.path.join(datafolder,xmlfolder), docnumb + '.xml'), 'w') as file:
                                file.write(data_to_write)
                                data_to_write = ""
                                docnumb = "error"
                                copydocno = True
                        i=i+1

def getbigcsvfile(xmlfolder, filename):
    db = dataset.connect()  # create table
    table = db['PATENT_DATA']
    toinsert = dict()
    for root, dirs, files in os.walk(xmlfolder): #walk it
        print(files)
        i=0
        for file in files:
            if file.endswith(".xml") and file[:2] != 'D0':
                try:
                    contents = open(os.path.join(xmlfolder, file),"r").read()
                    soup = BeautifulSoup(contents)
                    toinsert['patent_number'] = ' '.join(soup.find_all('doc-number')[0].getText().split())
                    toinsert['grant_date'] = ' '.join(soup.find_all('date')[0].getText().split())
                    toinsert['filing_date'] = ' '.join(soup.find_all('date')[1].getText().split())
                    toinsert['app_number'] = ' '.join(soup.find_all('doc-number')[1].getText().split())
                    toinsert['art_unit'] = ' '.join(soup.find_all('department')[0].getText().split())
                    toinsert['title'] = ' '.join(soup.find_all('invention-title')[0].getText().split())
                    toinsert['abstract'] = ' '.join(soup.find_all('abstract')[0].getText().split())
                    toinsert['description'] = ' '.join(soup.find_all('description')[0].getText().split())
                    toinsert['claims'] = ' '.join(soup.find_all('claims')[0].getText().split())
                    table.insert(toinsert)
                    i=i+1
                    if i%1000 ==0:
                        print(i)
                    if i%10000 ==0:
                        freeze(table, format='csv', filename=filename) #save table as csv
                except:
                    continue

    freeze(table, format='csv', filename=filename) #save table as csv
    print('Upload Successful.')

if __name__ == '__main__':
    DownloadLocation = r"C:\Users\BillyErmlick\Downloads"
    IPGlocation = 'PatentData'
    XMLlocation = "IndividualFiles"
    start=time.time()
    # unzipdownloads(DownloadLocation, IPGlocation)
    then=time.time()
    print("unzipped in ",round(then-start,2)/60, "minutes")
    # getxmlfiles(IPGlocation, XMLlocation)
    then=time.time()
    print("converted to XML in ",round(then-start,2)/60, "minutes")
    getbigcsvfile(os.path.join(IPGlocation,XMLlocation),"CSVs/GrantData.csv")
    then=time.time()
    print("CSV created in ",round(then-start,2)/60, "minutes")

    import pandas as pd
    df = pd.read_csv('GrantData.csv')
    print(list(df))
    print(df.iloc[1,:])
    print(len(df))
