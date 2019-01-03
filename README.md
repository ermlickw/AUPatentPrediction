# AUPatentPrediction
## Billy Ermlick
****************************************************************************
This is a repo to analyze data stored in https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/2018/. 
WINDOWS ONLY CURRENTLY

Steps for use:
1) Download project to any directory
2) Clear downloads folder
3) Go to https://bulkdata.uspto.gov/data/patent/grant/redbook/fulltext/2018/ and download as many IPG###### zipped files as needed
4) Edit lines 120 in parser.py to suit your download directory location
5) run parser.py 
6) run factory.py
7) run analyzer.py
8) run app.py 

*****************************************************************************
parser.py - used to extract downloaded zipped files, convert each IPG######.xml file into individual xml files, and create a single CSV file from each of the individual xml files. Results in a single CSV for all utility patents holding patent number, grant date, filing date, app number, art unit, title, abstract, description, and claims. Saves IndividualFiles in PatentData and GrantData in CSVs. 

factory.py - utilized for building art unit feature vector and classifier models based on the created single CSV file. Shows performance on a randomly subsetted test set. Saves TrainTestPrepared Data, TFIDFvectorizers, Classifiers, and CMs.

analyzer.py - used to find top tokens used in each art unit for use in the web app. Saves TopWords.

auxiliary.py - used for storing minor functions

app.py - used for predicting new data. 
