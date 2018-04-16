
# coding: utf-8

import nltk
import numpy as np
import codecs
import eli5
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
import scipy
from sklearn.metrics import make_scorer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[133]:

# Data reading
myfile=r"ner.txt"
data=open(myfile,mode='r',encoding='latin1')

data=data.readlines()

labels=[]
tokens=[]
temp1=[]
temp2=[]
for word in data:
    if word == '\n':
        tokens.append(temp1)
        labels.append(temp2)
        temp1=[] #
        temp2=[] #
    else:
        
        temp2.append(word[-2]) #appending labels
        temp1.append(word[0:(len(word)-3)]) # appending words
#print(tokens)
#print(labels)

W_embed = gensim.models.Word2Vec(tokens,min_count=1,size = 30)





#kmeans clustering
from sklearn.cluster import KMeans
tokens_train, tokens_test, labels_train, labels_test = train_test_split(tokens, labels, test_size=0.2,random_state=4)

# applying Kmeans on train data 

data_x=[]
data_y=[]
for i in range(0,len(tokens_train)):
    for j in range(0,len(tokens_train[i])):
        data_x.append(W_embed.wv[tokens_train[i][j]])
        data_y.append(labels_train[i][j])
#print(len(data_x))
data_x,data_y = np.array(data_x),np.array(data_y)
#print(data_x)
#print(data_y)


kmeans = KMeans(n_clusters=3, random_state=0).fit(data_x.reshape(-1,30))
#print(data_y[-6])
#kmeans.predict(data_x[-6].reshape(-1,30))[0]







docs=[]
for i in range(len(tokens)):
    temp1=[]
    for j in range(len(tokens[i])):
        temp1.append((tokens[i][j],labels[i][j]))
    docs.append(temp1)
#print(docs[0])


# In[118]:


# POS Tagging of the docs
data=[]
for doc in docs:
    tokens = [t for t,label in doc ]
    tagged=nltk.pos_tag(tokens)
    data.append([(w,pos,label) for (w,label),(word,pos) in zip(doc,tagged)])
    



# features from word net 

def no_of_contexts(word):
    temp=0
    for syn in wn.synsets(word):
        temp+=1
    return temp


def contain_digit(str):
    for ch in list(str):
        if ch.isdigit()==True:
            return True
    return False




def word2features(sent, i): #taking window size of 3 words
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'suffix': word[-5:],
        'prefix': word[0:5],
        'cluster': kmeans.predict((W_embed.wv[word]).reshape(-1,30))[0] ,
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': contain_digit(word),
        'postag': postag,
        
        #'word.upper()':word.upper(),
        'no_of_contexts':no_of_contexts(word),
        'word_lemma':lemma.lemmatize(word,'v'),
        'alpha':word.isalpha(),
        'word_len':len(word),
        
        
    }
    
    if i > 0: # for words which are not the start word of sentence
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:cluster': kmeans.predict((W_embed.wv[word1]).reshape(-1,30))[0],
            #'-1:word.upper()': word1.upper(),
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': contain_digit(word1),
            '-1:alpha':word1.isalpha(),
            '-1:postag': postag1,
            
            '-1:no_of_contexts':no_of_contexts(word1),
            
            
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:cluster': kmeans.predict((W_embed.wv[word1]).reshape(-1,30))[0],
            '+1:word.upper()': word1.upper(),
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()':contain_digit(word1),
            '+1:postag': postag1,
            
            '+1:no_of_contexts':no_of_contexts(word1),
            
        
        })
    else:
        features['EOS'] = True
    
    
    return features





#Function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

#Function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=4)



def print_score(y_test,y_pred):
    from sklearn.metrics import classification_report

    
    labels = {"O": 0, "D": 1,"T":2}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    # Printing classification report
    print(classification_report(truths, predictions,target_names=["O", "D","T"]))

	# Function to calculate accuracy
def accuracy(y_test,y_pred):
    sum,total=0,0
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            total+=1
            if y_test[i][j]==y_pred[i][j]:
                sum+=1
    return ((sum*100)/total)





# hyper parameter tunning(uncomment for parameter tuning)

#labels=["O","D","T"]

#crf = sklearn_crfsuite.CRF(algorithm='lbfgs', 
                          # max_iterations=1000,
                           #all_possible_transitions=True,
                           #verbose=False)

#crf.fit(X_train,y_train)


#dictionary for parameters
#params_space = {
 #   'c1': scipy.stats.expon(scale=0.5),
  #  'c2': scipy.stats.expon(scale=0.5),
#}



# use the f1 score metric for evaluation
#f1_scorer = make_scorer(metrics.flat_f1_score,
#                        average='weighted', labels=labels)

# search
#rs = RandomizedSearchCV(crf, params_space,
 #                       cv=10,
  #                      verbose=1,
   #                     n_jobs=-1,
    #                    n_iter=20,
     #                   scoring=f1_scorer)
#rs.fit(X_train, y_train)


#print('Best params:', rs.best_params_)
#print('Best F-1 score:', rs.best_score_)



# Performance on test set 

crf = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.02,c2=0.3 ,max_iterations=1000, all_possible_transitions=True,verbose=False)
crf.fit(X_train,y_train)
labels=["O","D","T"]
y_pred=crf.predict(X_test)

print("Classification Report on Test Data :")
print_score(y_test,y_pred)
print("F1 score (unweighted average) is %lf "% (metrics.flat_f1_score(y_test, y_pred,
                      average='macro', labels=labels)))


					  
# Writing labels on output file
file = open(r"output.txt", "w")


for i in range(len(labels_test)):
    for j in range(len(labels_test[i])):
        file.write(tokens_test[i][j]+" "+labels_test[i][j]+"\n")
    file.write("\n")

        
file.close()

