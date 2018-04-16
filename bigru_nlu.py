



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
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report





myfile=r"ner.txt"
data=open(myfile,mode='r',encoding='latin1')
data=data.readlines()

#Data acquisition
labels=[]
tokens=[]
temp1=[]
temp2=[]
count,max_len=0,0
for word in data:
    if word == '\n':
        max_len=max(max_len,count)
        tokens.append(temp1)
        labels.append(temp2)
        temp1=[] #
        temp2=[] #
        count=0
    else:
        count+=1
        temp2.append(word[-2]) #appending labels
        temp1.append(word[0:(len(word)-3)]) # appending words
#print(tokens)
#print(labels)





# finding sentence length for every sentence
sent_len=np.zeros((len(tokens),1))
for i in range(len(tokens)):
    sent_len[i,0]=len(tokens[i])



# padding the sentences to equal length
for i in range(len(tokens)):
    for j in range(len(tokens[i]),max_len):
        tokens[i].append('EOS')
   


# In[49]:


#making the embeddings
W_embed = gensim.models.Word2Vec(tokens,min_count=1,size = 30)
v=list(W_embed.wv.vocab)
word2indices=dict((c,i) for i,c in enumerate(v))


# In[50]:


# maikng an embedding matrix
embedding_matrix = np.zeros((len(word2indices),30),dtype='float32')
for word in word2indices.keys():
    embedding_matrix[word2indices[word] ]= W_embed.wv[word]


# In[51]:


# converting all sentences to their index in vocab
data_x=np.zeros((len(tokens),max_len))
for i in range(len(tokens)):
    for j in range(max_len):
        data_x[i][j]=word2indices[tokens[i][j]]

#one hot vector conversion
unique={'O':0,'D':1,'T':2}
data_y=np.zeros(((data_x.shape[0],data_x.shape[1],3)))
for i in range(len(tokens)):
    for j in range(len(tokens[i])):
        if tokens[i][j] == "EOS":
            data_y[i,j,unique['O']]=1 #EOS are Others Labeled
           # print("j")
        else:
            data_y[i,j,unique[labels[i][j]]]=1



# In[52]:


X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(data_x, data_y, sent_len,test_size=0.2,random_state=4)


# In[53]:


def next_batch(x_data, y_data, data_len, batch_id, batch_size):
    start = batch_id*batch_size
    end = min(start + batch_size, x_data.shape[0])
    X = x_data[start:end,:]
    Y = y_data[start:end,:,:]
    length = data_len[start:end]
    return X,Y,length


# In[74]:


## building Tensorflow Graph
import tensorflow as tf
tf.reset_default_graph()

state_dim=80
epochs=20
batch_size=100
total_batch=X_train.shape[0]/batch_size
keep_prob=0.7


# Placeholders
x = tf.placeholder(dtype = tf.int32, shape = [None,data_x.shape[1]])
y = tf.placeholder(dtype = tf.int32, shape = [None,data_y.shape[1],data_y.shape[2]])
len_ = tf.placeholder(dtype = tf.int32, shape = [None,])
prob = tf.placeholder_with_default(1.0, shape=())


# Weights 
layer2 = {'weights': tf.Variable(tf.random_normal([2*state_dim,3])), 
            'biases': tf.Variable(tf.random_normal([3]))}\



# network layers 
word_embeddings=tf.get_variable("word_embeddings",initializer=embedding_matrix,dtype=tf.float32)
embedded_input=tf.nn.embedding_lookup(word_embeddings,x)
gru_cell=tf.contrib.rnn.GRUCell(num_units=state_dim)
gru_cell=tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell, input_keep_prob=prob)

((fw_out, bw_out), (fw_state, bw_state))=(tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell, cell_bw=gru_cell, sequence_length = len_,inputs=embedded_input,dtype=tf.float32))

output=tf.concat([fw_out,bw_out],axis=2)

output_shape = tf.shape(output)[2]


# aggregating over all batches
cost=0
for i in range(batch_size):
    temp=output[i,0:len_[i],:]
    temp=tf.reshape(temp,[len_[i],output_shape])
    y_sample=y[i,0:len_[i],:]
    y_sample=tf.reshape(y_sample,[len_[i],3])
    out=tf.add(tf.matmul(temp,layer2['weights']),layer2['biases'])
    cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out[0:len_[i],:], labels = y_sample))
    
cost/=batch_size


#
accuracy=0
output_labels=[]
for i in range(batch_size):
    temp=output[i,0:len_[i],:]
    temp=tf.reshape(temp,[len_[i],output_shape])
    y_sample=y[i,0:len_[i],:]
    y_sample=tf.reshape(y_sample,[len_[i],3])
    out=tf.add(tf.matmul(temp,layer2['weights']),layer2['biases'])
    true_classified= tf.equal(tf.argmax(y_sample,axis=1),tf.argmax(out,axis=1))
    output_labels.append(tf.argmax(out,axis=1))
    accuracy += tf.reduce_sum(tf.cast(true_classified,dtype='float'))
    
    
optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()
    
    
    


# In[93]:


with tf.Session() as sess:
    sess.run(init)
    for i in range(25):
        loss=0
        for j in range(int(total_batch)):
            batch_x,batch_y,batch_len= next_batch(X_train, y_train, len_train, j, batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y,len_:batch_len.reshape(-1,),prob:keep_prob})
            loss+=c/total_batch
        print('Loss in epoch %d'%i +'is %lf'%loss)
        
    # for accuracy
    accuracy_test=0
    y_pred=[]
    for i in range(int(X_test.shape[0]/batch_size)):
        batch_x,batch_y,batch_len= next_batch(X_test, y_test, len_test, i, batch_size)
        lab,acc=sess.run([output_labels,accuracy],feed_dict={x:batch_x,y:batch_y,len_:batch_len.reshape(-1,)})
        accuracy_test+=acc
        y_pred+=lab
print("Test Accuracy is %lf"% (accuracy_test/sum(len_test)))        


# In[78]:





# In[94]:


y_pred2=[]
for l in y_pred:
    y_pred2+=list(l)
y_pred2=np.array(y_pred2)

y_test2=[]
for i in range(700):
    for j in range(int(len_test[i])):
        y_test2.append(np.argmax(y_test[i,j,:]))
    

print(classification_report(y_test2, y_pred2,target_names=["O", "D","T"]))

