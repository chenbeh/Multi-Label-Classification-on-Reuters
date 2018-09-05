
# coding: utf-8

# In[2]:


try:
    import nltk
except ModuleNotFoundError:
    get_ipython().system('pip install nltk')


# In[3]:


## This code downloads the required packages.
## You can run `nltk.download('all')` to download everything.

nltk_packages = [
    ("reuters", "corpora/reuters.zip")
]

for pid, fid in nltk_packages:
    try:
        nltk.data.find(fid)
    except LookupError:
        nltk.download(pid)


# In[4]:


from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict


# ## Setting up train/test data

# In[5]:


train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])


# In[6]:


all_categories = sorted(list(set(reuters.categories())))


# ## Feature engineering

# First we will start by extracting the features. In this work, we will use the occurence of the words as features. To do this, we need first to determine the vocabulary that we will be limited to. In fact, we will work with the words int the training as our vocabluary(with a little filtering).  
# 
# So, we must tokenize the documents and then extract the chosen words. We begin by tokenizing the text, transform to lower case all the words. Despite that every word is important in the category classification, we tolerate to delete the stopwords and the ponctuations. Then count the tokens. the training sample will be a vector of occurence of the extracted words in the each document. By thinking more about this idea, I discovred that this method is very similar the bag of words or 1-gram algorithm.
#   
#    Luckily, by exploiting the available package on the net, I was able to find a predefined method in sklearn that implement back of words. Furthermore, this method iplements also the tokenizing, the counting and the stopwords moving. So, I think that it will be very good if we use it instead of starting from scratch 

# In[7]:


#CountVectorizer is bag of words sklearn implementation :
# extract only at least two char words
# According to the documentation, tokenization and counting the freq of each token is done too
# @parameters :
# binary : true if occurence only , false if frequency
# stop_words : english  remove english stopwords
vectorizer = CountVectorizer(binary="True",stop_words="english")

# fit the vectorizer to the train documents(7769)
# Our feature will be the words in the training data only ( we must not add the test data)
X_train = vectorizer.fit_transform(train_documents)

# calculate the occurence of the training words in the test documents(~3020)
X_test = vectorizer.transform(test_documents)
X_test


# In[8]:


#print the vectors extracted from the training data
# column_j : word_j in vocabluary, rows_i: occurence of the extracted training words in document_i
print( vectorizer.fit_transform(train_documents).todense())

# the vocab extracted form the training data
#{word_1 : id_1, word_2 : id_2, ..., word_N: id_N}
print( vectorizer.vocabulary_ )


# ## Category preprocessing
# 
# This is about a supervised problem since for each document we have its category(the output to be predicted). So it is a labeled data.    
# Before proceding to the classifier learning, we must make some preparation in the target category. It is better(mandatory) to work with numeric value when using machine learning algorithms. In this part, we can keep the solution given in the example.  
# At the end, each sample category is represented by a vector of all the categories. A cell is set if it is one of the actual categories.

# In[9]:


# The MultiLabelBinarizer will convert the categories from each article
# into a vector with 0 or 1 depending if the article is from the category or not.
mlb = MultiLabelBinarizer()
mlb.fit(train_categories + test_categories)
print("These are the all categories from the MultiLabelBinarizer:\n{}".format(", ".join(mlb.classes_)))
y_train = mlb.transform(train_categories)
y_test  = mlb.transform(test_categories)


# # Classifier learning
# 
# Now we discuss the choice of the model to train. The decision was inspired from the differents baseline and the references for the document classification. This depend on the category of our problems(regression// classification), the type (supervised//unsupervised) and the size of training data (<100K samples). For our case, it is a classification (we will predict discrete calsses or categories not numbers), supervised ( data is labeled) and the training data don't exceed 100K samples. Concerning this criteria, the stat of the art, in the document classification problems, advice us to use the linear SVM or Naive Bayes as classifier.  

# #### Kfold impelentation
# 
# We divide our trainin data to train set and validation set by using the KFold from sklearn.  
# Kfold can run only if we have at least two classes or categories for every samples, otherwise it will not run because it can not make a split properly when the sample contain only one class. This happen because we are not taking into consideration the case thath the document belong to any of the categories. And belonging to any category is in fact a class. If we take this case into consideration, Kfold will work for any category.  
# **=>** For this reason, I choose to implement Kfold separately.

# In[10]:


from sklearn.model_selection import KFold
# From the scikit-learn library, we are using Bernoulli's Naive Bayes
X = X_train
kf = KFold(n_splits=5)
kf.get_n_splits(X)
# Using Naive Bayes for Multi-Label-Classificaition:
# One classifier for each label:
clfs = OrderedDict()
for train_index, test_index in kf.split(X):
    X_train_fold, X_val = X[train_index], X[test_index]
    y_train_fold, y_val = y_train[train_index], y_train[test_index]

    for i, category in enumerate(all_categories):
        #clf = BernoulliNB()
        clf = svm.SVC(kernel="linear", C=1.0)
    
        # We train each classifier individually, so we must use
        # only 0 or 1 as y_train.
        y_train_clf = [yt[i] for yt in y_train_fold]
        y_val_clf = [yt[i] for yt in y_val]
        unique = np.unique(y_train_clf)
        # .fit() will train the model with the training data
        if len(unique)> 1 :
            clf.fit(X_train_fold, y_train_clf)
            #validate
            #pred = np.zeros((len(y_val), 1)
            pred = clf.predict(X_val)
            print(category)
            print("Accuracy : {:.4f}".format(metrics.accuracy_score(y_val_clf, pred)))
            clfs[category] = clf
    
clfs


# In[11]:


# From the scikit-learn library, we are using Bernoulli's Naive Bayes


# Using Naive Bayes for Multi-Label-Classificaition:
# One classifier for each label:
clfs = OrderedDict()

for i, category in enumerate(all_categories):
    #clf = BernoulliNB()
    clf = svm.SVC(kernel="linear", C=1.0)
    
    # We train each classifier individually, so we must use
    # only 0 or 1 as y_train.
    y_train_clf = [yt[i] for yt in y_train]

    # .fit() will train the model with the training data
    clf.fit(X_train, y_train_clf)
    print(category)
    clfs[category] = clf
    
clfs


# ## Evaluation of the model
#  Now we will evaluate our model on the test set. To do this we start by making the prediction and then calculate the metrics of evaluation, to know the acuuracy,precision, recall and F.

# In[12]:


# do prediction for each article in the test documents
y_pred = np.zeros((len(y_test), len(all_categories)))

for i, (cat, clf) in enumerate(clfs.items()):
    y_pred[:, i] = clf.predict(X_test)


# ## Classifier Evaluation
# For the evaluation, we will calculate the macro and micro metrics.

# #### Macro Evaluation

# In[13]:


print("Accuracy : {:.4f}".format(metrics.accuracy_score(y_test, y_pred)))
print("Precision: {:.4f}".format(metrics.precision_score(y_test, y_pred, average='macro')))
print("Recall   : {:.4f}".format(metrics.recall_score(y_test, y_pred, average='macro')))
print("F1-Score : {:.4f}".format(metrics.f1_score(y_test, y_pred, average='macro')))


# #### Micro Evaluation

# In[14]:


print(metrics.classification_report(y_true=y_test, y_pred=y_pred, target_names=mlb.classes_))


# From this results, we can say that our algorithm can classify correctly ~76% of test inputs. But we can notice that the macro-precision and macro-recall are not so good. In fact, in our model, only 55% of the slected classification are coorect and only 35% of the correct classification are slected.  
# On the other hand, for the micro evaluation, we can say that it is good. And this is due to that the micro precision and accuracy are affected mostly with the classes that usually happens in the dataset.  
# The precision is bigger than the recall and we can deduct that the model is able to classify correctly the categories, but he is not able to find all the categories.
# ## interpretation and discussion
# 
# So it is clear from the test results that our model does not behave good the rare classes. And from this point we will try to find the problem and try to give some solutions to enhance the model.  
# As first intuituion, I get the idea to test the classifier on the train set and below are the results

# In[15]:


# do prediction for each article on the train documents
y_pred = np.zeros((len(y_train), len(all_categories)))

for i, (cat, clf) in enumerate(clfs.items()):
    y_pred[:, i] = clf.predict(X_train)
    
print("Accuracy train : {:.4f}".format(metrics.accuracy_score(y_train, y_pred)))
print("Precision train: {:.4f}".format(metrics.precision_score(y_train, y_pred, average='macro')))
print("Recall train   : {:.4f}".format(metrics.recall_score(y_train, y_pred, average='macro')))
print("F1-Score train : {:.4f}".format(metrics.f1_score(y_train, y_pred, average='macro')))


# ## Problem found and solutions
# According to the results above, we can notice that all the metrics are near to 1. So, from this we can conclude that our model suffer from the problem of overfitting. He is learning by heart the training data and he is not able to classify an unseen samples.
# In order to address this problem we can thing about 3 soluions :
# * **Increase the number of samples on the training data:** A very common solution for every machine learning problem is to get more data. This is the wish for every ML algorithm. But I think that in our case it is hard to get more data because we our limited by the number of documents in the corpus
# * **Decrease the number of features:** A second solution is to decrese the number of features to know the number of words and keep only the infomative features
# * **Regularization of classifier:** The other 2 solutions affecte the data but this one affect the model. In this methode, we tune and chenge the parameter of the classifier and find the best one that fit our data.

# ### Decrease the number of features
# 
# To decrease the number of features, I found 2 solutions. The first one is to specify the parameter **max_features** in **countVectoriezer** method (according to SKELARN documentation). The second one is most tricky, we can implement **PCA** algorithm in order to reduce the dimensiality of the feature space.

# In[16]:


vectorizer = CountVectorizer(stop_words="english",max_features=10000)
# fit the vectorizer to the train documents(7769)
# Our feature will be the words in the training data only ( we must not add the test data)
X_train = vectorizer.fit_transform(train_documents)
# calculate the occurence of the training words in the test documents(~3020)
X_test = vectorizer.transform(test_documents)
X_test


# In[17]:


clfs = OrderedDict()
for i, category in enumerate(all_categories):
    #clf = BernoulliNB()
    clf = svm.SVC(kernel="linear",C=1.0)
    
    # We train each classifier individually, so we must use
    # only 0 or 1 as y_train.
    y_train_clf = [yt[i] for yt in y_train]

    # .fit() will train the model with the training data
    clf.fit(X_train, y_train_clf)
    
    clfs[category] = clf
    
clfs
# do prediction for each article in the test documents
y_pred = np.zeros((len(y_test), len(all_categories)))

for i, (cat, clf) in enumerate(clfs.items()):
    y_pred[:, i] = clf.predict(X_test)
print("Accuracy : {:.4f}".format(metrics.accuracy_score(y_test, y_pred)))
print("Precision: {:.4f}".format(metrics.precision_score(y_test, y_pred, average='macro')))
print("Recall   : {:.4f}".format(metrics.recall_score(y_test, y_pred, average='macro')))
print("F1-Score : {:.4f}".format(metrics.f1_score(y_test, y_pred, average='macro')))

# do prediction for each article in the train documents
y_pred = np.zeros((len(y_train), len(all_categories)))

for i, (cat, clf) in enumerate(clfs.items()):
    y_pred[:, i] = clf.predict(X_train)
    
print("Accuracy train : {:.4f}".format(metrics.accuracy_score(y_train, y_pred)))
print("Precision train: {:.4f}".format(metrics.precision_score(y_train, y_pred, average='macro')))
print("Recall train   : {:.4f}".format(metrics.recall_score(y_train, y_pred, average='macro')))
print("F1-Score train : {:.4f}".format(metrics.f1_score(y_train, y_pred, average='macro')))


# * **PCA ==> Truncated SVD**  
# Yes I know that I told you that I will use PCA for the dimensiality reduction. The thing that I don't know is that PCA doesn't accept sparse input like we have. Hopefully, there is an alternative which is TruncatedSVD.

# In[18]:


from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
svd = TruncatedSVD(n_components=1000, n_iter=7, random_state=42)

vectorizer = CountVectorizer(binary="True",stop_words="english")
# fit the vectorizer to the train documents(7769)
# Our feature will be the words in the training data only ( we must not add the test data)
X_train = vectorizer.fit_transform(train_documents)
# calculate the occurence of the training words in the test documents(~3020)
X_test = vectorizer.transform(test_documents)
X_train = svd.fit_transform(X_train)
X_train
X_test = svd.transform(X_test)
X_train
clfs = OrderedDict()
for i, category in enumerate(all_categories):
    #clf = BernoulliNB()
    clf = svm.SVC(kernel="linear")
    
    # We train each classifier individually, so we must use
    # only 0 or 1 as y_train.
    y_train_clf = [yt[i] for yt in y_train]

    # .fit() will train the model with the training data
    clf.fit(X_train, y_train_clf)
    
    clfs[category] = clf
    
clfs
# do prediction for each article in the test documents
y_pred = np.zeros((len(y_test), len(all_categories)))

for i, (cat, clf) in enumerate(clfs.items()):
    y_pred[:, i] = clf.predict(X_test)
print("Accuracy : {:.4f}".format(metrics.accuracy_score(y_test, y_pred)))
print("Precision: {:.4f}".format(metrics.precision_score(y_test, y_pred, average='macro')))
print("Recall   : {:.4f}".format(metrics.recall_score(y_test, y_pred, average='macro')))
print("F1-Score : {:.4f}".format(metrics.f1_score(y_test, y_pred, average='macro')))


# ##### results 
# Decreasing the numver of features doens not enhance the performance, but contrary, after a certain threshold it become worse for the calissifier.

# ### Regularization 
# As the last step to enhance the model, I tried to change the hyperparameters of the classifer starting from the kenerl, the penalty parameter C and also by tuning the coefficient of gamme. But all the tentatives didn't enhance the performance.

# # Conlcusion
# So this is are the two solution stried in this exemple to avoid the overfitting and to enhance the performance. I have a doubt also that our model is not overfitting, the true problem is that our classifier is not able to learn the behaviour of the categories that contain only fex examples. Some category include only one document. By this, our algoritmh is not able to recognize this latter one and for this rease=on our performance is not good on the macro metrics. In the other side, for the micro metrics it looks very good and this because they are emphasized by the common and frequent categories.  
# Finally, I will say that the last choice and solution that I din't try is to add more labeled data especially the rare categories. By this our model will be able to be genral and to have a good patter about all the categories.

# ***************************
# **************************
# *****************************

# # For testing the model please run this final model before

# In[19]:


vectorizer = CountVectorizer(binary="True",stop_words="english" )
# fit the vectorizer to the train documents(7769)
# Our feature will be the words in the training data only ( we must not add the test data)
X_train = vectorizer.fit_transform(train_documents)
# calculate the occurence of the training words in the test documents(~3020)
X_test = vectorizer.transform(test_documents)

# The MultiLabelBinarizer will convert the categories from each article
# into a vector with 0 or 1 depending if the article is from the category or not.
mlb = MultiLabelBinarizer()
mlb.fit(train_categories + test_categories)
y_train = mlb.transform(train_categories)
y_test  = mlb.transform(test_categories)

clfs = OrderedDict()
for i, category in enumerate(all_categories):
    #clf = BernoulliNB()
    clf = svm.SVC(kernel="linear",C=1.0)
    
    # We train each classifier individually, so we must use
    # only 0 or 1 as y_train.
    y_train_clf = [yt[i] for yt in y_train]

    # .fit() will train the model with the training data
    clf.fit(X_train, y_train_clf)
    
    clfs[category] = clf
    
clfs


# In[ ]:


# Input sentence
example_text = "The vast majority of the predictions end up on the diagonal (predicted label = actual label), where we want them to be. However, there are a number of misclassifications, and it might be interesting to see what those are caused"

# Tokenize
example_tokens = nltk.word_tokenize(example_text)

# Extract features
example_features = [[25 if w in example_tokens else 0 for w in vectorizer.vocabulary_ .keys()]]

# Do prediction
example_preds = [clf.predict(example_features)[0] for clf in clfs.values()]

# Convert predictions back to labels
example_labels = mlb.inverse_transform(np.array([example_preds]))

# Print labels
print("Example text: {}".format(example_text))
print("Example labels: {}".format(example_labels))

