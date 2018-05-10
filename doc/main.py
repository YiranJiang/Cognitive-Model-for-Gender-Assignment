
# coding: utf-8

# In[3]:


import numpy as np
import numpy.random as npr
import nltk
import time
import scipy as sp
import scipy.sparse as sps
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import inspect
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import compress
import logging,gensim,os
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem.snowball import SnowballStemmer

setwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/'
prewd = os.path.abspath(os.path.join(setwd, os.pardir))

stemmer = SnowballStemmer("french")
models = gensim.models.Word2Vec.load(prewd+'/data/stemmed_frwiki.bin')


# In[4]:


class suffix:
    msuffix = '-an,-and, -ant, -ent, -in, -int, -om, -ond, -ont,-eau, -au, -aud, -aut, -o, -os, -ot,-ai, -ais, -ait, -es, -et,-ou, -out, -out, -oux,-i, -il, -it, -is,-y,-at, -as, -ois,-oit,-u,-us,-ut,-eu,-er,-age, -ege, –ème, -ome,-òme, -aume, -isme,-as, -is, -os, -us, -ex,-it, -est,-al, -el, -il, -ol, -eul, -all,-if, -ef,-ac, -ic, -oc, -uc,-am, -um, -en,-air, -er, -erf, -ert, -ar, -arc, -ars, -art, -our, -ours, -or, -ord, -ors, -ort, -ir, -oir,-eur,-ail, -eil, -euil, -ueil,-ing'
    msuffix = msuffix.split(',')

    fsuffix = 'aie, -oue, -eue, -ion, -te, – ée, -ie, -ue, -asse, -ace, -esse, -ece, -aisse, -isse,-ice, -ousse, -ance, -anse, -ence, -once,-enne, -onne, -une, -ine, -aine, -eine, -erne,-ande, -ende, -onde, -ade, -ude, -arde, -orde,-euse, -ouse, -ase, -aise, -ese, -oise, -ise, -yse, -ose, -use,-ache, -iche, -eche, -oche, -uche, -ouche, -anche,-ave, -eve, -ive,-iere, -ure, -eure,-ette, -ete, –ête, -atte, -otte, -oute, -orte, -ante, -ente, -inte, -onte,-alle, -elle, -ille, -olle,-aille, -eille, -ouille,-appe, -ampe, -ombe,-igue'
    fsuffix = fsuffix.split(',')

    ms = []
    for i in range(0,len(msuffix)):
        t = str(msuffix[i])
        t = t.replace(' ','')
        t = t.replace('-','')
        t = t.replace("–",'')
        ms.append(t)

    fs = []
    for i in range(0,len(fsuffix)):
        t = str(fsuffix[i])
        t = t.replace(' ','')
        t = t.replace('-','')
        t = t.replace("–",'')
        fs.append(t)
        


# In[5]:


class data:
    def __init__(self, data):
        self.data = pd.read_csv(data, sep="	")

    def cleaning(self):
        self.df = pd.DataFrame(self.data)
        self.df = self.df[['X1_ortho','X5_genre']]
        self.frenchword = self.df['X1_ortho'].tolist()
        self.f_m = self.df['X5_genre'].tolist()
        
        self.remove = [i for i in range(len(self.frenchword)) if 
                       type(self.frenchword[i]) == float]
        
        for i in range(len(self.remove)):
            print("removing numer:",i+1,str(self.frenchword[self.remove[i]-i]))
            del self.f_m[self.remove[i]-i]
            del self.frenchword[self.remove[i]-i]
                               

        self.y = []
        for i in range(len(self.f_m)):
            if self.f_m[i] == 'f':
                self.y.append(-1)
            else:
                self.y.append(1)
                
        print("Done!")
        return((self.frenchword,self.y))

    
    def deep_cleaning(self):
        self.df = pd.DataFrame(self.data)
        self.df = self.df[['X1_ortho','X5_genre']]
        self.frenchword = self.df['X1_ortho'].tolist()
        self.f_m = self.df['X5_genre'].tolist()
        self.remove = [i for i in range(len(self.frenchword)) if 
                       type(self.frenchword[i]) == float]
        
        for i in range(len(self.remove)):
            print("removing numer:",i+1,str(self.frenchword[self.remove[i]-i]))
            del self.f_m[self.remove[i]-i]
            del self.frenchword[self.remove[i]-i]
                
        self.fword=[]
        self.mword=[]

        for i in range(len(self.f_m)):
            if self.f_m[i] == 'm':
                self.mword.append(self.frenchword[i])
            if self.f_m[i] == 'f':
                self.fword.append(self.frenchword[i])
                
        self.mword = [stemmer.stem(word) for word in self.mword]
        self.fword = [stemmer.stem(word) for word in self.fword]
        self.fword = list(set(self.fword))
        self.mword = list(set(self.mword))
        
        self.allword = self.mword + self.fword
        
        self.emptysetind = []
        self.train = []
        
        self.j =0

        for i in range(len(self.allword)):
            if self.j ==0:
                try:
                    self.j=1
                    self.train = models.wv[self.allword[i]]
                except: 
                    self.j=0
                    self.emptysetind.append(i)
            else:
                try:
                    self.train = np.concatenate((self.train, models.wv[self.allword[i]]), axis=0)
                except: 
                    self.emptysetind.append(i)        
        
        self.train = np.matrix(self.train)
        
        self.train = self.train.reshape(len(self.allword)-len(self.emptysetind),100)
        
        self.y = np.concatenate((np.repeat(1,len(self.mword)),np.repeat(-1,len(self.fword))),axis = 0)
        self.y = list(self.y)
        
        for i in range(len(self.emptysetind)):
            del self.y[self.emptysetind[i]-i]
            
        return((self.train,self.y))


# In[6]:


class Maxent:
    def __init__(self,address,k,feature_type):
        self.matrix = data(address)
        self.data = self.matrix.cleaning()
        self.y = self.data[1]
        self.k = k
        self.feature_type = feature_type
        self.func = {1:self.fm_features_1,2:self.fm_features_2,3:self.fm_features_3}[self.feature_type]
        
    def fm_features_1(self,name):
        features = {}
        features['alwayson'] = True
        features["last_1_letter"] = name[-1].lower()
        features["last_2_letters"] = name[-2:].lower()
        return features

    def fm_features_2(self,name):
        features = {}
        features['alwayson'] = True
        features["last_1_letter"] = self[-1].lower()
        features["last_2_letters"] = self[-2:].lower()
        features["last_3_letters"] = self[-3:].lower()

        return features

    def fm_features_3(self,name):
        features = {}
        features['alwayson'] = True
        for fsuffix in suffix.fs:
            features['has(%s)' % fsuffix] = self.endswith(fsuffix)
        for msuffix in suffix.ms:
            features['has(%s)' % msuffix] = self.endswith(msuffix)
        return features


    def CV(self):
        # Codes borrowed from the tutorial class.
        np.random.seed(123)
        self.n = len(self.y)
        self.y = np.array(self.y)
        self.test = np.zeros(self.n, dtype=np.bool)
        self.n_p1 = np.sum(self.y == 1)
        self.test[npr.choice(np.nonzero(self.y)[0], int(self.n*1/10), replace=False)] = True
        self.train = np.logical_not(self.test)

        self.n_train = np.sum(self.train)
        self.n_test = np.sum(self.test)
        self.n_train_p = np.sum(self.y[self.train] == 1)
        self.n_train_n = np.sum(self.y[self.train] == -1)
        self.n_test_p = np.sum(self.y[self.test] == 1)
        self.n_test_n = np.sum(self.y[self.test] == -1)
        self.train_idx = np.nonzero(self.train)[0]
        npr.shuffle(self.train_idx)
        self.cv_grid = np.linspace(0, self.n_train, self.k + 1).astype(np.int32)
        

        start_time = time.time()
        self.train_cv = np.copy(self.train)
        self.test_cv = np.copy(self.train)
        self.tmp_err = 0.0

        for j in range(self.k):
            self.train_cv[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]] = False
            self.test_cv[self.train_idx[0:self.cv_grid[j]]] = False
            self.test_cv[self.train_idx[self.cv_grid[j+1]:]] = False
            self.this_train = list(compress(self.data[0],self.train_cv))
            self.this_test = list(compress(self.data[0],self.test_cv))
            self.train_y = self.y[self.train_cv]
            self.test_y = self.y[self.test_cv]
            self.train_ffeats = []
            self.train_mfeats = []
            self.test_ffeats = []
            self.test_mfeats = []
            for i in range(len(self.train_y)):
                if self.train_y[i] == -1:
                    self.train_ffeats.append(
                        (self.func(self.this_train[i]),'feminine'))
                if self.train_y[i] == 1:
                    self.train_mfeats.append(
                        (self.func(self.this_train[i]),'masculine'))
            self.classifier = nltk.MaxentClassifier.train(self.train_ffeats+self.train_mfeats,max_iter= 20) 
            print(self.classifier.show_most_informative_features(100))

            for i in range(len(self.test_y)):
                if self.test_y[i] == -1:
                    self.test_ffeats.append(
                        (self.func(self.this_test[i]),'feminine'))
                if self.test_y[i] == 1:
                    self.test_mfeats.append(
                        (self.func(self.this_test[i]),'masculine'))
            self.testfeats = self.test_ffeats + self.test_mfeats
            self.tmp_err += np.sum(nltk.classify.accuracy(self.classifier, self.testfeats))
            self.train_cv[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]] = True
            self.test_cv[self.train_idx[0:self.cv_grid[j]]] = True
            self.test_cv[self.train_idx[self.cv_grid[j+1]:]] = True
        self.cv_maxent = self.tmp_err / self.k
        print("--- %s seconds ---" % (time.time() - start_time))

        self.re_train_ffeats = []
        self.re_train_mfeats = []
        self.re_test_ffeats = []
        self.re_test_mfeats = []
        self.re_train = list(compress(self.data[0],self.train))
        self.re_test = list(compress(self.data[0],self.test))
        self.re_train_y = self.y[self.train]
        self.re_test_y = self.y[self.test]
        for i in range(len(self.re_train_y)):
            if self.re_train_y[i] == -1:
                self.re_train_ffeats.append(
                (self.func(self.re_train[i]),'feminine'))
            if self.re_train_y[i] == 1:
                self.re_train_mfeats.append(
                (self.func(self.re_train[i]),'masculine'))

        for i in range(len(self.re_test_y)):
            if self.re_test_y[i] == -1:
                self.re_test_ffeats.append(
                (self.func(self.re_test[i]),'feminine'))
            if self.re_test_y[i] == 1:
                self.re_test_mfeats.append(
                (self.func(self.re_test[i]),'masculine'))
        self.re_testfeats = self.re_test_ffeats + self.re_test_mfeats
        self.classifier = nltk.MaxentClassifier.train(self.re_train_ffeats+self.re_train_mfeats,max_iter= 20) 
        print(nltk.classify.accuracy(self.classifier, self.re_testfeats))
        print(self.classifier.show_most_informative_features(100))
        return(self.cv_maxent)

        


# In[12]:


class SVM_Linear:
    def __init__(self,address,k,C_arr_len,C_arr_start,C_arr_end):
        self.matrix = data(address)
        self.data = self.matrix.deep_cleaning()
        print(self.data[0].shape)
        self.X = self.data[0]
        self.y = self.data[1]
        self.k = k
        self.C_arr_len = C_arr_len
        self.C_arr_start = C_arr_start
        self.C_arr_end = C_arr_end
    
    def CV(self):
        # Codes borrowed from the tutorial class.
        np.random.seed(123)
        self.n = len(self.y)
        self.y = np.array(self.y)
        self.test = np.zeros(self.n, dtype=np.bool)
        self.n_p1 = np.sum(self.y == 1)
        self.test[npr.choice(np.nonzero(self.y)[0], int(self.n*1/10), replace=False)] = True
        self.train = np.logical_not(self.test)

        self.n_train = np.sum(self.train)
        self.n_test = np.sum(self.test)
        self.n_train_p = np.sum(self.y[self.train] == 1)
        self.n_train_n = np.sum(self.y[self.train] == -1)
        self.n_test_p = np.sum(self.y[self.test] == 1)
        self.n_test_n = np.sum(self.y[self.test] == -1)
        self.train_idx = np.nonzero(self.train)[0]
        npr.shuffle(self.train_idx)
        self.cv_grid = np.linspace(0, self.n_train, self.k + 1).astype(np.int32)
        

        start_time = time.time()
        self.train_cv = np.copy(self.train)
        self.test_cv = np.copy(self.train)
        self.tmp_err = 0.0

        
        start_time = time.time()
        self.C_arr = np.exp(np.linspace(self.C_arr_start,self.C_arr_end, self.C_arr_len))
        self.cv_err_linsvm = np.empty(self.C_arr_len)
        self.train_cv = np.copy(self.train)
        
        for i in range(self.C_arr_len):
            print ("tuning parameter: ", i)
            self.svm_linear = sksvm.LinearSVC(penalty="l2", dual=True, tol=1e-5, C=self.C_arr[i], max_iter=1000)
            self.tmp_err = 0.0
            for j in range(self.k):
                self.train_cv[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]] = False
                self.svm_linear.fit(self.X[self.train_cv, :], self.y[self.train_cv])
                self.y_pred_k = self.svm_linear.predict(self.X[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]])
                self.tmp_err += np.sum(self.y_pred_k != self.y[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]])
                self.train_cv[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]] = True
            self.cv_err_linsvm[i] = self.tmp_err / self.k  
        print("--- %s seconds ---" % (time.time() - start_time))            
        self.plot(self.C_arr,self.cv_err_linsvm)
        print(np.min(self.cv_err_linsvm))
        self.C_opt_linsvm = self.C_arr[np.argmin(self.cv_err_linsvm)]
        print ("C_opt: ",self.C_opt_linsvm)        

        self.svm_linear = sksvm.LinearSVC(penalty="l2", dual=True, tol=1e-5, C=self.C_opt_linsvm, max_iter=10000)
        print(self.svm_linear.n_support_)
        print(self.svm_linear.support_vectors_)
        self.svm_linear.fit(self.X[self.train, :], self.y[self.train])
        self.y_test_pred_linsvm = self.svm_linear.predict(self.X[self.test, :])

        print (("False rate",
                np.sum(self.y[self.test] != self.y_test_pred_linsvm) / float(self.n_test)))
        print (("False positive",
                np.sum(np.logical_and(self.y[self.test] == -1, self.y_test_pred_linsvm == 1)) / float(self.n_test_n)))
        print (("True positive",
                np.sum(np.logical_and(self.y[self.test] == 1, self.y_test_pred_linsvm == 1)) / float(self.n_test_p)))
        print (("False negative",
                np.sum(np.logical_and(self.y[self.test] == 1, self.y_test_pred_linsvm == -1)) / float(self.n_test_p)))
        print (("True negative",
                np.sum(np.logical_and(self.y[self.test] == -1, self.y_test_pred_linsvm == -1)) / float(self.n_test_n)))

        
    def plot(self,C_arr,cv_err_linsvm):
        self.fg = plt.figure(figsize=(5, 5))
        self.ax = self.fg.add_subplot(1, 1, 1)

        self.ax.set_xlabel("$\lambda$", size="large")
        self.ax.set_ylabel("err", size=25)
        self.ax.set_xscale("log")

        self.line_cv = self.ax.plot(C_arr, cv_err_linsvm/((self.n*9/10)*(1/self.k)), label="cv_err")
        self.ax.grid()

        self.ax.legend(loc="best", fontsize="xx-large")
        plt.savefig('CV.png')       



# In[7]:


class SVM_Kernel:
    def __init__(self,address,k,C_arr_len,C_arr_start,C_arr_end,gamma_start,gamma_end):
        self.matrix = data(address)
        self.data = self.matrix.deep_cleaning()
        self.X = self.data[0]
        self.y = self.data[1]
        self.k = k
        self.C_arr_len = C_arr_len
        self.C_arr_start = C_arr_start
        self.C_arr_end = C_arr_end
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
    
    def CV(self):
        # Codes borrowed from the tutorial class.
        np.random.seed(123)
        self.n = len(self.y)
        self.y = np.array(self.y)
        self.test = np.zeros(self.n, dtype=np.bool)
        self.n_p1 = np.sum(self.y == 1)
        self.test[npr.choice(np.nonzero(self.y)[0], int(self.n*1/10), replace=False)] = True
        self.train = np.logical_not(self.test)

        self.n_train = np.sum(self.train)
        self.n_test = np.sum(self.test)
        self.n_train_p = np.sum(self.y[self.train] == 1)
        self.n_train_n = np.sum(self.y[self.train] == -1)
        self.n_test_p = np.sum(self.y[self.test] == 1)
        self.n_test_n = np.sum(self.y[self.test] == -1)
        self.train_idx = np.nonzero(self.train)[0]
        npr.shuffle(self.train_idx)
        self.cv_grid = np.linspace(0, self.n_train, self.k + 1).astype(np.int32)
        

        start_time = time.time()
        self.train_cv = np.copy(self.train)
        self.test_cv = np.copy(self.train)
        self.tmp_err = 0.0

        
        start_time = time.time()
        self.C_arr = np.exp(np.linspace(self.C_arr_start,self.C_arr_end, self.C_arr_len))
        self.gm = np.exp(np.linspace(self.gamma_start, self.gamma_end, self.C_arr_len))
        self.CV = []
        for i in range(self.C_arr_len):
            for j in range(self.C_arr_len):
                self.CV.append((self.C_arr[i],self.gm[j]))

        self.cv_err_svm_kern = np.empty((self.C_arr_len)**2)
        self.train_cv = np.copy(self.train)
        self.t = 0
        
        for i in range(self.C_arr_len):
            for j in range(self.C_arr_len):
                print ("tuning parameter: ", self.t)
                self.svm_kern = sksvm.SVC(C=self.C_arr[i], kernel="rbf", gamma=self.gm[j], tol=1e-4, max_iter=-1)
                self.tmp_err = 0.0
                for j in range(self.k):
                    self.train_cv[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]] = False
                    self.svm_kern.fit(self.X[self.train_cv, :], self.y[self.train_cv])
                    self.y_pred_k = self.svm_kern.predict(self.X[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]])
                    self.tmp_err += np.sum(self.y_pred_k != self.y[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]])
                    self.train_cv[self.train_idx[self.cv_grid[j] : self.cv_grid[j + 1]]] = True
                self.cv_err_svm_kern[self.t] = self.tmp_err / self.k
                self.t = self.t+1
        print("--- %s seconds ---" % (time.time() - start_time))

        print(np.sum(self.cv_err_linsvm)/self.k)
        self.opt_kern = self.CV[np.argmin(self.cv_err_svm_kern)]
        print ("opt: ",self.opt_kern)       
        
        self.plot(self.gm,self.C_arr,self.cv_err_svm_kern)

        self.svm_kern = sksvm.SVC(C=self.opt_kern[0], kernel="poly", gamma=self.opt_kern[1], degree=3, tol=1e-4, max_iter=-1)
        self.svm_kern.fit(self.X[self.train, :], self.y[self.train])
        self.y_test_pred_kernel = self.svm_kern.predict(self.X[self.test, :])

        print (("False rate",
                np.sum(self.y[self.test] != self.y_test_pred_kernel) / float(self.n_test)))
        print (("False positive",
                np.sum(np.logical_and(self.y[self.test] == -1, self.y_test_pred_kernel == 1)) / float(self.n_test_n)))
        print (("True positive",
                np.sum(np.logical_and(self.y[self.test] == 1, self.y_test_pred_kernel == 1)) / float(self.n_test_p)))
        print (("False negative",
                np.sum(np.logical_and(self.y[self.test] == 1, self.y_test_pred_kernel == -1)) / float(self.n_test_p)))
        print (("True negative",
                np.sum(np.logical_and(self.y[self.test] == -1, self.y_test_pred_kernel == -1)) / float(self.n_test_n)))

        
    def plot(self,gm,C_arr,cv_err_svm_kern):
        self.new = []
        self.num = (self.n*9/10)*(1/self.k)
        print(self.num)
        for i in range(self.C_arr_len):
            self.new.append(cv_err_svm_kern[self.C_arr_len*i:self.C_arr_len*i+self.C_arr_len]/self.num)
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(self.new, interpolation='nearest', cmap=plt.cm.hot)
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gm)), np.round(gm,4), rotation=45)
        plt.yticks(np.arange(len(C_arr)), np.round(C_arr,4))
        plt.title('CV Error')
        plt.savefig('CV_Kernel.png')
        plt.show()   

    
    


# In[15]:

Maxent(prewd+'/data/Lexique_test.txt',5,1).CV()
SVM_Linear(prewd+'/data/Lexique_test.txt',5,100,-10,-2).CV()
SVM_Kernel(prewd+'/data/Lexique_test.txt',5,10,-3,1,-10,-4).CV()

