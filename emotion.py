import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB


from dataHandler import DataHandler
from boW import EOS_token

RSEED = 50
SIZE = 10
NCLASS = 3
TFIDF = False
NORMALIZE = False

d = DataHandler()
d.createDictionary()

#prepare data
data = d.readDataPreproc(pre=True)[1:]

x = [d.askDictionary.seq2tensor(xi[1], tfidf=TFIDF) for xi in data]
x = np.array(x)

if NORMALIZE:
    x = preprocessing.normalize(x)

y = [0 if yi[2]=="alegria" else 1 if yi[2]=="neutro" else 2 for yi in data]
if NCLASS == 3:
    y = [0 if yi[2]=="alegria" else 1 if yi[2]=="neutro" else 2 for yi in data]
else:
    y = [0 if yi[2]=="alegria" else 1 for yi in data]
y=np.array(y)
n0 = np.sum(y==0)
n1 = np.sum(y==1)
n2 = np.sum(y==2)
print("Quantidade de exemplos alegres: {}\nQuantidade de exemplos neutros: {}\nQuantidade de exemplos tristes: {}\n".format(n0,n1,n2))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# train RF
#######################################
model = RandomForestClassifier(n_estimators=8, max_depth = 4,
							   random_state=RSEED, verbose = 0)

model.fit(x_train, y_train)
y_pred_rf = model.predict(x_test)


# train SVM 
#######################################
#model_svm = svm.SVC(kernel='linear', C=0.01, verbose=True)
model_svm = svm.SVC(kernel='rbf', C=48, gamma=0.0002, random_state=RSEED)
model_svm.fit(x_train, y_train)
y_pred_svm = model_svm.predict(x_test)


#train naive bayes
#######################################
model_gnb = GaussianNB()
model_gnb.fit(x_train, y_train)
y_pred_gnb = model_gnb.predict(x_test)

model_mnb = MultinomialNB()
model_mnb.fit(x_train, y_train)
y_pred_mnb = model_mnb.predict(x_test)


print('\nAcuracia RF {:.2f}%'.format(accuracy_score(y_pred_rf, y_test)*100))
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)

print('\nAcuracia SVM {:.2f}%'.format(accuracy_score(y_pred_svm, y_test)*100))
cm = confusion_matrix(y_test, y_pred_svm)
print(cm)

print('\nAcuracia GNB {:.2f}%'.format(accuracy_score(y_pred_gnb, y_test)*100))
cm = confusion_matrix(y_test, y_pred_gnb)
print(cm)

print('\nAcuracia MNB {:.2f}%'.format(accuracy_score(y_pred_mnb, y_test)*100))
cm = confusion_matrix(y_test, y_pred_mnb)
print(cm)
