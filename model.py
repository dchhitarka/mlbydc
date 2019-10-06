import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC
import pickle

df = pd.read_csv('dataset.csv')
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = tts(x,y,test_size=0.25, random_state=1)
classifier = SVC(kernel='linear')
classifier.fit(x_train,y_train)

pickle.dump(classifier, open("model.pkl","wb"))
