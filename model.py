import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

import pickle




df=pd.read_csv('pima-data.csv')
diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)



df['diabetes'] = df['diabetes'].map(diabetes_map)

X=df[['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age','skin']]
y=df['diabetes']
y=y.astype('int')
X=X.astype('int')




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(predictions)

a=[]
for i in range(9):
    j = int(input("enter:"))
    a.append(j)
a_predict=logmodel.predict(np.array(a).reshape(1,9))
print(a_predict)

pickle.dump(logmodel,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
