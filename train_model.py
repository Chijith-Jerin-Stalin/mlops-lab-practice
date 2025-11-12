import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("Iris.csv")

X = df.iloc[:,1:-1]
le = LabelEncoder()

y = le.fit_transform(df.iloc[:,-1])

X_train, X_test, y_trian, y_test = train_test_split(X,y,train_size=0.25,random_state=0)

rdcmodel = RandomForestClassifier(n_estimators=200,random_state=42)
drcmodel = DecisionTreeClassifier(criterion="entropy",random_state=0)
lrmodel = LinearRegression()

rdcmodel.fit(X_train,y_trian)
drcmodel.fit(X_train,y_trian)
lrmodel.fit(X_train,y_trian)

with open("rdcmodel.pkl","wb") as f:
    pickle.dump(rdcmodel,f)

with open("drcmodel.pkl","wb") as f:
    pickle.dump(drcmodel,f)

with open("lrmodel.pkl","wb") as f:
    pickle.dump(lrmodel,f)

print("✅ Model trained and saved as model.pkl")