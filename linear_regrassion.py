import pandas as pd
from pandas import read_csv
A = read_csv("G:\\Python_csv\\50_Startups.csv")
X = A[["RND"]]
Y = A[["PROFIT"]]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=25)
import seaborn as sb
sb.distplot(Y)
sb.distplot(ytrain)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(xtrain,ytrain)
pred = model.predict(xtest)
from sklearn.metrics import r2_score
rsq = r2_score(ytest,pred)
print("R-squared is %.2f"%(rsq))
