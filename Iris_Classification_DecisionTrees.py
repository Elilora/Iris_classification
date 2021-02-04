import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score,plot_confusion_matrix

iris =pd.read_csv(r'C:\Users\HP\Documents\PycharmProjects\dataset\Iris.csv')
iris.drop('Id',axis=1, inplace = True)
x = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris['Species']

corr_matrix = iris.corr().round(2) 
 
fig, ax = plt.subplots(figsize=(10,8))
cmap=sb.diverging_palette(0,245.4,68.8,50,as_cmap=True)
mask=np.triu(np.ones_like(corr_matrix, dtype=np.bool))
heatmap=sb.heatmap(corr_matrix, vmin= -0.8,fmt=".2f",annot=True,annot_kws={"size":10},linewidth=0.2,
                   cmap=cmap,mask=mask,square=True,cbar_kws={'shrink':.8})
plt.show()
heatmap.get_figure().savefig('iris_heatmap.png')

sns.pairplot(iris, hue='Species', height=2)

model=DecisionTreeClassifier()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=3, stratify=y)

model.fit(x_train,y_train)
y_pred=model.predict(x_test)

plot_confusion_matrix(model, x_test, y_test, cmap=plt.cm.Blues)

print("Accuracy Score : ", accuracy_score(y_test,y_pred))
print("Precision Score : ", precision_score(y_test,y_pred, average='weighted'))
print("Recall Score : ", recall_score(y_test,y_pred, average= 'weighted'))
print("F1 Score : ",f1_score(y_test,y_pred, average="weighted"))
