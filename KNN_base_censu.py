import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

with open('census.pkl', 'rb') as f:
    x_censo_treinamento,y_censo_treinamento,x_censo_teste,y_censo_teste =pickle.load(f)


print(f'x_treinamento: {x_censo_treinamento.shape}')
print(f'y_treinamento: {y_censo_treinamento.shape}')

print(f'x_teste: {x_censo_teste.shape}')
print(f'y_teste:{y_censo_teste.shape}')

#definição do algoritmo a ser utilizado
knn_censo= KNeighborsClassifier(n_neighbors=5,metric= 'minkowski', p=2 )
#definição do realizando o treinamento da algoritmo
knn_censo.fit(x_censo_treinamento,y_censo_treinamento)
#
previsoes =knn_censo.predict(x_censo_teste)

precisao =accuracy_score(y_censo_teste,previsoes)
cm = ConfusionMatrix(knn_censo)
cm.fit(x_censo_treinamento,y_censo_treinamento)
score_cm= cm.score(x_censo_teste,y_censo_teste)
plt.savefig("matriz_de_confusao.png", dpi =300, format='png') 
cm.show()
print(classification_report(y_censo_teste,previsoes))
print(f'{precisao}')