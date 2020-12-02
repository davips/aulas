# Importa coisas para abrir ARFF direto da URL.
import pandas as pd
import urllib.request
from io import StringIO
from scipy.io import arff
import numpy as np
import pandas as pd


def df_remoto(url_do_arff, class_index=-1):
    """
    Exemplo de uso: 
    X,y = Xy_remoto("https://raw.githubusercontent.com/lpfgarcia/ucipp/master/uci/hepatitis.arff")
    """
    # Abre um conjunto de dados mais difícil: banana.
    file = urllib.request.urlopen(url_do_arff).read().decode()
    data = arff.loadarff(StringIO(file))
    # pd.read_csv(BytesIO(csv), encoding="latin1"))
    # Codifica todos os valores de binário para texto/float.
    def f(cell):
      v = cell.decode() if isinstance(cell, bytes) else cell
      try:
        return float(v)
      except ValueError:
        return v
    return pd.DataFrame(data[0]).applymap(f)

#  Binariza atributos nominais (cada atributo com n valores distintos vira n colunas).
from sklearn.preprocessing import OneHotEncoder as OHE
def nom_to_num(X):
  ohe = OHE()
  return ohe.fit_transform(X)


# Importa de arquivo local (ou do google drive)
# LEMBRETE: scipy funciona em vez de liac-arff?
def Xy_local(filename, class_index=-1):
    """
    Exemplo de uso: 
    X,y = Xy_local("/home/usuario/dataset.arff")
    """
    # !pip install liac-arff
    
    # Abre conjunto de dados.
    file = open(filename, "r")
    dic = arff.load(file, encode_nominal=False)
    data = np.array(dic["data"])

    # Converte para matrizes (numpy/sklearn).
    X = np.array(data)
    np.delete(X, class_index, axis=1)
    y = np.array(data[:, class_index])
    return X, y

# Função para avaliar acurácia.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

def acc_std(classifier, X, y):
  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  accs = cross_val_score(classifier, X, y, cv=cv)
  return round(np.mean(accs)*1000)/1000, round(np.std(accs)*1000)/1000


# Substitui valores ausentes pela moda.

def imputa_moda(df, cols=None):
  if cols is None:
      cols = mode.columns
        
  # Tira a cópia do dataframe, pois vai alterá-lo.
  df_mode = df.copy()

  for column in cols:
      df_mode[column].fillna(df_mode[column].mode()[0], inplace=True)
  return df_mode

# Converte dataframe do pandas em matrizes numpy para uso no sklearn.
def df_to_Xy(df):
  X = df.to_numpy()[:, :-1]
  y = df.to_numpy()[:, -1]
  return X, y


# Plota árvore
from sklearn.tree import plot_tree as plot
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as TREE
def mostra_arvore(tree, X, y, altura=4):
  plt.figsize=150
  tree = TREE(criterion="gini", random_state=42, max_depth=altura)
  tree.fit(X, y)  # treina no conjunto todo para analisar a árvore
  fig, ax = plt.subplots(figsize=(14, 6))
  plot(tree, ax=ax, filled=True)
  plt.show()
