import numpy as np
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')

# Carga y combinación de datos
tweets_elon = pd.read_csv('C:/Users/anton/Documents/others/code/Analisis de sentimiento/datos_tweets_elonmusk.csv')
tweets_edlee = pd.read_csv('C:/Users/anton/Documents/others/code/Analisis de sentimiento/datos_tweets_mayoredlee.csv')
tweets_bgates = pd.read_csv('C:/Users/anton/Documents/others/code/Analisis de sentimiento/datos_tweets_BillGates.csv')

# Unir dataframes
tweets = pd.concat([tweets_elon, tweets_edlee, tweets_bgates], ignore_index=True)

# Selección y renombramiento de columnas
tweets = tweets[['screen_name', 'created_at', 'status_id', 'text']]
tweets.columns = ['autor', 'fecha', 'id', 'texto']

# Parseo de fechas
tweets['fecha'] = pd.to_datetime(tweets['fecha'])

# Función de limpieza y tokenización
def limpiar_tokenizar(texto):
    '''
    Limpia y tokeniza el texto en palabras individuales.
    '''
    nuevo_texto = texto.lower()
    nuevo_texto = re.sub(r'http\S+', ' ', nuevo_texto)
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_texto = re.sub(regex, ' ', nuevo_texto)
    nuevo_texto = re.sub(r'\d+', ' ', nuevo_texto)
    nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
    nuevo_texto = nuevo_texto.split(sep=' ')
    nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
    return nuevo_texto

# Aplicación de la función de limpieza y tokenización
tweets['texto_tokenizado'] = tweets['texto'].apply(limpiar_tokenizar)
tweets_tidy = tweets.explode(column='texto_tokenizado').drop(columns='texto').rename(columns={'texto_tokenizado': 'token'})

# Palabras totales utilizadas por cada autor
print('Palabras totales por autor:')
print(tweets_tidy.groupby(by='autor')['token'].count())

# Longitud media y desviación de los tweets
temp_df = pd.DataFrame(tweets_tidy.groupby(by=["autor", "id"])["token"].count())
print('Longitud media y desviación de los tweets de cada autor:')
print(temp_df.reset_index().groupby("autor")["token"].agg(['mean', 'std']))

# Top 5 palabras más utilizadas por cada autor
print('Top 5 palabras más utilizadas por cada autor:')
print(tweets_tidy.groupby(['autor', 'token'])['token'].count().reset_index(name='count').groupby('autor').apply(lambda x: x.sort_values('count', ascending=False).head(5)))

# Excluir stopwords
stop_words = list(stopwords.words('english')) + ["amp", "xa", "xe"]
tweets_tidy = tweets_tidy[~(tweets_tidy["token"].isin(stop_words))]

# Top 10 palabras por autor (sin stopwords)
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 7))
for i, autor in enumerate(tweets_tidy.autor.unique()):
    df_temp = tweets_tidy[tweets_tidy.autor == autor]
    counts = df_temp['token'].value_counts(ascending=False).head(10)
    counts.plot(kind='barh', color='firebrick', ax=axs[i])
    axs[i].invert_yaxis()
    axs[i].set_title(autor)
fig.tight_layout()
plt.show()

# Pivotado de datos
tweets_pivot = tweets_tidy.groupby(["autor", "token"])["token"].agg(["count"]).reset_index().pivot(index="token", columns="autor", values="count")
tweets_pivot.columns.name = None

# Test de correlación (coseno)
def similitud_coseno(a, b):
    distancia = cosine(a, b)
    return 1 - distancia

correlacion = tweets_pivot.corr(method=similitud_coseno)
print('Correlación de coseno entre autores:')
print(correlacion)

# Gráfico de correlación
f, ax = plt.subplots(figsize=(6, 4))
temp = tweets_pivot.dropna()
sns.regplot(x=np.log(temp.elonmusk), y=np.log(temp.BillGates), scatter_kws={'alpha': 0.05}, ax=ax)
for i in np.random.choice(range(temp.shape[0]), 100):
    ax.annotate(text=temp.index[i], xy=(np.log(temp.elonmusk[i]), np.log(temp.BillGates[i])), alpha=0.7)
plt.show()

# Número de palabras comunes entre autores
palabras_elon = set(tweets_tidy[tweets_tidy.autor == 'elonmusk']['token'])
palabras_bill = set(tweets_tidy[tweets_tidy.autor == 'BillGates']['token'])
palabras_edlee = set(tweets_tidy[tweets_tidy.autor == 'mayoredlee']['token'])

print(f"Palabras comunes entre Elon Musk y Ed Lee: {len(palabras_elon.intersection(palabras_edlee))}")
print(f"Palabras comunes entre Elon Musk y Bill Gates: {len(palabras_elon.intersection(palabras_bill))}")

# Cálculo del log of odds ratio de cada palabra (elonmusk vs mayoredlee)
tweets_pivot = tweets_tidy.groupby(["autor", "token"])["token"].agg(["count"]).reset_index().pivot(index="token", columns="autor", values="count").fillna(value=0)
tweets_pivot.columns.name = None

tweets_unpivot = tweets_pivot.melt(value_name='n', var_name='autor', ignore_index=False).reset_index()
tweets_unpivot = tweets_unpivot[tweets_unpivot.autor.isin(['elonmusk', 'mayoredlee'])]
tweets_unpivot = tweets_unpivot.merge(tweets_tidy.groupby('autor')['token'].count().rename('N'), how='left', on='autor')

tweets_logOdds = tweets_unpivot.copy()
tweets_logOdds['odds'] = (tweets_logOdds.n + 1) / (tweets_logOdds.N + 1)
tweets_logOdds = tweets_logOdds[['token', 'autor', 'odds']].pivot(index='token', columns='autor', values='odds')
tweets_logOdds.columns.name = None

tweets_logOdds['log_odds'] = np.log(tweets_logOdds.elonmusk / tweets_logOdds.mayoredlee)
tweets_logOdds['abs_log_odds'] = np.abs(tweets_logOdds.log_odds)
tweets_logOdds['autor_frecuente'] = np.where(tweets_logOdds.log_odds > 0, "elonmusk", "mayoredlee")

print('Top 10 palabras más diferenciadoras:')
print(tweets_logOdds.sort_values('abs_log_odds', ascending=False).head(10))

# Top 15 palabras más características de cada autor
top_30 = tweets_logOdds[['log_odds', 'abs_log_odds', 'autor_frecuente']].groupby('autor_frecuente').apply(lambda x: x.nlargest(15, columns='abs_log_odds').reset_index()).reset_index(drop=True).sort_values('log_odds')

f, ax = plt.subplots(figsize=(4, 7))
sns.barplot(x='log_odds', y='token', hue='autor_frecuente', data=top_30, ax=ax)
ax.set_title('Top 15 palabras más características de cada autor')
ax.set_xlabel('log odds ratio (@elonmusk / mayoredlee)')
plt.show()

# Cálculo term-frequency (tf)
tf = tweets_tidy.copy()
tf = tf.groupby(["id", "token"])["token"].agg(["count"]).reset_index()
tf['total_count'] = tf.groupby('id')['count'].transform(sum)
tf['tf'] = tf["count"] / tf["total_count"]

# Inverse document frequency (idf)
idf = tweets_tidy.copy()
total_documents = idf["id"].drop_duplicates().count()
idf = idf.groupby(["token", "id"])["token"].agg(["count"]).reset_index()
idf['n_documentos'] = idf.groupby('token')['count'].transform(sum)
idf['idf'] = np.log(total_documents / idf['n_documentos'])
idf = idf[["token", "n_documentos", "idf"]].drop_duplicates()

# Term Frequency - Inverse Document Frequency (tf-idf)
tf_idf = pd.merge(left=tf, right=idf, on="token")
tf_idf["tf_idf"] = tf_idf["tf"] * tf_idf["idf"]

# Reparto train y test
datos_X = tweets.loc[tweets.autor.isin(['elonmusk', 'mayoredlee']), 'texto']
datos_y = tweets.loc[tweets.autor.isin(['elonmusk', 'mayoredlee']), 'autor']
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.01)
tfidf_data = tfidf_vectorizer.fit_transform(datos_X)

X_train, X_test, y_train, y_test = train_test_split(tfidf_data, datos_y, test_size=0.3, random_state=1)

# Entrenamiento del modelo SVM
def entrenar_modelo(X_train, y_train):
    modelo_svm_lineal = svm.SVC(kernel="linear", C=1.0)
    modelo_svm_lineal.fit(X=X_train, y=y_train)
    return modelo_svm_lineal

# Evaluación del modelo
def evaluar_modelo(modelo, X_test, y_test):
    predicciones_test = modelo.predict(X=X_test)
    error = 100 * (y_test != predicciones_test).mean()
    return error, confusion_matrix(y_true=y_test, y_pred=predicciones_test)

modelo_final = entrenar_modelo(X_train, y_train)
error, matriz_confusion = evaluar_modelo(modelo_final, X_test, y_test)

print(f"Error de test: {error}%")
print("Matriz de confusión")
print(pd.DataFrame(matriz_confusion, columns=["Elon Musk", "Mayor Ed Lee"], index=["Elon Musk", "Mayor Ed Lee"]))
