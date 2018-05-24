from pandas import read_csv
from platform import system
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Data:

    def __init__(self, path):

        #Lê arquivo disponibilizado pelo Avito
        if system() == 'Windows': self.__data = read_csv(path, encoding='latin1');
        else: self.__data = read_csv(path, encoding='utf8');
        print('Dados carregados');

        #Stop words russas
        stopWords = open('file/stopwords.txt', encoding='utf8').read().split('\n');
        
        #Obetendo cabeçalhos
        heads = self.__data.columns.values[(self.__data.columns.values !='activation_date') & (self.__data.dtypes == 'object') ];
        
        #Stop words russas
        stopWords = open('file/stopwords.txt', encoding='utf8').read().split('\n');
        print("Stop Words carregadas");
        
        #Pré processa os dados
        for head in heads:
        
            tfidf = TfidfVectorizer(sublinear_tf=True, stop_words=stopWords, max_df=0.8);
            print(tfidf.fit_transform(self.__data[head][self.__data[head].notnull()].tolist()));
            print('Pré-processou', head);

        print('Pré-processo concluído');

    #Exibe os dados
    def read(self): print(self.__data);

    #Obtém os dados
    def getData(self): return self.__data;

    #Obtém os cabeçalhos
    def getHeads(self): return self.__data.columns.values;

    #Obtém a classificação
    def getClassification(self, classification, value): return np.where(self.__data[classification].unique() == value)[0][0];

d = Data('file/train.csv');
