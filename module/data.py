import pandas as pd
import numpy as np
import platform

class Data:

    def __init__(self, path):

        #Lê arquivo disponibilizado pelo Avito
        if platform.system() == 'Windows': self.__data = pd.read_csv(path, encoding='latin1');
        else: self.__data = pd.read_csv(path, encoding='utf8');
        print('Dados carregados');
        
        #Obetendo cabeçalhos
        heads = self.__data.columns.values[(self.__data.columns.values !='activation_date') & (self.__data.dtypes == 'object') ];

        #words = set();
        
        #Stop words russas
        stopWords = open('file/stopwords.txt', encoding='utf8').read().split('\n');
        print("Stop Words carregadas");

        #Pré processa os dados
        for head in heads:
                
            #Passa as palavras para minúsculo
            self.__data[head] = self.__data[head].str.lower();
                
            #Remove stop words
            self.__data[head].replace(stopWords, '', inplace=True);

            #Obtém as palavras que aparecem em cada cabeçalho
            #self.__data[head].str.split(' ').apply(words.update);
                
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
#d.read();
