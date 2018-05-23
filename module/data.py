import pandas as pd
import numpy as np

class Data:

    def __init__(self, path):
        self.__data = pd.read_csv(path, encoding='latin-1');

        #Stop Words russas
        stopWords = open('file/stopwords-ru.txt', encoding='latin-1').read().split('\n');

        #Obetendo cabeçalhos
        for head in self.__data.columns.values:

            #Verifica se os dados do cabeçalho são do tipo string
            if self.__data[head].dtype == 'O':

                #Passa as palavras para minúsculo
                self.__data[head] = self.__data[head].str.lower();

                #Remove stop words
                self.__data[head].replace(stopWords, '', inplace=True);

    #Exibe os dados
    def read(self): print(self.__data);

    #Obtém os dados
    def getData(self): return self.__data;

    #Obtém os cabeçalhos
    def getHeads(self): return self.__data.columns.values;

    #Obtém a classificação
    def getClassification(self, classification, value): return np.where(self.__data[classification].unique() == value)[0][0];

data = Data('file/train.csv');
data.read();
