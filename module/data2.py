from pandas import read_csv, DataFrame, Series, get_dummies
from platform import system
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from geopy import geocoders
from numpy import where

class Data:

    def __init__(self, path):

        #Lê arquivo disponibilizado pelo Avito
        if system() == 'Windows': self.__data = read_csv(path, encoding='latin1');
        else: self.__data = read_csv(path, encoding='utf8');
        self.__dp = DataFrame(index=self.__data.index.tolist(),
                              columns=self.__data.columns[
                                  (self.__data.columns != 'city')
                                  &(self.__data.columns != 'region')
                                  &(self.__data.columns != 'item_id')
                                  &(self.__data.columns != 'user_id')
                                  &(self.__data.columns != 'image')
                                  &(self.__data.columns != 'image_top_1')].tolist());
        print('Dados carregados');

        '''
        #df = DataFrame();

        #Stop words russas
        stopWords = open('file/stopwords.txt', encoding='utf8').read().split('\n');

        #Obetendo cabeçalhos
            [
                'region', 'city', 'parent_category_name', 'category_name',
                'param_1', 'param_2', 'param_3', 'title', 'description', 'price',
                'user_type'
            ]
            
        heads = self.__data.columns.values[
            (self.__data.columns.values != 'item_id')
            &(self.__data.columns.values != 'user_id')
            &(self.__data.columns.values != 'activation_date')
            &(self.__data.columns.values != 'image')
            &(self.__data.dtypes == 'object') ];
        
        #Stop words russas
        stopWords = open('file/stopwords.txt', encoding='utf8').read().split('\n');
        print("Stop Words carregadas");

        #Configura o  tfidf
        vectorizer = TfidfVectorizer(sublinear_tf=True,
                                stop_words=stopWords,
                                min_df=2,
                                max_df=0.8);
        print('Tfidf configurado');
        
        #Pré processa os dados
        for head in heads:
        
            vectorizer.fit_transform(self.__data[head][self.__data[head].notnull()].tolist());
            #print(vectorizer.get_feature_names());
            print('Pré-processou', head);
        
        #print(df);
        print('Pré-processo concluído');
        '''

    #Exibe os dados
    def read(self): print(self.__data);

    #Obtém os dados
    def getData(self): return self.__dp;

    #Obtém os cabeçalhos
    def getHeads(self): return self.__data.columns.values;

    def getCityRegion(self):
        cityRegion = self.__data.loc[:, ["city", "region"]].apply(lambda l: " ".join(l), axis=1)
        g = geocoders.Nominatim();
        coord = [];
        for location in cityRegion:
            if system() == 'Windows':
                geocode = g.geocode("Самара Самарская область", timeout=10, language='pt');
                coord.append([geocode.latitude, geocode.longitude]);
                break;
            geocode = g.geocode(location, timeout=10, language='en');
            coord.append([geocode.latitude, geocode.longitude]);
        #self.__dp['city_region'] = coord;
        return coord;

    def getCategory(self): pass;
        
    def getParentCategoryName(self): pass;

    def getCategoryName(self): pass;

    def getParam1(self): pass;

    def getParam2(self): pass;

    def getParam3(self): pass;

    def getTitle(self): pass;

    def getDescription(self): pass;

    def getPrice(self): pass;

    def getItemSeqNumber(self): pass;

    def getActivationDate(self): pass;

    def getUserType(self): pass;

    def getImage(self): pass;

    def getImageTop1(self): pass;

    def getDealProbability(self): pass;

    #Obtém a classificação
    def getClassification(self, classification, value): return where(self.__data[classification].unique() == value)[0][0];

d = Data('file/train.csv');
print(d.getCityRegion());
