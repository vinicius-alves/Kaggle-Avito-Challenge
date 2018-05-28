from pandas import read_csv

data = read_csv('../data/train_small.csv');

#StopWords e Minúsculo

'''
Obtém as colunas que sejam do tipo string e diferentes de:
item_id, user_id, activation_date e image
'''
heads = data.columns.values[
            (data.columns.values != 'item_id')
            &(data.columns.values != 'user_id')
            &(data.columns.values != 'activation_date')
            &(data.columns.values != 'image')
            &(data.dtypes == 'object') ];

#Obtém as stop words russas
stopWords = open('file/stopwords.txt', encoding='utf8').read().split('\n');
for head in heads:
    #Coloca todos valores do tipo string minúsculo
    data[head] = data[head].str.lower();
    #Retira as stop words dos valores
    data[head].replace(stopWords, '', inplace=True);

#Retirando os valores nulos dos Params
'''
Obtém os valores mais comuns em cada coluna param.
Os valores obtidos são de acordo com as colunas anteriores, ou seja,
ele obtém o valor mais comum do param_2 baseado no valor mais comum do
param_1 por exemplo.
'''
commom = [
    data['param_1'].value_counts().index[0],
    data['param_2'][data['param_1'] == data['param_1'].value_counts().index[0]].value_counts().index[0],
    data['param_3'][ (data['param_1'] == data['param_1'].value_counts().index[0]) & (data['param_2'] == data['param_2'].value_counts().index[0])].value_counts().index[0]
    ]

'''
Realiza a troca dos valores nulos em cada param pelos valores mais comuns
em cada param de acordo com o valor mais comum do param anterior
'''
data['param_1'].where(data['param_1'].notnull(), commom[0], inplace=True);
data['param_2'].where(data['param_2'].notnull(), commom[1], inplace=True);
data['param_3'].where(data['param_3'].notnull(), commom[2], inplace=True);
