# -*- coding: utf-8 -*-
"""
Last editted on Thu Jul 25 08:15:55 2019

@author: Matheus Di Vaio
"""

import numpy as np

import E_TS_NN_plotter as plotter

def getDataFromTXT(seq_len, data_source, normalize=False, normalizador = 1):
    


    Xt_Pocos_PCA_06_12_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_Pocos_PCA_06_12_50.txt"
    Xt_Pocos_PCA_20_40_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_Pocos_PCA_20_40_50.txt"
    Xt_Pocos_PCA_50_100_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_Pocos_PCA_50_100_50.txt"
    
    
    Xt_Pocos_PCA_50_30_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_Pocos_PCA_50_30_50.txt"
    Xt_Pocos_PCA_NEW_50_30_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_Pocos_PCA_NEW_50_30_50.txt"
    Xt_Pocos_PCA_NEWNORM_50_30_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_Pocos_PCA_NEWNORM_50_30_50.txt"
    Xt_Pocos_PCA_NEWNORM_ssmm_50_30_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_Pocos_PCA_NEWNORM_ssmm_50_30_50.txt"
    

    Xt_PocosB_06_12_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_PocosB_06_12_50.txt"
    Xt_PocosB_20_40_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_PocosB_20_40_50.txt"
    Xt_PocosB_50_100_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_PocosB_50_100_50.txt"

    Xt_PocoA_06_12_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_PocoA_06_12_50.txt"
    Xt_PocoA_20_40_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_PocoA_20_40_50.txt"
    Xt_PocoA_50_100_50_path = "G:\Meu Drive\Acadêmico\BRDrilling_Matheus\Programacao\Processamento de Dados\Dados\TXT\Xt_PocoA_50_100_50.txt"
    
    
    comp_06_12_50   = [23466, 6609, 2324, 7834]
    comp_20_40_50   = [22449, 8245, 2248, 7752]
    comp_50_100_50  = [20408, 5489, 2086, 6960]
    comp_50_30_50   = [20243, 5703, 1988, 6585]
    comp_NEW_50_30_50   = [20243-7, 5703, 1988, 6585]
    comp_NEWNORM_50_30_50   = [20243-70-20, 5703, 1988, 6585]

    comp_old_50_100_50  = [20332, 5165, 1534, 6903]

    DATA = np.loadtxt(Xt_Pocos_PCA_50_100_50_path)
    comp = comp_50_100_50


    '''    
    DATA_Label = ["ElapsedTime",         # 0          
                  "RPM",                 # 1           -
                  "SWOB",                # 2           -
                  "STOR",                # 3           -
                  "ROP",                 # 4           -
                  "SS_STOR",             # 5           -
                  "SPPA",                # 6
                  "SS",                  # 7           -           -
                  "score1",   # Geral      8           -
                  "score2",   # Geral      9           -
                  "score3",   # Geral      1           -
                  "score4",   # Geral      11
                  "score5",   # Geral      12
                  "score1",   # Padrao     13
                  "score2",   # Padrao     14      
                  "score3",   # Padrao     15
                  "score4",   # Padrao     16
                  "score5"]   # Padrao     17
    '''
    
    


    if data_source == 'RAW Data':
        dataX = DATA[:, 1:6]
    elif data_source == 'PCA Geral':
        dataX = DATA[:, 8:13]
#        dataX = DATA[:, 8:38]
#        dataX = DATA[:, 8:13]
    elif data_source == 'PCA Padrao':
        dataX = DATA[:, 13:]


#        dataX = DATA[:, 13:16]
#        dataX = DATA[:, 38:]
#        dataX = DATA[:, 38:38+10]



    dataY = DATA[:, 7].reshape(DATA.shape[0], -1)
#    dataY = DATA[:, -1].reshape(DATA.shape[0], -1)
    dataY = dataY/normalizador
    
    maxDataLen = DATA.shape[0] // seq_len * seq_len

    dataX = dataX[:maxDataLen]
    dataY = dataY[:maxDataLen]

    if normalize == True:
        for i in range(dataX.shape[1]):
            dataX[:, i] = (dataX[:, i] - np.mean(dataX[:, i])) / np.std(dataX[:, i])

#        for i in range(dataY.shape[1]):
#            dataY[:, i] = (dataY[:, i] - np.mean(dataY[:, i])) / np.std(dataY[:, i])

    return dataX, dataY, comp



def slicer_for_sequential_data(dataX,
                               dataY,
                               comp, 
                               normalizador,
                               seq_len=10, 
                               avanco=1,  
                               bins = 50,
                               domain = 'ABAA', 
                               RANDOM = True,
                               equalize_data=False,
                               ):


    """
        Slices a matrix with format [n_samples x dimension] in:
        test, train and validation with it's respective "proportion"
        
        dataX[n_samples, dimension] = matrix with the data containing the input matrix
        dataY[n_samples, dimension] = matrix with the data containing the target matrix
        
#        shuffle_dataset: True or False. It shuffles the dataX and dataY the 
#        same way. Respecting the seq_len
    """
    
    

    if len(dataX.shape) == 1:
        input_dim = 1
    else:
        input_dim = dataX.shape[1]

    if len(dataY.shape) == 1:
        output_dim = 1
    else:
        output_dim = dataY.shape[1]

    slicesX2 = [dataX[i : i + seq_len] for i in range(len(dataX) - seq_len - avanco)]
    slicesX2 = np.stack(slicesX2, axis=0)
    slicesX2 = slicesX2.reshape(slicesX2.shape[0], input_dim * seq_len)
    slicesX = slicesX2

    slicesY2 = [
        dataY[i + seq_len + avanco] for i in range(len(dataY) - seq_len - avanco)
    ]
    slicesY2 = np.stack(slicesY2, axis=0)
    slicesY = slicesY2
 
    gap = [[0,    0],
           [0,    0],
           [0,    0],
           [0,    0],
           ]
    
    comp_soma = np.cumsum(comp)-10
    intervalos = [np.arange(0 + gap[0][0], comp_soma[0] + gap[0][1])]  
    proporcoes = [0.75, 0.25]
    
    
    for i in range(len(comp)-1):
        intervalos.append(np.arange(comp_soma[i] + gap[i+1][0], comp_soma[i+1] + gap[i+1][1]))
    
    XwellArA = slicesX[intervalos[0][0]:intervalos[0][-1], :]
    XwellArB = slicesX[intervalos[1][0]:intervalos[1][-1], :]
    XwellBr1 = slicesX[intervalos[2][0]:intervalos[2][-1], :]
    XwellBr2 = slicesX[intervalos[3][0]:intervalos[3][-1], :]
    
    YwellArA = slicesY[intervalos[0][0]:intervalos[0][-1], :]
    YwellArB = slicesY[intervalos[1][0]:intervalos[1][-1], :]
    YwellBr1 = slicesY[intervalos[2][0]:intervalos[2][-1], :]
    YwellBr2 = slicesY[intervalos[3][0]:intervalos[3][-1], :]


    
    indexWArA_random = np.arange(len(XwellArA)); np.random.shuffle(indexWArA_random)
    indexWArB_random = np.arange(len(XwellArB)); np.random.shuffle(indexWArB_random)
    indexWBr1_random = np.arange(len(XwellBr1)); np.random.shuffle(indexWBr1_random)
    indexWBr2_random = np.arange(len(XwellBr2)); np.random.shuffle(indexWBr2_random)
    
    indexWArA_seq = np.arange(len(XwellArA))
    indexWArB_seq = np.arange(len(XwellArB))
    indexWBr1_seq = np.arange(len(XwellBr1))
    indexWBr2_seq = np.arange(len(XwellBr2))
    
    indexWArA = np.arange(len(XwellArA))
    indexWArB = np.arange(len(XwellArB))
    indexWBr1 = np.arange(len(XwellBr1))
    indexWBr2 = np.arange(len(XwellBr2))
    
    train = []
    test = []
    val = []
    

    if domain == 'ABAAxABAA':
        if RANDOM is True:
        
            indexWArA_train = indexWArA_random[:int(len(indexWArA)*proporcoes[0])]; 
            indexWArA_train = np.sort(indexWArA_train, axis=0)
            indexWArA_test  = indexWArA_random[int(len(indexWArA)*proporcoes[0]):]; 
            indexWArA_test  = np.sort(indexWArA_test, axis=0) 
            indexWArB_train = indexWArB_random[:int(len(indexWArB)*proporcoes[0])]; 
            indexWArB_train = np.sort(indexWArB_train, axis=0)
            indexWArB_test  = indexWArB_random[int(len(indexWArB)*proporcoes[0]):]; 
            indexWArB_test  = np.sort(indexWArB_test, axis=0) 
            indexWBr1_train = indexWBr1_random[:int(len(indexWBr1)*proporcoes[0])]; 
            indexWBr1_train = np.sort(indexWBr1_train, axis=0)
            indexWBr1_test  = indexWBr1_random[int(len(indexWBr1)*proporcoes[0]):]; 
            indexWBr1_test  = np.sort(indexWBr1_test, axis=0)
            indexWBr2_train = indexWBr2_random[:int(len(indexWBr2)*proporcoes[0])]; 
            indexWBr2_train = np.sort(indexWBr2_train, axis=0)
            indexWBr2_test  = indexWBr2_random[int(len(indexWBr2)*proporcoes[0]):]; 
            indexWBr2_test  = np.sort(indexWBr2_test, axis=0)
            
        else:
            
            indexWArA_train = indexWArA_seq[:int(len(indexWArA)*proporcoes[0])]
            indexWArA_test  = indexWArA_seq[int(len(indexWArA)*proporcoes[0]):] 
            indexWArB_train = indexWArB_seq[:int(len(indexWArB)*proporcoes[0])]
            indexWArB_test  = indexWArB_seq[int(len(indexWArB)*proporcoes[0]):] 
            indexWBr1_train = indexWBr1_seq[:int(len(indexWBr1)*proporcoes[0])]
            indexWBr1_test  = indexWBr1_seq[int(len(indexWBr1)*proporcoes[0]):] 
            indexWBr2_train = indexWBr2_seq[:int(len(indexWBr2)*proporcoes[0])]
            indexWBr2_test  = indexWBr2_seq[int(len(indexWBr2)*proporcoes[0]):] 
                
        Xtrain = np.concatenate((
                XwellArA[indexWArA_train],
                XwellArB[indexWArB_train],
                XwellBr1[indexWBr1_train],
                XwellBr2[indexWBr2_train],
                ), axis=0)
        Ytrain = np.concatenate((
                YwellArA[indexWArA_train],
                YwellArB[indexWArB_train],
                YwellBr1[indexWBr1_train],
                YwellBr2[indexWBr2_train],
                ), axis=0)
        Xtest = np.concatenate((
                XwellArA[indexWArA_test],
                XwellArB[indexWArB_test],
                XwellBr1[indexWBr1_test],
                XwellBr2[indexWBr2_test],
                ), axis=0)
        Ytest = np.concatenate((
                YwellArA[indexWArA_test],
                YwellArB[indexWArB_test],
                YwellBr1[indexWBr1_test],
                YwellBr2[indexWBr2_test],
                ), axis=0)
    
    
    if domain == 'ABAxABAA':
        if RANDOM is True:
            
            indexWArA_train = indexWArA_random[:int(len(indexWArA)*proporcoes[0])]; 
            indexWArA_train = np.sort(indexWArA_train, axis=0)
            indexWArA_test  = indexWArA_random[int(len(indexWArA)*proporcoes[0]):]; 
            indexWArA_test  = np.sort(indexWArA_test, axis=0)
            indexWArB_train = indexWArB_random[:int(len(indexWArB)*proporcoes[0])]; 
            indexWArB_train = np.sort(indexWArB_train, axis=0)
            indexWArB_test  = indexWArB_random[int(len(indexWArB)*proporcoes[0]):]; 
            indexWArB_test  = np.sort(indexWArB_test, axis=0)
            indexWBr1_train = indexWBr1_random[:int(len(indexWBr1)*proporcoes[0])]; 
            indexWBr1_train = np.sort(indexWBr1_train, axis=0)
            indexWBr1_test  = indexWBr1_random[int(len(indexWBr1)*proporcoes[0]):]; 
            indexWBr1_test  = np.sort(indexWBr1_test, axis=0)
            indexWBr2_test  = indexWBr2_random[:]
            indexWBr2_test  = np.sort(indexWBr2_test, axis=0)
                

        else:
            
            indexWArA_train = indexWArA_seq[:int(len(indexWArA)*proporcoes[0])]
            indexWArA_test  = indexWArA_seq[int(len(indexWArA)*proporcoes[0]):] 
            indexWArB_train = indexWArB_seq[:int(len(indexWArB)*proporcoes[0])]
            indexWArB_test  = indexWArB_seq[int(len(indexWArB)*proporcoes[0]):] 
            indexWBr1_train = indexWBr1_seq[:int(len(indexWBr1)*proporcoes[0])]
            indexWBr1_test  = indexWBr1_seq[int(len(indexWBr1)*proporcoes[0]):] 
            indexWBr2_test  = indexWBr2_seq[int(len(indexWBr2)*proporcoes[0]):] 
                
        Xtrain = np.concatenate((
                XwellArA[indexWArA_train],
                XwellArB[indexWArB_train],
                XwellBr1[indexWBr1_train],
                ), axis=0)
        Ytrain = np.concatenate((
                YwellArA[indexWArA_train],
                YwellArB[indexWArB_train],
                YwellBr1[indexWBr1_train],
                ), axis=0)
        Xtest = np.concatenate((
                XwellArA[indexWArA_test],
                XwellArB[indexWArB_test],
                XwellBr1[indexWBr1_test],
                XwellBr2[indexWBr2_test],
                ), axis=0)
        Ytest = np.concatenate((
                YwellArA[indexWArA_test],
                YwellArB[indexWArB_test],
                YwellBr1[indexWBr1_test],
                YwellBr2[indexWBr2_test],
                ), axis=0)
         

    if domain == 'ABAxABA':
        if RANDOM is True:
            
            indexWArA_train = indexWArA_random[:int(len(indexWArA)*proporcoes[0])]; 
            indexWArA_train = np.sort(indexWArA_train, axis=0)
            indexWArA_test  = indexWArA_random[int(len(indexWArA)*proporcoes[0]):]; 
            indexWArA_test  = np.sort(indexWArA_test, axis=0) 
            indexWArB_train = indexWArB_random[:int(len(indexWArB)*proporcoes[0])]; 
            indexWArB_train = np.sort(indexWArB_train, axis=0)
            indexWArB_test  = indexWArB_random[int(len(indexWArB)*proporcoes[0]):]; 
            indexWArB_test  = np.sort(indexWArB_test, axis=0) 
            indexWBr1_train = indexWBr1_random[:int(len(indexWBr1)*proporcoes[0])]; 
            indexWBr1_train = np.sort(indexWBr1_train, axis=0)
            indexWBr1_test  = indexWBr1_random[int(len(indexWBr1)*proporcoes[0]):]; 
            indexWBr1_test  = np.sort(indexWBr1_test, axis=0) 
                
        else:
            
            indexWArA_train = indexWArA_seq[:int(len(indexWArA)*proporcoes[0])]
            indexWArA_test  = indexWArA_seq[int(len(indexWArA)*proporcoes[0]):] 
            indexWArB_train = indexWArB_seq[:int(len(indexWArB)*proporcoes[0])]
            indexWArB_test  = indexWArB_seq[int(len(indexWArB)*proporcoes[0]):] 
            indexWBr1_train = indexWBr1_seq[:int(len(indexWBr1)*proporcoes[0])]
            indexWBr1_test  = indexWBr1_seq[int(len(indexWBr1)*proporcoes[0]):] 
                
        Xtrain = np.concatenate((
                XwellArA[indexWArA_train],
                XwellArB[indexWArB_train],
                XwellBr1[indexWBr1_train],
                ), axis=0)
        Ytrain = np.concatenate((
                YwellArA[indexWArA_train],
                YwellArB[indexWArB_train],
                YwellBr1[indexWBr1_train],
                ), axis=0)
        Xtest = np.concatenate((
                XwellArA[indexWArA_test],
                XwellArB[indexWArB_test],
                XwellBr1[indexWBr1_test],
                ), axis=0)
        Ytest = np.concatenate((
                YwellArA[indexWArA_test],
                YwellArB[indexWArB_test],
                YwellBr1[indexWBr1_test],
                ), axis=0)
            
            
    if domain == 'AAxABA':
        if RANDOM is True:
            
            indexWArA_train = indexWArA_random[:int(len(indexWArA)*proporcoes[0])]; 
            indexWArA_train = np.sort(indexWArA_train, axis=0)
            indexWArA_test  = indexWArA_random[int(len(indexWArA)*proporcoes[0]):]; 
            indexWArA_test  = np.sort(indexWArA_test, axis=0) 
            indexWArB_test  = indexWArB_random[:]                                 ; 
            indexWArB_test  = np.sort(indexWArB_test, axis=0)
            indexWBr1_train = indexWBr1_random[:int(len(indexWBr1)*proporcoes[0])]; 
            indexWBr1_train = np.sort(indexWBr1_train, axis=0)
            indexWBr1_test  = indexWBr1_random[int(len(indexWBr1)*proporcoes[0]):]; 
            indexWBr1_test  = np.sort(indexWBr1_test, axis=0) 
                

        else:
            
            indexWArA_train = indexWArA_seq[:int(len(indexWArA)*proporcoes[0])]
            indexWArA_test  = indexWArA_seq[int(len(indexWArA)*proporcoes[0]):] 
            indexWArB_test  = indexWArB_seq[:] 
            indexWBr1_train = indexWBr1_seq[:int(len(indexWBr1)*proporcoes[0])]
            indexWBr1_test  = indexWBr1_seq[int(len(indexWBr1)*proporcoes[0]):] 
                
        Xtrain = np.concatenate((
                XwellArA[indexWArA_train],
                XwellBr1[indexWBr1_train],
                ), axis=0)
        Ytrain = np.concatenate((
                YwellArA[indexWArA_train],
                YwellBr1[indexWBr1_train],
                ), axis=0)
        
        Xtest = np.concatenate((
                XwellArA[indexWArA_test],
                XwellArB[indexWArB_test],
                XwellBr1[indexWBr1_test],
                ), axis=0)
        Ytest = np.concatenate((
                YwellArA[indexWArA_test],
                YwellArB[indexWArB_test],
                YwellBr1[indexWBr1_test],
                ), axis=0)
            


    clusters = [train, test, val]
    

    
    
    x_train = Xtrain
    y_train = Ytrain
    x_test  = Xtest
    y_test  = Ytest
    

    print('TAMANHO FINAL DOS CLUSTERS')
    print('train', x_train.shape)
    print('test ', x_test.shape)
    print('----------------------------------------')

    x_val = []
    y_val = []

    if equalize_data is True:
        
        y_train = y_train*normalizador
        hist, regions = np.histogram(y_train, bins=bins, range=(0,normalizador))

        regions = regions.reshape(-1,1)  
        regions = np.hstack([np.zeros([len(regions), 1]), regions])
        
        for i in range(1, len(regions)):
            regions[i, 0] = regions[i-1, 1]

        regions = np.delete(regions,0, 0)
        regions_count = np.zeros(len(regions))

        for i in range(y_train.shape[0]):
            for j in range(len(regions)):
                if y_train[i, ...] > regions[j, 0] and y_train[i] <= regions[j, 1]:
                    regions_count[j] += 1
                    break
                
                
        max_pop = int(np.max(regions_count))

        y_train_eq = y_train
        x_train_eq = x_train
        for i in range(len(regions_count)):
            index = np.where(
                np.logical_and(y_train >= regions[i, 0], y_train <= regions[i, 1])
            )[0]
            
#            print('i: ', i, regions[i, 0], regions[i, 1])

            diff = max_pop - len(index)
#            print(max_pop)
            np.random.shuffle(index)
            
            while diff > 0 and diff < max_pop:

                
                
                if diff >= len(index):
                    extra_dots = index
                else:
                    extra_dots = index[:diff]

                y_train_eq = np.vstack([y_train_eq, y_train[extra_dots]])
                x_train_eq = np.vstack([x_train_eq, x_train[extra_dots]])


            
                diff -= len(extra_dots)
                
                if len(extra_dots) == 0:
                    diff = -50
                

#            print(diff, len(np.where(np.logical_and(y_train_eq >= regions[i, 0], y_train_eq <= regions[i, 1]))[0]))
              

        regions_count2 = np.zeros(len(regions))
        for i in range(y_train_eq.shape[0]):

            for j in range(len(regions)):
                if (
                    y_train_eq[i, ...] > regions[j, 0]
                    and y_train_eq[i] <= regions[j, 1]
                ):
                    regions_count2[j] += 1
                    break
        print(regions_count2)
        
        y_train = y_train / normalizador
        y_train_eq = y_train_eq / normalizador

        
        
        
        return x_train, x_train_eq, x_val, x_test, y_train, y_train_eq, y_val, y_test, clusters

    x_train_eq = x_train
    y_train_eq = y_train
    
    return x_train, x_train_eq, x_val, x_test, y_train, y_train_eq, y_val, y_test, clusters


def get_batch_sequential(X, Y, i, input_dim, batch_size=32, seq_len=10, normalize=False):
    """ 
        Get a Sequential Batch with fized size
        batch = [batch_size, seq_len, dimension]
        
        X = n_samples x n_sample_dimension => Input data
        Y = n_samples x n_sample_dimension => Target
        
        return: data, target
    """
    

    min_len = min(batch_size, X.shape[0] - 1 - i)

    if min_len < batch_size:
        i = int(np.random.randint(X.shape[0] - 1 - batch_size, size=1))

    data = X[i : i + batch_size, ...]
    target = Y[i : i + batch_size, ...]
    
    if normalize is True:
        data2 = data.reshape(data.shape[0], seq_len, input_dim)
        data_mean =  np.mean(data2, axis=0)
        data_mean2 = np.mean(data_mean, axis=0)
        data_std =  np.mean(data2, axis=0)
        data_std2 = np.mean(data_std, axis=0)
         
        data3 = (data2-data_mean2)/data_std2
        
        data = data3.reshape(batch_size,-1)
        
#        data_std = np.std(data2, axis=)
    
    #    data   = X[i:i + batch_size, :, :]
    #    target = Y[i:i + batch_size, :, :]

    return data, target, min_len






def get_weights_values(W, B, model):
    if len(W) == 0:
        primeira_rodada = True
    else:
        primeira_rodada = False
        count = 0
    for i in range(len(model.layers)):
        if 'Input' in str(model.layers[i]):
            pass
        elif 'Dropout' in str(model.layers[i]):
            pass
        else:
            if primeira_rodada is True:
                W.append(model.layers[i].get_weights()[0].reshape(-1, 1))
#                try:
                B.append(model.layers[i].get_weights()[1].reshape(-1, 1))
#                except IndexError:
#                    pass
                    
            else:
                W[count] = np.hstack([W[count], model.layers[i].get_weights()[0].reshape(-1, 1)])
#                try:
                B[count] = np.hstack([B[count], model.layers[i].get_weights()[1].reshape(-1, 1)])
#                except IndexError:
#                    pass
                count += 1
       
    return W, B


    
    
    
  
def weight_histogram(W, B, bins):
    W_hist = calculate_histogram2D(W, bins)
    B_hist = calculate_histogram2D(B, bins)
    
    return W_hist, B_hist

    

def calculate_histogram2D(Matrix, biny):
    Matrix_hist_H = []
    Matrix_hist_xedges = []
    Matrix_hist_yedges = []
    
    for i in range(len(Matrix)):
        y = Matrix[i]#[:,1:]
        binx = y.shape[1]
#        biny = y.shape[0]//biny+1
        x = np.ones([y.shape[0], binx])
        for j in range(binx):
            x[:,j] = j
        
        y = y.reshape(-1,)
        x = x.reshape(-1,)
        H, xedges, yedges = np.histogram2d(x, y, bins=(binx, biny))
        
        Matrix_hist_H.append(H)
        Matrix_hist_xedges.append(xedges)
        Matrix_hist_yedges.append(yedges)
 
    Matrix_hist = [Matrix_hist_H, Matrix_hist_xedges, Matrix_hist_yedges]  
    
    return Matrix_hist
    
    
    
    
    
    
    
    
    
    
    
    
    