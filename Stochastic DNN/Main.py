import numpy as np

import keras
from keras import regularizers, losses
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense, Activation, LSTM, Dropout

import tensorflow as tf
import tensorflow_probability as tfp


import E_TS_Data_Handler as dh
import E_TS_NN_plotter as plotter
import E_TS_Model as md
import pandas as pd
from copy import deepcopy

#import time
#                    start = time.time()      
#                    end = time.time()     
#                    print(end-start)     

num_iterations = 500


batch_size = 2048
seq_len = 6
hist1D_bins = 10
hist2D_bins = 100



avanco = 0
percentage_train = 1
test_after = 10
normalizador = 3.6

dropout_rate = 0.03

normalize_batch = False


proportion_train_test_val = [0.75, 0.25, 0]
prop_train_test_val = proportion_train_test_val


#save_figs = True
save_figs = False
#show_figs = True0
show_figs = True 



DATA_SOURCE = ['PCA Padrao']#'RAW Data', 'PCA Padrao', 'PCA Geral', 

'''
Options both by Random pick or Sequential pick
Poco: A   A  BR1 BR2
ROCHA A + B + A + A		VS		ROCHA A + B + A + A		ABAA VS ABAA	ABAAxABAA
ROCHA A + B + A			VS		ROCHA A + B + A + A		ABA  VS ABAA	ABAxABAA	
ROCHA A + B + A			VS		ROCHA A + B + A			ABA  VS ABA		ABAxABA
ROCHA A + A				VS		ROCHA A + B + A		    AA   VS ABA		AAxABA
'''
DOMAIN = ['AAxABA'] #    'AAxABA', 'ABAxABA', 'ABAxABAA', 'ABAAxABAA'
RANDOM = True                   # Random vs Sequential


DATA_EQUALIZATION = [True]#, False]
EPOCHS     = [500]




USE_SOURCE = False



if USE_SOURCE is True:
    DATA_SOURCE = ['PCA Padrao']
    DOMAIN =  ['Just Rock A']
    DATA_EQUALIZATION = [True]
    EPOCHS = [5000]
    epoch_in_focus = 3450
#    df_MODELS = EPOCHs_5000_DS_PCAPadrao__TDRockAwithB1andB2__DE_True__P_TTV_0750250
    
    
else:
    df_MODELS  = pd.DataFrame(index = [
                                     'metrics',
                                     'seq_len',
                                     'nome_rodada',
                                     'nome_title',
                                     'W_hist',
                                     'B_hist',
                                     'clusters',
                                     'models',
                                     'models_spoch',
                                     'best_model',
                                     'best_epoch',
                                     ])

kkkk = 1
#with tf.device('/cpu:0'):
#with tf.device('/gpu:0'):
if kkkk == 1:
    for ll in range(len(EPOCHS)):
        print(ll)
        epochs = EPOCHS[ll]
        for jj in range(len(DOMAIN)):
            domain = DOMAIN[jj]
            for ii in range(len(DATA_SOURCE)):
                data_source = DATA_SOURCE[ii]
                for kk in range(len(DATA_EQUALIZATION)):
                    data_equalization = DATA_EQUALIZATION[kk]
                    
#                    nome_rodada = (
#                            'EPOCHs_' +str(epochs)
#                            +'_DS_' + data_source 
#                            + '__TD ' + domain 
#                            + '__DE_' + str(data_equalization) 
#                            + '__P_TTV_' + str(prop_train_test_val)
#                    )
                    
                    nome_rodada = (data_source + '_' + domain)
                    
                    nome_title =  (
                            ' --EPOCHs '+str(epochs)
                            +'-- Data Souce ' + data_source
                            + ' -- Train domain ' + domain
                            + ' -- Data Equalization ' + str(data_equalization)
                            + ' -- Proportion TestTrainVal ' + str(prop_train_test_val)
                    )
                    
                    dataX, dataY, comp = dh.getDataFromTXT(
                            seq_len=seq_len, 
                            data_source=data_source, 
                            normalize=False,
                            normalizador=normalizador
                    )
                    
                    input_dim = dataX.shape[1]
                    output_dim = dataY.shape[1]
                    
                    x_train, x_train_eq, x_val, x_test, y_train, y_train_eq, y_val, y_test, clusters = dh.slicer_for_sequential_data(
                        dataX=dataX,
                        dataY=dataY,
                        comp=comp,
                        normalizador=normalizador,
                        seq_len=seq_len,
                        avanco=avanco,
                        bins = hist1D_bins,
                        domain=domain,
                        RANDOM=RANDOM,
                        equalize_data=data_equalization,
                    )
                    
                    
                    
                    if USE_SOURCE is True:
                        print('Using NN Model from SOURCE')
                        epochs = epoch_in_focus
                        
#                        models = df_MODELS.loc[['models'], [nome_rodada]].as_matrix()[0,0]
#                        models_epochs = df_MODELS.loc[['models_spoch'], [nome_rodada]].as_matrix()[0,0]
                        models = df_MODELS.loc['models']
                        models_epochs = df_MODELS.loc['models_spoch']
                        chosen_index = models_epochs.index(epoch_in_focus)
                        
                        model = models[1] 
                        best_model = models[chosen_index]
                        best_epoch = models_epochs[chosen_index]
                        
                        print('_______________________________________________________')
                        print('Iniciando: ')
                        print(nome_title)
                        print('_______________________________________________________')
                        
                    if USE_SOURCE is False:
                        print('Training NN Model')
                        
                        model = md.Create_model(seq_len, input_dim, output_dim, dropout_rate, normalize_batch)
                        best_model = model
                        models = []
                        models_epochs = []
                    
                        global_history_train = [0, 0]
                        global_metrics_train = [0, 0]
                        global_metrics_test = [0, 0]
    
    
                        W = []
                        B = []
    
                        lowest_loss = 100
                        metrics_train = np.zeros([1,2])
                        epoch = 0
                        count = 0
                        while epoch < epochs:
                            try:
                                test_after = 100   
                                if epoch >= epochs*0.5:
                                    test_after = 50 
                                if epoch >= epochs*0.8:
                                    test_after = 25
                                if epoch >= epochs*0.95-1:
                                    test_after = 10
                                
                                
#                                print('epoch: ', epoch, 'test_after: ', test_after)
                                
                                index = np.arange(x_train_eq.shape[0])
                                np.random.shuffle(index)
                                index = index[: int(len(index) * percentage_train)]
                                
                                i=0
                                
#                                batch_size = index.shape[0]
#                                index = [0, 0]
                                metrics_train = [0, 0]
                                
                                while i < index.shape[0]:
                                    
                                    batchX, batchY, minLen = dh.get_batch_sequential(
                                        x_train_eq, y_train_eq, i=index[i], batch_size=batch_size, seq_len=seq_len, 
                                        input_dim=input_dim,
                                        normalize=normalize_batch,
                                    )
                            
                                    metrics_train = np.vstack(
                                        [metrics_train, [model.train_on_batch(batchX, batchY)]]
                                    )
                                    i += batch_size
                                metrics_train = np.delete(metrics_train, 0, axis=0)
                                metrics_train = np.mean(metrics_train, axis=0) 
                                
                                W, B = dh.get_weights_values(W, B, model)
                                
                                test_after = 2
                                if epoch % test_after == 0:                           
                                    metrics_test = [0, 0]
                                    i = 0
                                    while i < x_test.shape[0]:
                                        
                                        batchX, batchY, minLen = dh.get_batch_sequential(
                                            x_test, y_test, i=i, batch_size=batch_size, seq_len=seq_len,input_dim=input_dim,
                                        normalize=normalize_batch,
                                        )
                            
                                        metrics_test = np.vstack(
                                            [metrics_test, [model.test_on_batch(batchX, batchY)]]
                                        )
                            
                                        i += batch_size
                                    metrics_test = np.delete(metrics_test, 0, axis=0)
                                    metrics_test = np.mean(metrics_test, axis=0)
                                    models_epochs = models_epochs + [epoch]
#                                    
#                                    count += 1
#                                    if count >= 10:
#                                    current_model = deepcopy(model)
                                    
                                    current_model = model
                                    models.append(current_model)
#                                        count = 0
                                        
                                    if metrics_test[0] < lowest_loss:
                                        count = 0
#                                        if epoch > 90:
#                                            current_model = deepcopy(model)
#                                        else:
#                                            current_model = model
                                        current_model = model
                                            
                                        lowest_loss = metrics_test[0]
                                        
                                        best_model = current_model
                                        best_epoch = epoch
                                        
#                                        models.append(current_model)
#                                        models_epochs = models_epochs + [epoch]
                                        
                                        print("Best Model updated, ", len(models), ' saved')
#                                    print('Updating Test Loss - Last Best Epoch: ', best_epoch, 'Best Loss: ', lowest_loss)
                                     
                                global_metrics_train = np.vstack(
                                    [global_metrics_train, [metrics_train]]
                                )
                                global_metrics_test = np.vstack(
                                    [global_metrics_test, [metrics_test]]
                                )
                                
                                
                            
                                loss_train = metrics_train[0]
                                metric_train = metrics_train[1] / 100.0
                                loss_test = metrics_test[0]
                                metric_test = metrics_test[1] / 100.0
                                
                                metrics_train = [0, 0]
                                
#                                print(
#                                        "Epoch:",
#                                        epoch,
#                                        "loss_train:{0:.5f}".format(loss_train, 5),
#                                #        "met_train:{0:.2f}".format(metric_train, 1),
#                                        "loss_test:{0:.5f}".format(loss_test, 2),
#                                #        "met_test:{0:.2f}".format(metric_test, 1),
#                                      )
                                
                                
                                if epoch >= 25:
                                    
                                    print(
                                        "Epoch:",
                                        epoch,
                                        "loss_train:{0:.5f}".format(loss_train, 5),
                                #        "met_train:{0:.2f}".format(metric_train, 1),
                                        "loss_test:{0:.5f}".format(loss_test, 2),
                                #        "met_test:{0:.2f}".format(metric_test, 1),
                                        "Diff:{0:.5f}".format(np.mean(global_metrics_test[-19:-8, 0]) - np.mean(global_metrics_test[-11:-1, 0]))
                                      )


                                    
                                    if (np.mean(global_metrics_test[-19:-8, 0]) - np.mean(global_metrics_test[-11:-1, 0])) < 0:
                                        count += 1
                                        
                                        if count >= 7:
                                            print('count = ', count)
                                            print('Moedelo parou de melhorar! Treinamento interrompido!!!!!!!!!!!')
                                            if global_metrics_test[-1, 0] < 0.1 and global_metrics_train[-1, 0] < 0.1:
                                                break
                                    else:
                                        count = 0
                                else:
                                    print(
                                        "Epoch:",
                                        epoch,
                                        "loss_train:{0:.5f}".format(loss_train, 5),
                                #        "met_train:{0:.2f}".format(metric_train, 1),
                                        "loss_test:{0:.5f}".format(loss_test, 2),
                                #        "met_test:{0:.2f}".format(metric_test, 1),
                                      )
                                        
                                epoch += 1
                            except:
                                break
                                epoch = 0
                    
                    
                    
                    
                    
                    ###############################################################################
                    # --------------------------------  Stochastic --------------------------------
                    ###############################################################################
                    
                    
                    #   ------------------------- Conjunto de treino:

                    print('Prediction STC Train')
                    y_train_pred_stc = np.zeros([x_train.shape[0], num_iterations])
                
                    for i in range(num_iterations):
                        y_train_pred_stc[:,i] = best_model.predict(x_train, batch_size=batch_size).reshape(-1,)
                        
                        if i % (num_iterations//5) == 0:
                            avancoStc = i/num_iterations * 100
                            print(avancoStc, '%')
                        if i >= (num_iterations-1):
                            print('100 %') 

                    y_train_pred_stc = y_train_pred_stc*normalizador
                    
                    # PERCENTILE
                    print('Estatistics Train')
                    
                    X_train = np.zeros([len
                                        (y_train_pred_stc), 6])
                    y_train_pred_stc = np.sort(y_train_pred_stc, axis=1)
                    
                    for i in range(len(X_train)):
                            
                        X_train[i, 0] = np.mean(y_train_pred_stc[i, :])
                        X_train[i, 5] = np.std(y_train_pred_stc[i, :])
                        X_train[i, 1] = X_train[i, 0] + 2 * X_train[i, 5]
                        X_train[i, 2] = X_train[i, 0] - 2 * X_train[i, 5]
                        X_train[i, 3] = y_train_pred_stc[i, num_iterations // 33]
                        X_train[i, 4] = y_train_pred_stc[i, -1 - num_iterations // 33]
          
                                         
                    
                    #   ------------------------- Conjunto de teste:
                    print('Prediction STC Test')
                    y_test_pred_stc = np.zeros([x_test.shape[0], num_iterations])
                
                    for i in range(num_iterations):
                        y_test_pred_stc[:,i] = best_model.predict(x_test, batch_size=batch_size).reshape(-1,)
                        
                        if i % (num_iterations//5) == 0:
                            avancoStc = i/num_iterations * 100
                            print(avancoStc, '%')
                        if i >= (num_iterations-1):
                            print('100 %') 
                    
                    y_test_pred_stc = y_test_pred_stc*normalizador
                    # PERCENTILE
                    print('Estatistics Test')
                    
                    X_test = np.zeros([len(y_test_pred_stc), 6])
                    y_test_pred_stc = np.sort(y_test_pred_stc, axis=1)
                    
                    for i in range(X_test.shape[0]):       
#                        
                        X_test[i, 0] = np.mean(y_test_pred_stc[i, :])
                        X_test[i, 5] = np.std(y_test_pred_stc[i, :])
                        X_test[i, 1] = X_test[i, 0] + 2 * X_test[i, 5]
                        X_test[i, 2] = X_test[i, 0] - 2 * X_test[i, 5]
                        X_test[i, 3] = y_test_pred_stc[i, num_iterations // 33]
                        X_test[i, 4] = y_test_pred_stc[i, -1 - num_iterations // 33]
                    
                    
                    
                    if USE_SOURCE is True:
#                        metrics = df_MODELS.loc[['metrics'], [nome_rodada]].as_matrix()[0,0]
#                        W_hist = df_MODELS.loc[['W_hist'], [nome_rodada]].as_matrix()[0,0]
#                        B_hist = df_MODELS.loc[['B_hist'], [nome_rodada]].as_matrix()[0,0]
#                        clusters = df_MODELS.loc[['clusters'], [nome_rodada]].as_matrix()[0,0]
                        
                        metrics = df_MODELS.loc['metrics']
                        W_hist = df_MODELS.loc['W_hist']
                        B_hist = df_MODELS.loc['B_hist']
                        clusters = df_MODELS.loc['clusters']
                        
                        best_epoch = epoch_in_focus
                        
                    else:
                        metrics = np.vstack([global_metrics_train[:,0], global_metrics_test[:,0]]).T
                        W_hist, B_hist = dh.weight_histogram(W, B, bins=hist2D_bins)
                    
                
                    
                    import NN_plotter as plotter
                    grid1 = plotter.generate_figures(
                                metrics=metrics,
                                best_epoch = best_epoch,
                                y_train=y_train*normalizador,
                                y_test=y_test*normalizador,
                                X_train=X_train,
                                X_test=X_test,
                                seq_len=seq_len,
                                nome_rodada=nome_rodada,
                                nome_title=nome_title,
                                save_figs=save_figs,
                                show_figs=show_figs
                            )
                   
             
                    grid2 = plotter.plot_histogram(X_train, y_train, y_train_eq, 
                                           X_test, y_test, 
                                           bins=hist1D_bins, 
                                           normalizador=normalizador,
                                           nome_rodada=nome_rodada,
                                           save_figs=save_figs, 
                                           show_figs=show_figs)
                    
                    plotter. plot_layoult(title=nome_rodada, grid1=grid1, grid2=grid2, save_figs=save_figs, show_figs=True)
                    
                    
                    
                    print("lowest_loss:", lowest_loss, 'mean std:', np.mean(X_test[:,5])*normalizador)
                    
#                    plotter.plot_histogram2D(W_hist, B_hist, nome_rodada=nome_rodada, nome_title=nome_title, save_figs=save_figs, show_figs=show_figs)
#                    plotter.plot_clusters(dataY*normalizador, clusters, save_figs=save_figs, show_figs=show_figs, nome_rodada=nome_rodada, nome_title=nome_title)
                    

                 
#                    if USE_SOURCE is False:
#                        print('Salvando variaveis')
##                        historical_values = [
##                             metrics,
##                             seq_len,
##                             nome_rodada,
##                             nome_title,
##                             W_hist,
##                             B_hist,
##                             clusters,
##                             models, 
##                             models_epochs, 
##                             best_model, 
##                             best_epoch,     
##                             ]
#                        df_MODELS[nome_rodada] = [
#                             metrics,
#                             seq_len,
#                             nome_rodada,
#                             nome_title,
#                             W_hist,
#                             B_hist,
#                             clusters,
#                             models, 
#                             models_epochs, 
#                             best_model, 
#                             best_epoch,     
#                             ]
#                        
#                        df_MODELS.to_pickle(nome_rodada + '.pkl')
#                        
#                        print('Deletando variaveis')
#                        df_MODELS=df_MODELS.drop([nome_rodada], axis=1)
#                        del metrics
#                        del W_hist
#                        del B_hist
#                        del clusters
#                        del models
#                        del models_epochs
#                        del best_model
#                        del best_epoch
#                        del global_metrics_test
#                        del global_metrics_train
#                        del y_test_pred_stc
#                        del y_train_pred_stc
                        
                        
                    
                    
