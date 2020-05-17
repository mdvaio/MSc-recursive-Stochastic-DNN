import numpy as np

import keras
from keras import regularizers, losses
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense, Activation, LSTM, Dropout







def Create_model(seq_len, input_dim, output_dim, dropout_rate, normalize_batch):
#               -----------------------
#               PReLu PReLu PReLu PReLu  tanh  tanh  tanh  tanh
#                   1     2     3     4     5     6     7     8
#    N_neurons = [ 400,  800,  800,  400,  100,    0,    0,    0]     # RAW DATA   dropout (0.05)  -  5 layers
#    N_neurons = [ 200,  200,  200,  200,   0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  4 layers  (AAxABA) 
 
    
    
# Testes de Arquitetura   
    
#    Linear
#    N_neurons = [ 800,  800,  800,  800,   0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  4 layers 
#    N_neurons = [ 400,  400,  400,  400,   0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  4 layers 
#    N_neurons = [ 200,  200,  200,  200,    0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  4 layers  
#    N_neurons = [ 150,  150,  150,  150,    0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  4 layers   
#    N_neurons = [ 100,  100,  100,  100,    0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  4 layers 
#    N_neurons = [  50,   50,   50,   50,    0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  4 layers 
    
#    Não - Linear
#    N_neurons = [ 400,  400,  400,  400,   200,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers 
    N_neurons = [ 200,  200,  200,  200,   0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers 
#    N_neurons = [ 150,  150,  150,  150,    75,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers   
#    N_neurons = [ 100,  100,  100,  100,    50,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers   
#    N_neurons = [  50,   50,   50,   50,    10,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers 
    
    

    
#    Não - Linear
#    N_neurons = [   100,    0,    0,    0,      0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  1 layers 
#    N_neurons = [   10000,    0,    0,    0,     0,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  1 layers 
#    N_neurons = [   1000,    1000,    0,    0,     1000,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  1 layers 

    
#    Não - Linear
#    N_neurons = [ 800,  1200,  1200,  800,   100,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers
#    N_neurons = [ 800,  800,  800,  800,   100,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers
#    N_neurons = [ 800,  800,  800,  800,   100,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers
#    N_neurons = [ 400,  400,  400,  400,   100,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers
    
#    N_neurons = [ 200,  200,  200,  200,   100,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers 
#    N_neurons = [ 150,  150,  150,  150,    75,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers   
#    N_neurons = [ 100,  100,  100,  100,    50,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers   
#    N_neurons = [  50,   50,   50,   50,    10,    0,    0,    0]      # PCA PADRAO dropout (0.3)   -  5 layers 

    
    dense1 = keras.layers.Dense(
        N_neurons[0],
        trainable=True,
#        activation=keras.layers.LeakyReLU(alpha=0.3),
        activation=keras.layers.PReLU(alpha_initializer='he_normal'),
#        activation="tanh",
        kernel_initializer="he_normal",
#        kernel_initializer=keras.initializers.RandomUniform(minval=-2, maxval=2, seed=None),
        bias_initializer="zeros",
    #    kernel_regularizer=regularizers.l2(0.005),
    #    activity_regularizer=regularizers.l1(0.1),
    )
    dropout1 = keras.layers.Dropout(dropout_rate)
    AlphaDropout1 = keras.layers.AlphaDropout(dropout_rate)
    batch_norm1 = keras.layers.BatchNormalization(axis=-1)
    
    
    
    dense2 = keras.layers.Dense(
        N_neurons[1],
#        activation=keras.layers.LeakyReLU(alpha=0.5),
        activation=keras.layers.PReLU(alpha_initializer='he_normal'),
#        activation="tanh",
        kernel_initializer="he_normal",
#        kernel_initializer=keras.initializers.RandomUniform(minval=-3, maxval=3, seed=None),
        bias_initializer="zeros",
    #    kernel_regularizer=regularizers.l2(0.005),
    #    activity_regularizer=regularizers.l1(0.1),
    )
    dropout2 = keras.layers.Dropout(dropout_rate)
    
    dense3 = keras.layers.Dense(
        N_neurons[2],
#        activation="tanh",
        activation=keras.layers.PReLU(alpha_initializer='he_normal'),
        kernel_initializer="he_normal",
#        kernel_initializer=keras.initializers.RandomUniform(minval=-3, maxval=3, seed=None),
        bias_initializer="zeros",
    #    kernel_regularizer=regularizers.l2(0.005),
    #    activity_regularizer=regularizers.l1(0.1),
    )
    dropout3 = keras.layers.Dropout(dropout_rate)
        
    
    dense4 = keras.layers.Dense(
        N_neurons[3],
#        activation="tanh",
        activation=keras.layers.PReLU(alpha_initializer='he_normal'),
        kernel_initializer="he_normal",
#        kernel_initializer=keras.initializers.RandomUniform(minval=-3, maxval=3, seed=None),
        bias_initializer="zeros",
    #    kernel_regularizer=regularizers.l2(0.005),
    #    activity_regularizer=regularizers.l1(0.1),
    )
    dropout4 = keras.layers.Dropout(dropout_rate)
    
    dense5 = keras.layers.Dense(
        N_neurons[4],
        activation="tanh",
        kernel_initializer="he_normal",
#        kernel_initializer=keras.initializers.RandomUniform(minval=-3, maxval=3, seed=None),
        bias_initializer="zeros",
    #    kernel_regularizer=regularizers.l2(0.005),
    #    activity_regularizer=regularizers.l1(0.1),
    )
    dropout5 = keras.layers.Dropout(dropout_rate)
    
    dense6 = keras.layers.Dense(
        N_neurons[5],
        activation="tanh",
        kernel_initializer="he_normal",
#        kernel_initializer=keras.initializers.RandomUniform(minval=-3, maxval=3, seed=None),
        bias_initializer="zeros",
    #    kernel_regularizer=regularizers.l2(0.005),
    #    activity_regularizer=regularizers.l1(0.1),
    )
    dropout6 = keras.layers.Dropout(dropout_rate)
    
    dense6 = keras.layers.Dense(
        N_neurons[6],
        activation="tanh",
        kernel_initializer="he_normal",
#        kernel_initializer=keras.initializers.RandomUniform(minval=-3, maxval=3, seed=None),
        bias_initializer="zeros",
    #    kernel_regularizer=regularizers.l2(0.005),
    #    activity_regularizer=regularizers.l1(0.1),
    )
    dropout6 = keras.layers.Dropout(dropout_rate)
    
    dense7 = keras.layers.Dense(
        N_neurons[7],
        activation="tanh",
        kernel_initializer="he_normal",
#        kernel_initializer=keras.initializers.RandomUniform(minval=-3, maxval=3, seed=None),
        bias_initializer="zeros",
    #    kernel_regularizer=regularizers.l2(0.005),
    #    activity_regularizer=regularizers.l1(0.1),
    )
    dropout7 = keras.layers.Dropout(dropout_rate)
    
  
    
##    dense8 = keras.layers.Dense(output_dim, activation="linear")
    dense8 = keras.layers.Dense(output_dim, activation="sigmoid", use_bias=True)
#    dense8 = keras.layers.Dense(output_dim, activation=keras.layers.PReLU(alpha_initializer='he_normal'), use_bias=True)



   
    inputs = keras.Input(shape=(seq_len*input_dim,))
    if N_neurons[0] != 0:
        x = dense1(inputs)
        x = dropout1(x, training=True)
#        x = AlphaDropout1(x, training=True)
#        x = batch_norm1(x)  
        print('1')
    if N_neurons[1] != 0:
        x = dense2(x)
        x = dropout2(x, training=True)
        print('2')
    if N_neurons[2] != 0:
        x = dense3(x)
        x = dropout3(x, training=True)
        
        print('3')
    if N_neurons[3] != 0:
        x = dense4(x)
        x = dropout4(x, training=True)
        
        print('4')
    if N_neurons[4] != 0:
        x = dense5(x)
        x = dropout5(x, training=True)
        
        print('5')
    if N_neurons[5] != 0:
        x = dense6(x)
        x = dropout6(x, training=True)
        
        print('6')
    if N_neurons[6] != 0:
        x = dense7(x)
        x = dropout7(x, training=True)
        
        print('7')
    
    outputs = dense8(x)
    
    

    
    model = keras.Model(inputs, outputs)
    model.summary()
    print('Input Dim: ', input_dim, 'Output Dim:' , output_dim)
    print('_________________________________________________________________')
    
    
#    lamb_norm = 1
#    b = batch_size
#    B = 70000*0.2//32
#    T = EPOCH
    lr = 0.0001
    lamb = 0  #0.001
#    
#    decay = lamb_norm*sqrt(B/(B*T)) 
    
    
    model.compile(
        #              optimizer=optimizers.SGD(lr=0.01,
        #                                       momentum=0.0,
        #                                       decay=0.0,
        #                                       nesterov=False),
        optimizer=keras.optimizers.Adam(
            lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lamb, amsgrad=False
        ),
        #              optimizer=keras.optimizers.Nadam(lr=0.002,
        #                                     beta_1=0.9,
        #                                     beta_2=0.999,
        #                                     epsilon=None,
        #                                     schedule_decay=0.004),
        loss=losses.mean_squared_error,
        metrics=["mean_absolute_percentage_error"],
    )
    
    return model
