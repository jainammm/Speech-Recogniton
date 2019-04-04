from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Dense, Input, 
    TimeDistributed, Activation, GRU)

def model(input_dim, units, recur_layers=7, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn_1 = GRU(units, activation='sigmoid',
        return_sequences=True, implementation=2, name='rnn_1')(input_data)
    
    # Here, we'll create rnn layers for number asked by the user (recur_layers)
    
    rnn_input_for_next_rnn = rnn_1
    for layer in range(recur_layers - 1):
        layer_name = "rnn_" +  str(layer+2)
        rnn = GRU(units, activation="sigmoid",return_sequences=True, 
                 implementation=2, name=layer_name) (rnn_input_for_next_rnn)
        
        batchnorm_name = "bn_" + str(layer)
        rnn_out = BatchNormalization(name=batchnorm_name)(rnn)
        rnn_input_for_next_rnn = rnn_out
        
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_out)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

