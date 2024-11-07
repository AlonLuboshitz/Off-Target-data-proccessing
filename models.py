


from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from tensorflow import keras
from keras.layers import Reshape, Conv1D, Input, Dense, Flatten, Concatenate, MaxPooling1D, Reshape

## No random state is initialized, np.random.seed(42) is used in the model runner file if needed
def get_cnn(sequence_length, bp_presenation, only_seq_info, if_bp,if_seperate_epi, num_of_additional_features, epigenetic_window_size, epigenetic_number, task = None):
        return create_convolution_model(sequence_length, bp_presenation, only_seq_info, if_bp,if_seperate_epi, num_of_additional_features, epigenetic_window_size, epigenetic_number, task)   
def get_xgboost_cw(scale_pos_weight, random_state, if_data_reproducibility):
    sub_sample = 1
    if if_data_reproducibility:
        sub_sample = 0.5

    return XGBClassifier(random_state=random_state,subsample=sub_sample,scale_pos_weight=scale_pos_weight, objective='binary:logistic',n_jobs=-1) 
def get_xgboost(random_state):
        return XGBClassifier(random_state=random_state, objective='binary:logistic',n_jobs=-1)
def get_logreg(random_state,if_data_reproducibility):
    if if_data_reproducibility:
        return LogisticRegression(random_state=random_state,solver='sag',n_jobs=-1)
    return LogisticRegression(random_state=random_state,n_jobs=-1)
def create_conv_seq_layers(seq_input,sequence_length,bp_presenation):
    seq_input_reshaped = Reshape((sequence_length, bp_presenation)) (seq_input)

    seq_conv_1 = Conv1D(32, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_input_reshaped)
    seq_acti_1 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_1)
    seq_drop_1 = keras.layers.Dropout(0.1)(seq_acti_1)
    
    seq_conv_2 = Conv1D(64, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_drop_1)
    seq_acti_2 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_2)
    seq_max_pooling_1 = MaxPooling1D(pool_size=3, padding="same")(seq_acti_2)

    seq_conv_3 = Conv1D(128, 3, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_max_pooling_1)
    seq_acti_3 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_3)

    seq_conv_4 = Conv1D(256, 2, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_acti_3)
    seq_acti_4 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_4)
    seq_max_pooling_2 = MaxPooling1D(pool_size=3, padding="same")(seq_acti_4)

    seq_conv_5 = Conv1D(512, 2, kernel_initializer='random_uniform', activation=None,strides=1, padding="valid")(seq_max_pooling_2)
    seq_acti_5 = keras.layers.LeakyReLU(alpha=0.2)(seq_conv_5)

    seq_flatten = Flatten()(seq_acti_5) 
    return seq_flatten
def create_conv_epi_layer(epi_input,kernal_size,strides,epigenetic_window_size,epigenetic_number):
    epi_input_reshaped = Reshape((epigenetic_window_size,epigenetic_number))(epi_input)
    epi_conv_6 = Conv1D(2,kernel_size=kernal_size,kernel_initializer='random_uniform',strides=strides,padding='valid')(epi_input_reshaped)
    epi_acti_6 = keras.layers.LeakyReLU(alpha=0.2)(epi_conv_6)
    epi_max_pool_3 = MaxPooling1D(pool_size=2,strides=2, padding='same')(epi_acti_6) 
    epi_seq_flatten = Flatten()(epi_max_pool_3)
    return epi_seq_flatten

def create_convolution_model(sequence_length, bp_presenation,only_seq_info,if_bp,if_seperate_epi,num_of_additional_features,epigenetic_window_size,epigenetic_number, task=None):
    # set seq conv layers
    seq_input = Input(shape=(sequence_length * bp_presenation))
    seq_flatten = create_conv_seq_layers(seq_input=seq_input,sequence_length=sequence_length,bp_presenation=bp_presenation)

    if (only_seq_info or if_bp): # only seq information given
        combined = seq_flatten
    elif if_seperate_epi: # epigenetics in diffrenet conv
        epi_feature = Input(shape=(epigenetic_window_size * epigenetic_number))
        epi_seq_flatten = create_conv_epi_layer(epi_input=epi_feature,kernal_size=(int(epigenetic_window_size/10)),strides=5,epigenetic_window_size=epigenetic_window_size,epigenetic_number=epigenetic_number)
        combined = Concatenate()([seq_flatten, epi_seq_flatten])
        
    else:
        feature_input = Input(shape=(num_of_additional_features))
        combined = Concatenate()([seq_flatten, feature_input])

    seq_dense_1 = Dense(256, activation='relu')(combined)
    seq_drop_2 = keras.layers.Dropout(0.3)(seq_dense_1)
    seq_dense_2 = Dense(128, activation='relu')(seq_drop_2)
    seq_drop_3 = keras.layers.Dropout(0.2)(seq_dense_2)
    seq_dense_3 = Dense(64, activation='relu')(seq_drop_3)
    seq_drop_4 = keras.layers.Dropout(0.2)(seq_dense_3)
    seq_dense_4 = Dense(40, activation='relu')(seq_drop_4)
    seq_drop_5 = keras.layers.Dropout(0.2)(seq_dense_4)
    
    # Set loss and last neuron for the task
    if task == "Classification":
        loss = keras.losses.BinaryCrossentropy()
        metrics = ['binary_accuracy']
        output = Dense(1, activation='sigmoid')(seq_drop_5)
    elif task == "Regression":
        loss = keras.losses.MeanSquaredError()
        metrics = ['mean_absolute_error']
        output = Dense(1, activation='linear')(seq_drop_5)
    else : RuntimeError("Task must be set to 'Classification' or 'Regression'")
    ## Set inputs and outputs sizes for the model to accepet.
    if (only_seq_info or if_bp):
        model = keras.Model(inputs=seq_input, outputs=output)
    elif if_seperate_epi:
        model = keras.Model(inputs=[seq_input,epi_feature], outputs=output)

    else:
        model = keras.Model(inputs=[seq_input, feature_input], outputs=output)
    
    
    model.compile(loss=loss, optimizer= keras.optimizers.Adam(learning_rate=1e-3), metrics=metrics)
    print(model.summary())
    return model



