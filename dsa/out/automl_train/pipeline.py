from tensorflow.train import cosine_decay, AdamOptimizer
from tensorflow.contrib.opt import AdamWOptimizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, CuDNNLSTM, GRU, CuDNNGRU, concatenate, Dense, BatchNormalization, Dropout, AlphaDropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
import csv
import sys
import warnings
from datetime import datetime
from math import floor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


def build_model(encoders):
    """Builds and compiles the model from scratch.

    # Arguments
        encoders: dict of encoders (used to set size of text/categorical inputs)

    # Returns
        model: A compiled model which can be used to train or predict.
    """

    # PassengerId
    input_passengerid = Input(shape=(1,), name="input_passengerid")

    # Pclass
    input_pclass_size = len(encoders['pclass_encoder'].classes_)
    input_pclass = Input(shape=(
        input_pclass_size if input_pclass_size != 2 else 1,), name="input_pclass")

    # Sex
    input_sex_size = len(encoders['sex_encoder'].classes_)
    input_sex = Input(
        shape=(input_sex_size if input_sex_size != 2 else 1,), name="input_sex")

    # Age
    input_age = Input(shape=(1,), name="input_age")

    # SibSp
    input_sibsp_size = len(encoders['sibsp_encoder'].classes_)
    input_sibsp = Input(
        shape=(input_sibsp_size if input_sibsp_size != 2 else 1,), name="input_sibsp")

    # Parch
    input_parch_size = len(encoders['parch_encoder'].classes_)
    input_parch = Input(
        shape=(input_parch_size if input_parch_size != 2 else 1,), name="input_parch")

    # Fare
    input_fare = Input(shape=(1,), name="input_fare")

    # Cabin
    input_cabin_size = len(encoders['cabin_encoder'].classes_)
    input_cabin = Input(
        shape=(input_cabin_size if input_cabin_size != 2 else 1,), name="input_cabin")

    # Embarked
    input_embarked_size = len(encoders['embarked_encoder'].classes_)
    input_embarked = Input(shape=(
        input_embarked_size if input_embarked_size != 2 else 1,), name="input_embarked")

    # Combine all the inputs into a single layer
    concat = concatenate([
        input_passengerid,
        input_pclass,
        input_sex,
        input_age,
        input_sibsp,
        input_parch,
        input_fare,
        input_cabin,
        input_embarked
    ], name="concat")

    # Multilayer Perceptron (MLP) to find interactions between all inputs
    hidden = Dense(128, activation='selu', name='hidden_1',
                   kernel_regularizer=l2(1e-2))(concat)
    hidden = AlphaDropout(0.0, name="dropout_1")(hidden)

    for i in range(4-1):
        hidden = Dense(64, activation="selu", name="hidden_{}".format(
            i+2), kernel_regularizer=l2(1e-2))(hidden)
        hidden = AlphaDropout(0.0, name="dropout_{}".format(i+2))(hidden)

    output = Dense(1, activation="sigmoid", name="output",
                   kernel_regularizer=None)(hidden)

    # Build and compile the model.
    model = Model(inputs=[
        input_passengerid,
        input_pclass,
        input_sex,
        input_age,
        input_sibsp,
        input_parch,
        input_fare,
        input_cabin,
        input_embarked
    ],
        outputs=[output])
    model.compile(loss="binary_crossentropy",
                  optimizer=AdamWOptimizer(learning_rate=0.001,
                                           weight_decay=0.05))

    return model


def build_encoders(df):
    """Builds encoders for fields to be used when
    processing data for the model.

    All encoder specifications are stored in locally
    in /encoders as .json files.

    # Arguments
        df: A pandas DataFrame containing the data.
    """

    # PassengerId
    passengerid_enc = df['PassengerId']
    passengerid_encoder = StandardScaler()
    passengerid_encoder_attrs = ['mean_', 'var_', 'scale_']
    passengerid_encoder.fit(df['PassengerId'].values.reshape(-1, 1))

    passengerid_encoder_dict = {attr: getattr(passengerid_encoder, attr).tolist()
                                for attr in passengerid_encoder_attrs}

    with open(os.path.join('encoders', 'passengerid_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(passengerid_encoder_dict, outfile, ensure_ascii=False)

    # Pclass
    pclass_counts = df['Pclass'].value_counts()
    pclass_perc = max(floor(0.5 * pclass_counts.size), 1)
    pclass_top = np.array(pclass_counts.index[0:pclass_perc], dtype=object)
    pclass_encoder = LabelBinarizer()
    pclass_encoder.fit(pclass_top)

    with open(os.path.join('encoders', 'pclass_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(pclass_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Sex
    sex_counts = df['Sex'].value_counts()
    sex_perc = max(floor(0.5 * sex_counts.size), 1)
    sex_top = np.array(sex_counts.index[0:sex_perc], dtype=object)
    sex_encoder = LabelBinarizer()
    sex_encoder.fit(sex_top)

    with open(os.path.join('encoders', 'sex_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(sex_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Age
    age_enc = df['Age']
    age_encoder = StandardScaler()
    age_encoder_attrs = ['mean_', 'var_', 'scale_']
    age_encoder.fit(df['Age'].values.reshape(-1, 1))

    age_encoder_dict = {attr: getattr(age_encoder, attr).tolist()
                        for attr in age_encoder_attrs}

    with open(os.path.join('encoders', 'age_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(age_encoder_dict, outfile, ensure_ascii=False)

    # SibSp
    sibsp_counts = df['SibSp'].value_counts()
    sibsp_perc = max(floor(0.5 * sibsp_counts.size), 1)
    sibsp_top = np.array(sibsp_counts.index[0:sibsp_perc], dtype=object)
    sibsp_encoder = LabelBinarizer()
    sibsp_encoder.fit(sibsp_top)

    with open(os.path.join('encoders', 'sibsp_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(sibsp_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Parch
    parch_counts = df['Parch'].value_counts()
    parch_perc = max(floor(0.5 * parch_counts.size), 1)
    parch_top = np.array(parch_counts.index[0:parch_perc], dtype=object)
    parch_encoder = LabelBinarizer()
    parch_encoder.fit(parch_top)

    with open(os.path.join('encoders', 'parch_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(parch_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Fare
    fare_enc = df['Fare']
    fare_encoder = StandardScaler()
    fare_encoder_attrs = ['mean_', 'var_', 'scale_']
    fare_encoder.fit(df['Fare'].values.reshape(-1, 1))

    fare_encoder_dict = {attr: getattr(fare_encoder, attr).tolist()
                         for attr in fare_encoder_attrs}

    with open(os.path.join('encoders', 'fare_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(fare_encoder_dict, outfile, ensure_ascii=False)

    # Cabin
    cabin_counts = df['Cabin'].value_counts()
    cabin_perc = max(floor(0.5 * cabin_counts.size), 1)
    cabin_top = np.array(cabin_counts.index[0:cabin_perc], dtype=object)
    cabin_encoder = LabelBinarizer()
    cabin_encoder.fit(cabin_top)

    with open(os.path.join('encoders', 'cabin_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(cabin_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Embarked
    embarked_counts = df['Embarked'].value_counts()
    embarked_perc = max(floor(0.5 * embarked_counts.size), 1)
    embarked_top = np.array(
        embarked_counts.index[0:embarked_perc], dtype=object)
    embarked_encoder = LabelBinarizer()
    embarked_encoder.fit(embarked_top)

    with open(os.path.join('encoders', 'embarked_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(embarked_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Target Field: Survived
    survived_encoder = LabelEncoder()
    survived_encoder.fit(df['Survived'].values)

    with open(os.path.join('encoders', 'survived_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(survived_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)


def load_encoders():
    """Loads the encoders built during `build_encoders`.

    # Returns
        encoders: A dict of encoder objects/specs.
    """

    encoders = {}

    # PassengerId
    passengerid_encoder = StandardScaler()
    passengerid_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'passengerid_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        passengerid_attrs = json.load(infile)

    for attr, value in passengerid_attrs.items():
        setattr(passengerid_encoder, attr, value)
    encoders['passengerid_encoder'] = passengerid_encoder

    # Pclass
    pclass_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'pclass_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        pclass_encoder.classes_ = json.load(infile)
    encoders['pclass_encoder'] = pclass_encoder

    # Sex
    sex_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'sex_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        sex_encoder.classes_ = json.load(infile)
    encoders['sex_encoder'] = sex_encoder

    # Age
    age_encoder = StandardScaler()
    age_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'age_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        age_attrs = json.load(infile)

    for attr, value in age_attrs.items():
        setattr(age_encoder, attr, value)
    encoders['age_encoder'] = age_encoder

    # SibSp
    sibsp_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'sibsp_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        sibsp_encoder.classes_ = json.load(infile)
    encoders['sibsp_encoder'] = sibsp_encoder

    # Parch
    parch_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'parch_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        parch_encoder.classes_ = json.load(infile)
    encoders['parch_encoder'] = parch_encoder

    # Fare
    fare_encoder = StandardScaler()
    fare_encoder_attrs = ['mean_', 'var_', 'scale_']

    with open(os.path.join('encoders', 'fare_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        fare_attrs = json.load(infile)

    for attr, value in fare_attrs.items():
        setattr(fare_encoder, attr, value)
    encoders['fare_encoder'] = fare_encoder

    # Cabin
    cabin_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'cabin_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        cabin_encoder.classes_ = json.load(infile)
    encoders['cabin_encoder'] = cabin_encoder

    # Embarked
    embarked_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'embarked_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        embarked_encoder.classes_ = json.load(infile)
    encoders['embarked_encoder'] = embarked_encoder

    # Target Field: Survived
    survived_encoder = LabelEncoder()

    with open(os.path.join('encoders', 'survived_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        survived_encoder.classes_ = np.array(json.load(infile))
    encoders['survived_encoder'] = survived_encoder

    return encoders


def process_data(df, encoders, process_target=True):
    """Processes an input DataFrame into a format
    sutable for model prediction.

    This function loads the encoder specifications created in
    `build_encoders`.

    # Arguments
        df: a DataFrame containing the source data
        encoders: a dict of encoders to process the data.
        process_target: boolean to determine if the target should be encoded.

    # Returns
        A tuple: A list containing all the processed fields to be fed
        into the model, and the processed target field.
    """

    # PassengerId
    passengerid_enc = df['PassengerId'].values.reshape(-1, 1)
    passengerid_enc = encoders['passengerid_encoder'].transform(
        passengerid_enc)

    # Pclass
    pclass_enc = df['Pclass'].values
    pclass_enc = encoders['pclass_encoder'].transform(pclass_enc)

    # Sex
    sex_enc = df['Sex'].values
    sex_enc = encoders['sex_encoder'].transform(sex_enc)

    # Age
    age_enc = df['Age'].values.reshape(-1, 1)
    age_enc = encoders['age_encoder'].transform(age_enc)

    # SibSp
    sibsp_enc = df['SibSp'].values
    sibsp_enc = encoders['sibsp_encoder'].transform(sibsp_enc)

    # Parch
    parch_enc = df['Parch'].values
    parch_enc = encoders['parch_encoder'].transform(parch_enc)

    # Fare
    fare_enc = df['Fare'].values.reshape(-1, 1)
    fare_enc = encoders['fare_encoder'].transform(fare_enc)

    # Cabin
    cabin_enc = df['Cabin'].values
    cabin_enc = encoders['cabin_encoder'].transform(cabin_enc)

    # Embarked
    embarked_enc = df['Embarked'].values
    embarked_enc = encoders['embarked_encoder'].transform(embarked_enc)

    data_enc = [passengerid_enc,
                pclass_enc,
                sex_enc,
                age_enc,
                sibsp_enc,
                parch_enc,
                fare_enc,
                cabin_enc,
                embarked_enc
                ]

    if process_target:
        # Target Field: Survived
        survived_enc = df['Survived'].values

        survived_enc = encoders['survived_encoder'].transform(survived_enc)

        return (data_enc, survived_enc)

    return data_enc


def model_predict(df, model, encoders):
    """Generates predictions for a trained model.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.
        encoders: a dict of encoders to process the data.

    # Returns
        A numpy array of predictions.
    """

    data_enc = process_data(df, encoders, process_target=False)

    headers = ['probability']
    predictions = pd.DataFrame(model.predict(data_enc), columns=headers)

    return predictions


def model_train(df, encoders, args, model=None):
    """Trains a model, and saves the data locally.

    # Arguments
        df: A pandas DataFrame containing the source data.
        encoders: a dict of encoders to process the data.
        args: a dict of arguments passed through the command line
        model: A compiled model (for TensorFlow, None otherwise).
    """
    X, y = process_data(df, encoders)

    split = StratifiedShuffleSplit(
        n_splits=1, train_size=args.split, test_size=None, random_state=123)

    for train_indices, val_indices in split.split(np.zeros(y.shape[0]), y):
        X_train = [field[train_indices, ] for field in X]
        X_val = [field[val_indices, ] for field in X]
        y_train = y[train_indices, ]
        y_val = y[val_indices, ]

    meta = meta_callback(args, X_val, y_val)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs,
              callbacks=[meta],
              batch_size=64)


class meta_callback(Callback):
    """Keras Callback used during model training to save current weights
    and metrics after each training epoch.

    Metrics metadata is saved in the /metadata folder.
    """

    def __init__(self, args, X_val, y_val):
        self.f = open(os.path.join('metadata', 'results.csv'), 'w')
        self.w = csv.writer(self.f)
        self.w.writerow(['epoch', 'time_completed'] + ['log_loss',
                                                       'accuracy', 'auc', 'precision', 'recall', 'f1'])
        self.in_automl = args.context == 'automl-gs'
        self.X_val = X_val
        self.y_val = y_val

    def on_train_end(self, logs={}):
        self.f.close()
        self.model.save_weights('model_weights.hdf5')

    def on_epoch_end(self, epoch, logs={}):
        y_true = self.y_val
        y_pred = self.model.predict(self.X_val)

        y_pred_label = np.round(y_pred)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logloss = log_loss(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred_label)
            precision = precision_score(y_true, y_pred_label, average='macro')
            recall = recall_score(y_true, y_pred_label, average='macro')
            f1 = f1_score(y_true, y_pred_label, average='macro')
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            auc_score = auc(fpr, tpr)

        metrics = [logloss,
                   acc,
                   auc_score,
                   precision,
                   recall,
                   f1]
        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        self.w.writerow([epoch+1, time_completed] + metrics)

        # Only run while using automl-gs, which tells it an epoch is finished
        # and data is recorded.
        if self.in_automl:
            sys.stdout.flush()
            print("\nEPOCH_END")
