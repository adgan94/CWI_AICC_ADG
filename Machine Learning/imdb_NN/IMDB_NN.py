import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate

# Configurations
NUM_WORDS = 10000
EPOCHS = 5
BATCH_SIZE = 512

# Load and Preprocess the IMDB Dataset
(X_train_sequences, y_train), (X_test_sequences, y_test) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

# Multi-hot encode the sequences
def multi_hot_encode_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):results[i, word_indices] = 1
    return results

# Split and encode the data
X_train = multi_hot_encode_sequences(X_train_sequences, NUM_WORDS)
X_test = multi_hot_encode_sequences(X_test_sequences, NUM_WORDS)

INPUT_DIM = X_train.shape[1]

sequential_model = Sequential([
    Dense(256, activation='relu', input_shape=(INPUT_DIM,)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

sequential_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_seq = sequential_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

loss_seq, acc_seq = sequential_model.evaluate(X_test, y_test, verbose=0)
print(f"Sequential Model Test Accuracy: {acc_seq:.4f}")

# Wide and Deep NN
input_layer = Input(shape=(INPUT_DIM,), name='input_layer')

# Deep Component (Generalization)
deep_branch = Dense(128, activation='relu', name='deep_layer_1')(input_layer)
deep_branch = Dropout(0.5)(deep_branch)
deep_branch = Dense(64, activation='relu', name='deep_layer_2')(deep_branch)
deep_branch = Dropout(0.5)(deep_branch)
deep_features = Dense(32, activation='relu', name='deep_features')(deep_branch)


# Wide Component (Memorization)
combined_features = concatenate([input_layer, deep_features], name='combined_features')

# Output Layer for Classification
output_layer = Dense(1, activation='sigmoid', name='output')(combined_features)

wide_deep_model = Model(inputs=input_layer, outputs=output_layer)

wide_deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_wd = wide_deep_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    verbose=1
)

loss_wd, acc_wd = wide_deep_model.evaluate(X_test, y_test, verbose=0)
print(f"Wide and Deep Model Test Accuracy: {acc_wd:.4f}")

# Final Comparison
print(f"Sequential NN Accuracy: {acc_seq:.4f}")
print(f"Wide and Deep NN Accuracy: {acc_wd:.4f}")