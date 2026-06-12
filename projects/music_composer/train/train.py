import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout

# Load preprocessed dataset
data = np.load("train_data.npz", allow_pickle=True)
X_seq = data["X_seq"]
X_genre = data["X_genre"]
y_target = data["y_target"]
vocab_size = int(data["vocab_size"])
SEQ_LENGTH = int(data["SEQ_LENGTH"])

# One-hot encode output labels
y_onehot = to_categorical(y_target, num_classes=vocab_size)

# Build two-input conditional LSTM model
# Input 1: note time series
input_notes = Input(shape=(SEQ_LENGTH,))
emb_notes = Embedding(input_dim=vocab_size, output_dim=128)(input_notes)
lstm_1 = LSTM(256, return_sequences=True)(emb_notes)
lstm_2 = LSTM(256)(lstm_1)
lstm_out = Dropout(0.2)(lstm_2)

# Input 2: genre condition (0=jazz,1=blues,2=pop)
input_genre = Input(shape=(1,))
emb_genre = Embedding(input_dim=3, output_dim=64)(input_genre)
genre_dense = Dense(128, activation="tanh")(emb_genre)

# Merge time-series feature + genre label
concat_layer = Concatenate()([lstm_out, genre_dense])
hidden_dense = Dense(256, activation="tanh")(concat_layer)
output_layer = Dense(vocab_size, activation="softmax")(hidden_dense)

# Compile model
model = Model(inputs=[input_notes, input_genre], outputs=output_layer)
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# Start training (BPTT auto executed inside Keras LSTM layer)
history = model.fit(
    [X_seq, X_genre],
    y_onehot,
    batch_size=64,
    epochs=120,
    validation_split=0.1
)

# Save trained model
model.save("music_lstm_genre_model.h5")
print("Training complete, model saved as music_lstm_genre_model.h5")