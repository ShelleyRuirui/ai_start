import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout, Flatten

# Load preprocessed dataset
data = np.load("train_data.npz", allow_pickle=True)
X_seq = data["X_seq"]
X_genre = data["X_genre"]
y_target = data["y_target"]
vocab_size = int(data["vocab_size"])
SEQ_LENGTH = int(data["SEQ_LENGTH"])
del data  # free memory of loaded npz object

lstm_dim = 64

# Build two-input conditional LSTM model
# Input 1: note time series
input_notes = Input(shape=(SEQ_LENGTH,))
emb_notes = Embedding(input_dim=vocab_size, output_dim=64)(input_notes)
# Use two LSTM layers to capture complex temporal dependencies in music
# sequences. The first LSTM returns sequences for the second LSTM to process
# the entire sequence context.
lstm_1 = LSTM(lstm_dim, return_sequences=True)(emb_notes)
lstm_2 = LSTM(lstm_dim)(lstm_1)
# Add dropout to prevent overfitting, especially important given the model complexity and dataset size.
lstm_out = Dropout(0.1)(lstm_2)

# Input 2: genre condition (0=jazz,1=blues,2=pop)
input_genre = Input(shape=(1,))
emb_genre = Embedding(input_dim=3, output_dim=64)(input_genre)
# Make shape align
flat_genre = Flatten()(emb_genre) 
# This layer is needed to transform the genre embedding into a compatible shape
# for concatenation and better capture non linear genre characteristics.
genre_dense = Dense(128, activation="tanh")(flat_genre)

# Merge time-series feature + genre label
concat_layer = Concatenate()([lstm_out, genre_dense])
hidden_dense = Dense(128, activation="tanh")(concat_layer)
output_layer = Dense(vocab_size, activation="softmax")(hidden_dense)

# Compile model
# 加入回调，loss 5轮不下降就把lr×0.5
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6)
callbacks = [
    EarlyStopping(monitor="loss", patience=5, restore_best_weights=True),
    reduce_lr,
    # 每1个epoch保存一次全部模型
    ModelCheckpoint(
        filepath="./checkpoints/model_epoch_{epoch:02d}_loss_{loss:.2f}.h5",
        monitor="loss",
        save_best_only=False,   # False=每轮都存；True=只存loss最低的
        save_weights_only=False,
        verbose=1
    )
]
optimizer = Adam(learning_rate=0.002)  # 默认0.001，小幅抬升学习率
model = Model(inputs=[input_notes, input_genre], outputs=output_layer)
# Use sparse_categorical_crossentropy and avoid OOM from
# categorical_crossentropy + one hot
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)
model.summary()

# Start training (BPTT auto executed inside Keras LSTM layer)
history = model.fit(
    [X_seq, X_genre],
    y_target,
    batch_size=16,
    epochs=100,
    # validation_split=0.1,
    verbose=1,
    callbacks=callbacks,
)

# Save trained model
model.save("music_lstm_genre_model.h5")
print("Training complete, model saved as music_lstm_genre_model.h5")