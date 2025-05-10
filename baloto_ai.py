from tensorflow import keras
from keras import layers
from art import text2art
import tensorflow as tf
import numpy as np
import shutil
import socket
import sys

# Bloquear el acceso a Internet
socket.socket = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Internet access is disabled"))

# === Callback Personalizado para Salida Estilizada de Entrenamiento ===
class CustomTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.total_epochs = None
        self.bar_length = 40

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        print("\033[1;36m" + "=" * 60 + "\033[0m")
        print("\033[1;35mTraining Progress:\033[0m")

    def on_epoch_end(self, epoch, logs=None):
        current = epoch + 1
        percent = current / self.total_epochs
        filled_len = int(self.bar_length * percent)
        bar = '█' * filled_len + '·' * (self.bar_length - filled_len)
        sys.stdout.write(f"\r\033[1;34m[Epoch {current}/{self.total_epochs}] [{bar}] {percent*100:.1f}%\033[0m")
        sys.stdout.flush()

        if current == self.total_epochs:
            print("\n\033[1;36m" + "=" * 60 + "\033[0m\n")


# === Funciones de Interfaz ===
def print_banner():
    print("\033[1;34m" + text2art("Baloto AI", font="block"))
    print("\033[1;33m" + "Lottery Prediction Artificial Intelligence".center(60) + "\033[0m")
    print("\033[1;36m" + "=" * 60 + "\033[0m")

def print_status(message):
    print(f"\033[1;34m[•]\033[0m \033[1;37m{message}\033[0m")

def print_intro():
    print_banner()
    print_status("Starting LotteryAI Prediction System...")
    print("\033[1;36m" + "=" * 60 + "\033[0m")

# === Funciones de Entrenamiento ===
def load_data():
    print_status("Loading training data...")
    if not tf.io.gfile.exists('data.txt'):
        raise FileNotFoundError("data.txt not found")

    data = np.genfromtxt('data.txt', delimiter=',', dtype=int)
    if data.shape[1] != 6:
        raise ValueError("Each row in data.txt must have exactly 6 numbers (5 regular + 1 bonus)")

    regular_numbers = data[:, :5]
    bonus_numbers = data[:, 5]

    train_size = int(0.8 * len(data))

    x_train = regular_numbers[:train_size]
    y_train_main = [regular_numbers[:train_size][:, i] for i in range(5)]
    y_train_bonus = bonus_numbers[:train_size]

    x_val = regular_numbers[train_size:]
    y_val_main = [regular_numbers[train_size:][:, i] for i in range(5)]
    y_val_bonus = bonus_numbers[train_size:]

    return (x_train, y_train_main, y_train_bonus), (x_val, y_val_main, y_val_bonus)

def create_model():
    print_status("Building neural network model...")
    input_layer = layers.Input(shape=(5,))
    x = layers.Embedding(input_dim=44, output_dim=32)(input_layer)
    x = layers.LSTM(64)(x)

    output_main = [layers.Dense(44, activation='softmax', name=f'main_output_{i}')(x) for i in range(5)]
    output_bonus = layers.Dense(44, activation='softmax', name='bonus_output')(x)

    model = keras.Model(inputs=input_layer, outputs=output_main + [output_bonus])

    loss_dict = {f'main_output_{i}': 'sparse_categorical_crossentropy' for i in range(5)}
    loss_dict['bonus_output'] = 'sparse_categorical_crossentropy'

    metrics_dict = {f'main_output_{i}': 'accuracy' for i in range(5)}
    metrics_dict['bonus_output'] = 'accuracy'

    model.compile(
        optimizer='adam',
        loss=loss_dict,
        metrics=metrics_dict
    )
    return model

def train_model(model, x_train, y_train_main, y_train_bonus, x_val, y_val_main, y_val_bonus):
    print_status("Training model (100 epochs)...\n")
    train_targets = {f'main_output_{i}': y_train_main[i] for i in range(5)}
    train_targets['bonus_output'] = y_train_bonus

    val_targets = {f'main_output_{i}': y_val_main[i] for i in range(5)}
    val_targets['bonus_output'] = y_val_bonus

    callback = CustomTrainingCallback()

    return model.fit(
        x_train, train_targets,
        validation_data=(x_val, val_targets),
        epochs=100,
        verbose=0,  # Desactiva la salida estándar
        callbacks=[callback]
    )

def predict_numbers(model, x_val):
    predictions = model.predict(x_val)
    pred_main = np.array([np.argmax(p, axis=1) for p in predictions[:-1]]).T
    pred_bonus = np.argmax(predictions[-1], axis=1)
    return pred_main, pred_bonus

def print_results(pred_main, pred_bonus):
    print_status("Generating predictions...")
    print("\033[1;36m" + "=" * 60 + "\033[0m")
    print("\033[1;32m" + "🎯 PREDICTED NUMBERS 🎯".center(60) + "\033[0m")
    print("\033[1;36m" + "-" * 60 + "\033[0m")
    if pred_main.shape[0] > 0:
        main_nums = ', '.join(map(str, np.sort(pred_main[0])))
        print(f"\033[1;33mMain Numbers:\033[0m \033[1;37m{main_nums}\033[0m")
        print(f"\033[1;33mBonus Number:\033[0m \033[1;37m{pred_bonus[0]}\033[0m")
    else:
        print("\033[1;31mNo predictions available\033[0m")
    print("\033[1;36m" + "=" * 60 + "\033[0m")

# === Ejecución Principal ===
def main():
    print("\x1b[H\x1b[2J\x1b[3J")  # Limpia la consola
    print_intro()
    try:
        (x_train, y_train_main, y_train_bonus), (x_val, y_val_main, y_val_bonus) = load_data()
        model = create_model()
        train_model(model, x_train, y_train_main, y_train_bonus, x_val, y_val_main, y_val_bonus)
        pred_main, pred_bonus = predict_numbers(model, x_val)
        print_results(pred_main, pred_bonus)
    except Exception as e:
        print("\033[1;31m[!] Error:\033[0m", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()