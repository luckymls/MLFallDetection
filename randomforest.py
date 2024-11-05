import os
import numpy as np
import joblib  # Per salvare e caricare il modello
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Flag per decidere se salvare il modello
save = True

# Funzione per limitare i dati alla durata massima
def limit_data_to_duration(data, max_duration=2400): # Limitato a 12 secondi (2400 letture ogni 5 ms (200 Hz)))
    return data[:max_duration]  # Limita la durata a max_duration campioni

# Funzione per estrarre le feature da un file di dati
def extract_features_from_file(data):

    accel_data = data[:, :3]  # Accelerazione
    rot_data = data[:, 3:6]   # Rotazione

    mean = np.mean(data, axis=0) # Media
    std_dev = np.std(data, axis=0) # Deviazione standard
    min_val = np.min(data, axis=0) # Valore minimo
    max_val = np.max(data, axis=0) # Valore massimo
    range_val = max_val - min_val # Range valore

    sum_accel = np.sum(np.linalg.norm(accel_data, axis=1)) # Norma 2 - accelerazione
    sum_rot = np.sum(np.linalg.norm(rot_data, axis=1)) # Norma 2 - rotazione

    # Calcola il tempo relativo in millisecondi
    num_samples = accel_data.shape[0] # Numero di campioni
    relative_time = np.arange(0, num_samples * 5, 5)  # Ogni campione è a 5 ms di distanza

    # Aggiungi una statistica del tempo (come la media e deviazione standard)
    mean_time = np.mean(relative_time) # Tempo medio
    std_dev_time = np.std(relative_time) # Deviazione standard del tempo

    # Costruisci le feature come concatenazione
    features = np.concatenate([mean, std_dev, range_val, [sum_accel, sum_rot, mean_time, std_dev_time]]) # Features
    
    return features

# Funzione per leggere i file e calcolare le feature
def load_fall_data_and_extract_features(directory, max_duration=2400):
    data = []
    labels = []
    
    for subject_folder in os.listdir(directory): 
        folder_path = os.path.join(directory, subject_folder)
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                if file_name.endswith('.txt') and "Readme" not in file_name:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    cleaned_data = []
                    for line in lines:
                        cleaned_line = line.strip().replace(';', '')
                        if cleaned_line:
                            try:
                                values = list(map(float, cleaned_line.split(',')))
                                cleaned_data.append(values[:6])
                            except ValueError as e:
                                print(f'Error converting line to float: {cleaned_line}. Error: {e}')
                    
                    if cleaned_data:
                        cleaned_data = np.array(cleaned_data)
                        limited_data = limit_data_to_duration(cleaned_data, max_duration=max_duration)
                        features = extract_features_from_file(limited_data)
                        data.append(features)
                        label = 1 if file_name.startswith('F') else 0
                        labels.append(label)

    return np.array(data), np.array(labels)

# Funzione per salvare il modello addestrato
def save_model(model, filename):
    global save
    if save:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")

# Funzione per caricare il modello
def load_model(filename):
    global save
    if save:
        if os.path.exists(filename):
            return joblib.load(filename)
        else:
            print(f"Model file {filename} does not exist.")
            return None

# Funzione per classificare nuovi dati da un secondo percorso
def classify_new_data(directory, model, scaler, max_duration=2400):
    new_data = []
    file_names = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []

    print(f"Scanning directory: {directory}")

    for subject_folder in os.listdir(directory):
        folder_path = os.path.join(directory, subject_folder)

        if os.path.isdir(folder_path):
            print(f"Found folder: {subject_folder}")
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                if file_name.endswith('.txt') and "Readme" not in file_name:
                    print(f"Processing file: {file_name}")
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    cleaned_data = []
                    for line in lines:
                        cleaned_line = line.strip().replace(';', '')
                        if cleaned_line:
                            try:
                                values = list(map(float, cleaned_line.split(',')))
                                cleaned_data.append(values[:6])
                            except ValueError as e:
                                print(f'Error converting line to float: {cleaned_line}. Error: {e}')
                    
                    if cleaned_data:
                        cleaned_data = np.array(cleaned_data)
                        limited_data = limit_data_to_duration(cleaned_data, max_duration=max_duration)
                        features = extract_features_from_file(limited_data)
                        new_data.append(features)
                        file_names.append(file_name)
                    else:
                        print(f"No valid data in file: {file_name}")
        else:
            print(f"Skipping non-folder: {subject_folder}")

    if not new_data:
        print("No data found in the second directory.")
        return []

    new_data = np.array(new_data)
    print(f"Shape of new data before scaling: {new_data.shape}")

    new_data = scaler.transform(new_data)
    predictions = model.predict(new_data)
    
    return list(zip(file_names, predictions))

# Classe per gestire il caricamento e la classificazione con il modello
class RandomTree:
    model_filename = "random_forest_model.joblib"
    scaler_filename = "scaler.joblib"
    
    def __init__(self, max_duration=2400):
        self.max_duration = max_duration
        self.load()

    def load(self):
        self.model = load_model(self.model_filename)
        if self.model is None:
            print(f"Model file {self.model_filename} does not exist.")
        self.scaler = joblib.load(self.scaler_filename) if os.path.exists(self.scaler_filename) else None

    def load_file(self, file_path):
        self.file = file_path

    def classify(self):
        return classify_single_file(self.file, self.model, self.scaler, self.max_duration)

# Impostazioni per i file del modello e scaler
model_filename = "random_forest_model.joblib"
scaler_filename = "scaler.joblib"

# Carica il modello addestrato (se esistente)
rf = load_model(model_filename)

# Se il modello esiste, usa quello salvato per classificare nuovi dati
if rf:
    print("Using saved model to classify new data.")
    
    # Carica anche lo scaler addestrato
    scaler = joblib.load(scaler_filename)

    # Classifica i nuovi dati da un secondo percorso
    second_directory = "NEW_DATA/"  # Specifica il percorso corretto
    new_predictions = classify_new_data(second_directory, rf, scaler)

    # Mostra le predizioni per i nuovi dati
    if new_predictions:
        for file_name, prediction in new_predictions:
            activity = "Fall" if prediction == 1 else "Normal activity"
            print(f'File: {file_name}, Predicted: {activity}')
else:
    print("No saved model found, training the model...")

    # Carica i dati e le feature dal dataset SisFall
    directory = "SIS/"  # Specifica il percorso corretto
    
    data, labels = load_fall_data_and_extract_features(directory)

    # Verifica le dimensioni di data e labels
    print(f'Shape of data: {data.shape}')
    print(f'Length of labels: {len(labels)}')

    # Dividi il dataset in training e testing set
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # Normalizzazione
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Addestra il modello
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Salva il modello e lo scaler
    save_model(rf, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to {scaler_filename}")

    # Predizione e valutazione sul test set
    y_pred = rf.predict(X_test)

    # Valutazione del modello
    cm = confusion_matrix(y_test, y_pred)

    # Estrai i valori dalla Confusion Matrix
    TN, FP, FN, TP = cm.ravel()

    # Stampa la Confusion Matrix
    print("Confusion Matrix:")
    print(cm)

    # Stampa i dettagli
    print(f"Veri Negativi (TN): {TN}")
    print(f"Falsi Positivi (FP): {FP}")
    print(f"Falsi Negativi (FN): {FN}")
    print(f"Veri Positivi (TP): {TP}")

    # Stampa il classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Stampa i file classificati erroneamente
    for i, (file_name, true_label, pred_label) in enumerate(zip(os.listdir(directory), labels, y_pred)):
        if true_label != pred_label:
            activity_true = "Fall" if true_label == 1 else "Normal activity"
            activity_pred = "Fall" if pred_label == 1 else "Normal activity"
            print(f"Incorrectly classified file: {file_name}, True label: {activity_true}, Predicted label: {activity_pred}")
