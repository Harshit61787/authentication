import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Feature extraction with fixed length
def extract_mfcc(path, max_len=50):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return np.mean(mfcc, axis=1), mfcc

# Load data
X_rf, y_rf, X_cnn, y_cnn = [], [], [], []
for label, d in enumerate(['human','ai']):
    for f in os.listdir(d):
        if f.endswith('.wav'):
            m, full = extract_mfcc(os.path.join(d,f))
            X_rf.append(m); y_rf.append(label)
            X_cnn.append(full); y_cnn.append(label)

X_rf = np.array(X_rf); y_rf = np.array(y_rf)
X_cnn = np.array(X_cnn)[..., np.newaxis]  # (n,13,50,1)
y_cat = to_categorical(y_cnn)

# Split
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_cnn, y_cat, test_size=0.3, random_state=42)

# RF
rf = RandomForestClassifier(n_estimators=100)
rf.fit(Xr_tr, yr_tr)

# CNN
cnn = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(13,50,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(2,activation='softmax')
])
cnn.compile('adam','categorical_crossentropy',metrics=['accuracy'])
cnn.fit(Xc_tr, yc_tr, epochs=10, verbose=1)

# Save
joblib.dump(rf, 'random_forest_model.pkl')
cnn.save('cnn_model.h5')

# Evaluate & print
rf_acc = accuracy_score(yr_te, rf.predict(Xr_te))
_, cnn_acc = cnn.evaluate(Xc_te, yc_te, verbose=0)
print(f"RF Test Accuracy: {rf_acc*100:.2f}%")
print(f"CNN Test Accuracy: {cnn_acc*100:.2f}%")
