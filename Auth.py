from flask import Flask, request, jsonify
import librosa, numpy as np, tempfile, os, joblib, tensorflow as tf
from flask_cors import CORS

rf = joblib.load('random_forest_model.pkl')
cnn = tf.keras.models.load_model('cnn_model.h5')

def extract_features(path, max_len=50):
    y, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    rf_feat = np.mean(mfcc, axis=1)
    cnn_feat = mfcc.reshape(1,13,max_len,1)
    return rf_feat, cnn_feat

from flask import Flask, request, jsonify, render_template
# … your imports …

app = Flask(__name__, static_folder='static', template_folder='xyz')
CORS(app)
@app.route('/')
def home():
    return render_template('main1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify(error='No audio provided.')
    f = request.files['audio']
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        f.save(tmp.name)
    try:
        rf_feat, cnn_feat = extract_features(tmp.name)
        rfp = rf.predict([rf_feat])[0]
        cp = np.argmax(cnn.predict(cnn_feat)[0])
        return jsonify(
            rf_prediction = 'Human' if rfp==0 else 'AI',
            cnn_prediction = 'Human' if cp==0 else 'AI'
        )
    except Exception as e:
        return jsonify(error=str(e))
    finally:
        os.remove(tmp.name)

if __name__=='__main__':
    app.run(host='127.0.0.1', port=81, debug=True)

