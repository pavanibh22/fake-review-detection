from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

MAXLEN = 281

app = Flask(__name__)

def preprocess(user_input):
    try:
        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            sequenced_input = tokenizer.texts_to_sequences([user_input])
            processed_input = pad_sequences(sequenced_input, maxlen=MAXLEN)
            return processed_input
    except Exception:
        return None

# Load data from a pickle file (data.pickle)
def load_data():
    try:
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        return loaded_model
    except FileNotFoundError:
        return None

# Define a route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle the form submission
@app.route('/process', methods=['POST'])
def process():
    user_review = request.form.get('user_review')  # Get the input from the form

    if user_review=="": 
        return render_template('index.html', result=None)
    # You can process user_input here or perform any other desired actions.
    # Load data from the pickle file
    loaded_model = load_data()

    # preprocess the text
    processed_input = preprocess(user_review)

      # Make a prediction using the loaded model
    prediction = loaded_model.predict(processed_input)
    predicted_classes = np.where(prediction > 0.5, 1, 0)

    print("prediction: ", predicted_classes)

    if(predicted_classes[0][0] == 0):
        result = False
    elif(predicted_classes[0][0] == 1):
        result = True
    else:
        result = None

    print(f"Input String: {user_review}, Data from Pickle: {prediction}")

    return render_template('index.html', result=result, user_review=user_review)

if __name__ == '__main__':
    app.run(debug=True)
