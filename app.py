from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained model
model = load_model('ocr_model.h5')  # Replace with the actual filename of your model

# Load the LabelEncoder
le = LabelEncoder()
le.classes_ = np.load('label_encoder_classes.npy')  # Replace with the actual filename of your LabelEncoder classes

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image
    image = cv2.resize(image, (64, 64))
    
    # Ensure the image has 3 channels (for RGB images)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Convert the image to float32
    image = np.array(image, dtype=np.float32)

    # Normalize the image
    image = image / 255.0

    # Reshape the image to (1, 64, 64, 3)
    image = image.reshape(1, 64, 64, 3)

    return image

# def preprocess_image(image):
#     # Resize the image
#     image = cv2.resize(image, (64, 64))

#     # Ensure the image has 3 channels (for RGB images)
#     if len(image.shape) == 2:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     # Convert the image to float32
#     image = np.array(image, dtype=np.float32)

#     # Normalize the image
#     image = image / 255.0

#     # Reshape the image to (1, 64, 64, 3)
#     image = image.reshape(1, 64, 64, 3)

#     return image

# def preprocess_sentence(sentence):
#     # Convert the sentence to a sequence of integers (indices of characters)
#     sentence = [word2index[word] for word in sentence.split()]
    
#     # Reshape the sentence to (1, length of sentence)
#     sentence = np.array(sentence).reshape(1, -1)
    
#     return sentence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image from the POST request
        file = request.files['file']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Preprocess the image
        processed_img = preprocess_image(img)

        # Make predictions
        prediction = model.predict(processed_img)
        predicted_label = int(np.argmax(prediction))  # Convert to standard Python integer

        # Convert the predicted label back to the original class name
        predicted_class = le.inverse_transform([predicted_label])[0]

        # Return the predicted class name
        return jsonify({'result': predicted_class})

# def predict():
#     if request.method == 'POST':
#         # Get the image from the POST request
#         file = request.files['file']
#         img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

#         # Preprocess the image
#         processed_img = preprocess_image(img)

#         # Make predictions
#         predictions = model.predict(processed_img)
#         predicted_labels = [int(np.argmax(pred)) for pred in predictions]  # Convert to standard Python integers

#         # Convert the predicted labels back to the original class names
#         predicted_classes = le.inverse_transform(predicted_labels)

#         # Return the predicted class names as a list
#         return jsonify({'result': predicted_classes})

if __name__ == '__main__':
    app.run(debug=True)
