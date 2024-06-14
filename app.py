from flask import Flask, request, render_template
import numpy as np
import cv2
import pickle

app = Flask(__name__)

class_names = ['Building', 'Forest', 'Glacier', 'Mountains', 'Sea', 'Streets']
# Load the trained SVM model and PCA
with open('knn_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

def histogram_equalization(image):
    """Apply histogram equalization to the image."""
    # Convert the image to 8-bit grayscale if it's not already
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    # Apply histogram equalization
    return cv2.equalizeHist(image)

def edge_detection_canny(image, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection."""
    image = (image * 255).astype(np.uint8)  # Ensure the image is in 8-bit format
    return cv2.Canny(image, low_threshold, high_threshold)

def gaussian_blur(image, kernel_size=(5, 5)):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def preprocess_image(image_path):
    """Preprocess the uploaded image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as 8-bit grayscale
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

def extract_features_for_single_image(image):
    """Extract features for a single image."""
    # Low-level features
    hist_eq = histogram_equalization(image)
    gauss_blur = gaussian_blur(image)
    canny_edges = edge_detection_canny(image)
    
    # Combine features into a single feature vector
    feature_vector = np.concatenate([
        hist_eq.flatten(),
        gauss_blur.flatten(),
        canny_edges.flatten()
    ])
    
    return feature_vector

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = "static/uploads/" + file.filename
            

            file.save(image_path)
            
            image = preprocess_image(image_path)
            features = extract_features_for_single_image(image)
            features_reduced = pca.transform([features])
            prediction = svm_model.predict(features_reduced)
            class_name = class_names[prediction[0]]
            
            return render_template('result.html', class_name=class_name, image_path ="uploads/" + file.filename
)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
