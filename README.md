
# Image Classification Project

## Introduction
This project involves developing an image classification system using traditional machine learning techniques. The main components include data preprocessing, feature extraction, dimensionality reduction, model training, evaluation, and deployment via a Flask web application.

## Project Structure
- `ML_assignment.ipynb`: Jupyter notebook containing the data processing, feature extraction, model training, and evaluation code.
- `app.py`: Flask application code for deploying the trained model.
- `static/uploads/`: Directory to store uploaded images.
- `templates/`: Directory containing HTML templates (`index.html` and `result.html`).

To run the Flask application, follow these steps:

1. **Navigate to the project directory**: Open a terminal or command prompt and navigate to the directory where your Flask application (`app.py`) is located.

2. **Set up a virtual environment (optional)**: It's good practice to set up a virtual environment to manage dependencies. If you haven't set up one yet, you can create it using `virtualenv` or `conda`.

   ```sh
   # Using virtualenv
   virtualenv env
   # Activate virtual environment
   source env/bin/activate
   ```

3. **Install dependencies**: Make sure you have installed all required dependencies by running:

   ```sh
   pip install opencv-python scikit-learn flask
   ```

4. **Run the Flask application**: Execute the `app.py` file.
   ```sh
   python app.py
   ```

5. **Access the application**: Once the Flask application is running, open a web browser and go to the address `http://127.0.0.1:5000`. You should see the upload page where you can upload images for classification.
6. **Upload an image**: Choose an image file and click on the "Upload" button.
7. **View the classification result**: After uploading, the application should display the classification result along with the uploaded image.
8. **Stopping the application**: To stop the Flask application, press `Ctrl + C` in the terminal where the application is running.

### Data Preprocessing and Loading
1. **Data Directory**: The dataset is organized in the `dataset_full` directory, which contains subdirectories for each class (e.g., Building, Forest, Glacier, etc.).
2. **Image Loading**: Images are loaded using OpenCV (`cv2`) library. Each image is resized to 128x128 pixels and converted to grayscale.
3. **Labeling**: Labels are assigned to each image based on the directory it belongs to (class name). Class names are extracted from the directory names.

### Feature Extraction
1. **Histogram Equalization**: Enhances the contrast of the images by equalizing their histograms.
2. **Gaussian Blur**: Reduces noise and details in the images using Gaussian blur.
3. **Canny Edge Detection**: Detects edges in the images using the Canny edge detector.

### Dimensionality Reduction
Principal Component Analysis (PCA) is applied to reduce the dimensionality of the feature vectors extracted from the images. This helps in reducing computational complexity and noise in the data.

### Model Training
A K-Nearest Neighbors (KNN) classifier is trained using the extracted features after dimensionality reduction.

### Evaluation
The trained model is evaluated on the validation set using accuracy score and classification report, which includes precision, recall, and F1-score for each class.

### Flask Application (`app.py`)
1. **Image Preprocessing**: Uploaded images are preprocessed similarly to the training data (resizing, normalization, etc.).
2. **Feature Extraction for Single Image**: Extracts features for a single uploaded image using the same techniques as in training.
3. **Model Prediction**: Uses the trained model to predict the class of the uploaded image.
4. **Rendering Results**: Renders the classification result along with the uploaded image.

### File Saving
The trained KNN model and PCA object are saved using the `pickle` library for future use.

### Flask Application Structure
- `index.html`: HTML template for uploading images.
- `result.html`: HTML template for displaying classification results.

### Flask Deployment
The Flask app is run locally, and it serves the trained model to classify uploaded images.

This setup enables users to upload images via a web interface and get real-time classification results.

If you need more clarification on any part, feel free to ask!




