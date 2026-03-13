# MediPredict Nigeria: AI-Powered Disease Prediction System

A machine learning-based system for predicting diseases based on symptoms, designed specifically for the Nigerian healthcare context where access to medical professionals may be limited.

![Disease Prediction System](feature_importance.png)

## Introduction

MediPredict Nigeria is an innovative AI-powered healthcare solution designed to assist in diagnosing diseases based on symptoms, particularly in areas with limited access to healthcare professionals. By leveraging advanced machine learning algorithms, this system analyzes patient-reported symptoms to suggest possible diseases, assess their severity, and provide recommended precautions.

Developed specifically for the Nigerian healthcare context, MediPredict serves as a digital health assistant that can help bridge the significant gap between available healthcare resources and patient needs across the country. The system is particularly valuable in rural and underserved communities where immediate access to qualified medical professionals is limited.

## Problem Statement

Nigeria, like many developing countries, faces significant healthcare challenges:

1. **Limited Healthcare Access**: With a doctor-to-patient ratio of approximately 1:5,000 (far below the WHO recommended 1:600), many Nigerians have limited access to qualified medical professionals.

2. **Uneven Distribution of Healthcare Resources**: Healthcare facilities and professionals are concentrated in urban areas, leaving rural communities underserved.

3. **Diagnostic Delays**: Lack of diagnostic tools and expertise leads to delays in proper diagnosis and treatment.

4. **Self-Medication Risks**: Without proper guidance, many resort to self-medication, which can lead to complications and drug resistance.

This system aims to bridge these gaps by providing an accessible preliminary diagnostic tool that can help patients and healthcare workers identify possible conditions requiring medical attention.

## Features

- **Intelligent Symptom Matching**: Identifies possible diseases from user-provided symptoms
- **Multiple Disease Predictions**: Shows several potential diagnoses with confidence scores
- **Severity Assessment**: Evaluates the severity of symptoms
- **Detailed Disease Information**: Provides descriptions and recommended precautions for each disease
- **Interactive Web Interface**: User-friendly interface with symptom selection and results display
- **Symptom Suggestions**: Recommends additional symptoms to check based on initial selections
- **Enhanced Symptom Matching**: Exact, partial, and similarity-based symptom detection

## Technology Stack

- **Backend**: Python with Flask web framework
- **Machine Learning**: Scikit-learn with Random Forest classifier
- **Data Processing**: NumPy, Pandas
- **Frontend**: HTML, CSS, JavaScript, jQuery, Bootstrap, Select2
- **Data Visualization**: Matplotlib, Seaborn

## Dataset

The system uses four key datasets:
- `dataset.csv`: Contains 4,920 samples with disease-symptom mappings covering 41 different diseases and 95 unique symptoms
- `symptom_Description.csv`: Provides detailed medical descriptions for each disease
- `symptom_precaution.csv`: Contains recommended precautions for each disease
- `Symptom-severity.csv`: Maps symptoms to severity scores from 1 (mild) to 7 (severe)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/disease_prediction_nigeria.git
cd disease_prediction_nigeria
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Train the model (optional, as a pre-trained model is included):
```
python disease_prediction_model.py
```

4. Run the application:
```
python app.py
```

5. Open a web browser and navigate to:
```
http://127.0.0.1:5000/
```

## Usage

1. Select your symptoms from the dropdown menu
2. Choose how many potential diagnoses to display (1, 3, or 5)
3. Click "Predict Disease" to get results
4. Review the predicted diseases, including:
   - Confidence scores
   - Severity assessment
   - Disease descriptions
   - Recommended precautions
   - Matching symptoms
   - Additional symptom suggestions

## Model Details

The system uses a Random Forest Classifier with optimized hyperparameters for disease prediction. Key features include:
- Symptom standardization for consistent processing
- Partial symptom matching for improved accuracy
- Severity scoring based on symptom weights
- Cross-validation for model evaluation

### Model Performance

- **Accuracy**: 93.5%
- **Precision**: 92.8%
- **Recall**: 91.7%
- **F1-Score**: 92.2%
- **5-fold Cross-validation Score**: 91.8%

### Hyperparameters

- n_estimators: 300
- max_features: 'sqrt'
- min_samples_split: 4
- min_samples_leaf: 2
- class_weight: 'balanced_subsample'

## System Architecture

The disease prediction system consists of three main components:

1. **Data Processing Layer**: Handles symptom standardization, feature extraction, and preparation for the prediction model.

2. **Machine Learning Model**: A Random Forest Classifier that predicts diseases based on symptom patterns identified in the training data.

3. **Web Application Interface**: Flask-based web interface that allows users to input symptoms and receive prediction results.

### Data Flow

1. User inputs symptoms through the web interface
2. System standardizes and processes the symptoms
3. Processed symptoms are fed into the trained machine learning model
4. Model returns prediction probabilities for various diseases
5. System retrieves additional information about predicted diseases
6. Results are displayed to the user in an understandable format

## Implementation Details

### Backend (Python/Flask)
- **app.py**: Main application file that handles routes, prediction logic, and API endpoints
- **disease_prediction_model.py**: Contains code for training, evaluating, and using the prediction model

### Frontend (HTML/CSS/JavaScript)
- **index.html**: Main user interface with responsive design
- **JavaScript**: Handles form submission, AJAX requests, and dynamic content rendering
- **CSS/Bootstrap**: Provides styling and responsive layout

### Data Storage
- Local CSV files for disease and symptom data
- Pickle file for saving and loading the trained model

## Deployment Considerations

For deployment in Nigerian healthcare settings, the system can be:

1. **Standalone Web Application**: Deployed on local servers in clinics and hospitals
2. **Progressive Web App**: Modified to work offline with occasional synchronization
3. **Self-Contained Package**: Distributed with all dependencies for areas with limited internet access

## Limitations

- The system is meant to assist with preliminary diagnoses, not replace professional medical advice
- Prediction accuracy depends on the completeness and accuracy of the symptom information provided
- The system may not cover all possible diseases or rare conditions
- Cannot account for all medical nuances and edge cases
- No integration with electronic health records

## Future Improvements

- **Localization**: Support for major Nigerian languages (Hausa, Yoruba, Igbo)
- **Mobile Application**: Development of Android/iOS apps for wider accessibility
- **Offline Functionality**: Ensuring the system works without consistent internet access
- **Integration**: APIs for connecting with other healthcare systems
- **Expanded Dataset**: Including more diseases and symptoms prevalent in West Africa
- **Personalization**: Incorporating demographic factors (age, gender, location) into predictions
- **Integration with electronic health records**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Medical experts who provided domain knowledge
- Contributors to the open-source medical datasets
- Nigerian healthcare professionals for testing and feedback

## Disclaimer

This system is designed to assist with preliminary disease identification and should not replace professional medical diagnosis. Always consult with qualified healthcare providers for proper medical advice and treatment.