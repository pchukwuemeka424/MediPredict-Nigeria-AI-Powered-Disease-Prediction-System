from flask import Flask, render_template, request, jsonify
from flask_mail import Mail, Message
import pickle
import numpy as np
import pandas as pd
import re

app = Flask(__name__)

# Configure email settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your-app-password'  # Replace with your app password
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@gmail.com'  # Replace with your email

mail = Mail(app)

# Load the model and related data
with open('disease_prediction_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Load the dataset for symptom matching
data = pd.read_csv('data/dataset.csv')

rf_model = model_data['model']
le = model_data['label_encoder']
symptoms_list = model_data['symptoms_list']
symptom_severity_dict = model_data['symptom_severity_dict']
disease_descriptions = model_data['disease_descriptions']
disease_precautions = model_data['disease_precautions']

# Get disease-symptom mapping for improved matching
disease_to_symptoms = model_data.get('disease_to_symptoms', {})

# Fix: Get disease classes from label encoder if not in model_data
if 'disease_classes' in model_data and model_data['disease_classes']:
    disease_classes = model_data['disease_classes']
else:
    # Fallback to classes from the label encoder
    disease_classes = le.classes_.tolist()

print(f"Loaded {len(disease_classes)} disease classes")

def standardize_symptom(symptom):
    """Standardize symptom text to ensure consistent format"""
    if not isinstance(symptom, str) or symptom == '0':
        return '0'
    
    # Trim whitespace and lowercase
    symptom = symptom.strip().lower()
    
    # Remove punctuation
    symptom = re.sub(r'[^\w\s]', '', symptom)
    
    # Replace spaces with underscores
    symptom = re.sub(r'\s+', '_', symptom)
    
    return symptom

def calculate_severity(symptoms):
    """Calculate severity score based on symptoms"""
    severity_score = 0
    for symptom in symptoms:
        if symptom in symptom_severity_dict:
            severity_score += symptom_severity_dict[symptom]
        else:
            # Try to find closest match for the symptom
            for known_symptom in symptom_severity_dict:
                if symptom in known_symptom or known_symptom in symptom:
                    severity_score += symptom_severity_dict[known_symptom]
                    break
    return severity_score

def predict_diseases(symptoms_input, top_n=3, confidence_threshold=5.0):
    """
    Predicts top N diseases based on a list of symptoms.
    Returns predicted diseases, confidence scores, descriptions, precautions and matching symptoms.
    """
    # Standardize input symptoms
    standardized_input = [standardize_symptom(s) for s in symptoms_input]
    
    # Create feature vector
    input_vector = np.zeros(len(symptoms_list))
    valid_symptoms = []
    
    for symptom in standardized_input:
        # Exact match
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            input_vector[index] = 1
            valid_symptoms.append(symptom)
            continue
            
        # Try to find closest match
        matched = False
        for i, known_symptom in enumerate(symptoms_list):
            # Fix: Ensure both symptom and known_symptom are strings before using 'in' operator
            if (isinstance(symptom, str) and isinstance(known_symptom, str) and 
                (symptom in known_symptom or known_symptom in symptom)):
                input_vector[i] = 1
                valid_symptoms.append(known_symptom)
                matched = True
                break
    
    # Handle case where no valid symptoms were provided
    if not valid_symptoms:
        return [{
            'disease': 'Unknown',
            'confidence': 0,
            'severity_score': 0,
            'description': "No valid symptoms were provided. Please check your symptoms and try again.",
            'precautions': ["Consult with a healthcare professional"],
            'matching_symptoms': []
        }]
    
    # Get prediction probabilities
    probabilities = rf_model.predict_proba([input_vector])[0]
    
    # Fix: Ensure we don't request more predictions than available classes
    actual_top_n = min(top_n, len(disease_classes), len(probabilities))
    
    # Get indices of top N predictions
    top_indices = probabilities.argsort()[::-1][:actual_top_n]
    
    predictions = []
    for idx in top_indices:
        # Fix: Double check that idx is valid
        if idx >= len(disease_classes):
            continue
            
        confidence = probabilities[idx] * 100
        if confidence >= confidence_threshold:
            disease = disease_classes[idx]
            
            # Find what symptoms typically match this disease using the improved mapping
            typical_symptoms = disease_to_symptoms.get(disease, set())
            
            # Find exact matching symptoms
            matching_symptoms = [s for s in valid_symptoms if s in typical_symptoms]
            
            # For better user experience, also check if any input symptom is similar to disease symptoms
            similar_matches = []
            for vs in valid_symptoms:
                if vs not in matching_symptoms:
                    for ts in typical_symptoms:
                        # Fix: Add type checking here as well
                        if (isinstance(vs, str) and isinstance(ts, str) and 
                            (vs in ts or ts in vs)):
                            similar_matches.append(vs)
                            break
            
            # Combine exact and similar matches
            all_matching = list(set(matching_symptoms + similar_matches))
            
            predictions.append({
                'disease': disease,
                'confidence': round(confidence, 2),
                'severity_score': calculate_severity(valid_symptoms),
                'description': disease_descriptions.get(disease, "No description available"),
                'precautions': [p for p in disease_precautions.get(disease, ["No specific precautions available"]) 
                               if p != '0' and p is not None],
                'matching_symptoms': all_matching,
                'all_symptoms': list(typical_symptoms)  # Add all typical symptoms for this disease
            })
    
    # If no predictions meet the threshold, include the top one regardless
    if not predictions and len(top_indices) > 0:
        top_idx = top_indices[0]
        # Fix: Check again that index is valid
        if top_idx < len(disease_classes):
            disease = disease_classes[top_idx]
            
            # Get typical symptoms for this disease
            typical_symptoms = disease_to_symptoms.get(disease, set())
            matching_symptoms = [s for s in valid_symptoms if s in typical_symptoms]
            
            similar_matches = []
            for vs in valid_symptoms:
                if vs not in matching_symptoms:
                    for ts in typical_symptoms:
                        # Fix: Add type checking here as well
                        if (isinstance(vs, str) and isinstance(ts, str) and 
                            (vs in ts or ts in vs)):
                            similar_matches.append(vs)
                            break
            
            all_matching = list(set(matching_symptoms + similar_matches))
            
            predictions.append({
                'disease': disease,
                'confidence': round(probabilities[top_idx] * 100, 2),
                'severity_score': calculate_severity(valid_symptoms),
                'description': disease_descriptions.get(disease, "No description available"),
                'precautions': [p for p in disease_precautions.get(disease, ["No specific precautions available"]) 
                               if p != '0' and p is not None],
                'matching_symptoms': all_matching,
                'all_symptoms': list(typical_symptoms)  # Add all typical symptoms for this disease
            })
        else:
            # Provide a fallback if we somehow have an invalid index
            predictions.append({
                'disease': 'Unknown',
                'confidence': 0,
                'severity_score': calculate_severity(valid_symptoms),
                'description': "Unable to determine a disease based on these symptoms.",
                'precautions': ["Consult with a healthcare professional"],
                'matching_symptoms': [],
                'all_symptoms': []
            })
    
    return predictions

def predict_disease(symptoms_input):
    """
    Legacy function that returns only the top prediction.
    """
    predictions = predict_diseases(symptoms_input, top_n=1)
    return predictions[0]

@app.route('/')
def home():
    # Convert all symptoms to strings before sorting to avoid type comparison errors
    string_symptoms = [str(symptom) for symptom in symptoms_list]
    return render_template('index.html', symptoms=sorted(string_symptoms))

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    
    if not selected_symptoms:
        return jsonify({
            'error': 'Please select at least one symptom'
        })
    
    # Get number of results to return
    top_n = request.form.get('top_n', 3, type=int)
    
    # Make predictions
    results = predict_diseases(selected_symptoms, top_n=top_n)
    
    return jsonify({
        'predictions': results,
        'selected_symptoms': selected_symptoms
    })

@app.route('/subscribe', methods=['POST'])
def subscribe():
    email = request.form.get('email')
    
    if not email:
        return jsonify({
            'error': 'Please provide a valid email address'
        })
    
    try:
        # Send welcome email with health tips
        msg = Message('Welcome to MediPredict Nigeria!', recipients=[email])
        msg.html = '''
        <h1>Welcome to MediPredict Nigeria!</h1>
        <p>Thank you for subscribing to our health tips and updates.</p>
        <p>Here are some important health tips for you:</p>
        <ul>
            <li>Wash your hands regularly with soap and water</li>
            <li>Maintain a balanced diet with plenty of fruits and vegetables</li>
            <li>Stay hydrated by drinking enough water daily</li>
            <li>Get regular exercise for at least 30 minutes a day</li>
            <li>Get enough sleep (7-9 hours per night)</li>
            <li>Avoid smoking and limit alcohol consumption</li>
            <li>Practice safe sex</li>
            <li>Get regular health check-ups</li>
        </ul>
        <p>If you have any symptoms, please use our disease prediction tool to get preliminary information.</p>
        <p>Remember: This tool is not a substitute for professional medical advice.</p>
        <p>Best regards,<br>The MediPredict Nigeria Team</p>
        '''
        mail.send(msg)
        
        return jsonify({
            'success': 'Thank you for subscribing! You will receive health tips in your email.'
        })
    except Exception as e:
        return jsonify({
            'error': f'Failed to send email. Please try again. Error: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)