import pandas as pd
import joblib

class GenderPredictor:
    def __init__(self):
        self.model = joblib.load('gender_prediction_model.pkl')
        self.scaler = joblib.load('scaler.pkl')

    def predict_gender(self, height, weight):
        input_data = pd.DataFrame([[height, weight]], columns=['Height', 'Weight'])
        
        input_data['BMI'] = input_data['Weight'] / ((input_data['Height'] / 100) ** 2)
        input_data['Height_Weight_Ratio'] = input_data['Height'] / input_data['Weight']
        
        final_input = input_data[['Height', 'Weight', 'BMI', 'Height_Weight_Ratio']]
        
        final_input_scaled = self.scaler.transform(final_input)
        
        prediction = self.model.predict(final_input_scaled)
        
        gender = 'Male' if prediction == 0 else 'Female'
        
        return gender

    def predict_for_dataset(self, data):
        data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
        data['Height_Weight_Ratio'] = data['Height'] / data['Weight']
        
        final_input = data[['Height', 'Weight', 'BMI', 'Height_Weight_Ratio']]
        final_input_scaled = self.scaler.transform(final_input)
    
        predictions = self.model.predict(final_input_scaled)
    
        data['Predicted_Gender'] = ['Male' if pred == 0 else 'Female' for pred in predictions]
        
        return data
gender_predictor = GenderPredictor()
height = float(input("Enter height (cm): "))
weight = float(input("Enter weight (kg): "))

predicted_gender = gender_predictor.predict_gender(height, weight)
print(f"The predicted gender is: {predicted_gender}")
predict_option = input("Do you want to check predictions for a dataset? (yes/no): ")

if predict_option.lower() == 'yes':
    dataset = pd.DataFrame({
        'Height': [188.0, 149.0, 157.0, 183.0, 158.0, 179.832, 160.0, 150.0, 164.0, 172.0, 
                   124.97, 182.0, 150.0, 173.0, 175.0, 180.0, 180.0, 176.784, 167.0, 170.0, 168.0],
        'Weight': [80, 53, 37, 60, 58, 65, 43, 40, 60, 75, 44, 75, 45, 60, 50, 74, 55, 64, 61, 70, 55]
    })
    
    predicted_dataset = gender_predictor.predict_for_dataset(dataset)
    print("Predictions for the dataset:")
    print(predicted_dataset)
