import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle

# Loading our pre-trained model
try:
    with open("FIFA_Rating_Generator.pkl", 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Loading the scaler
try:
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    st.error(f"An error occurred while loading the scaler: {str(e)}")

# Creating a Streamlit app
st.title('FIFA Player Rating Prediction')
st.write('Enter player information below:')

# Numeric columns
feature1 = st.slider('release_clause_eur', min_value=0.0, max_value=100.0, step=0.1)
feature2 = st.slider('value_eur', min_value=0.0, max_value=100.0, step=0.1)
feature3 = st.slider('movement_reactions', min_value=0.0, max_value=100.0, step=0.1)
feature4 = st.slider('wage_eur', min_value=0.0, max_value=100.0, step=0.1)
feature5 = st.slider('mentality_composure', min_value=0.0, max_value=100.0, step=0.1)
feature6 = st.slider('passing', min_value=0.0, max_value=100.0, step=0.1)
feature7 = st.slider('dribbling', min_value=0.0, max_value=100.0, step=0.1)
feature8 = st.slider('potential', min_value=0.0, max_value=100.0, step=0.1)
feature9 = st.slider('mentality_vision', min_value=0.0, max_value=100.0, step=0.1)
feature10 = st.slider('age', min_value=0.0, max_value=100.0, step=0.1)
feature11 = st.slider('shooting', min_value=0.0, max_value=100.0, step=0.1)
feature12 = st.slider('mentality_positioning', min_value=0.0, max_value=100.0, step=0.1)
feature13 = st.slider('movement_agility', min_value=0.0, max_value=100.0, step=0.1)
feature14 = st.slider('power_stamina', min_value=0.0, max_value=100.0, step=0.1)

# Predicting player's Overall stat
if st.button('Predict'):
    data = [
        feature1, feature2, feature3, feature4, 
        feature5, feature6, feature7, feature8,
        feature9, feature10, feature11, feature12, 
        feature13, feature14
    ]
    
    # Apply the scaler to the data
    scaled_data = scaler.transform([data])

    # Make the prediction using the loaded model
    prediction = model.predict([scaled_data])[0]

    st.write(f'Predicted Rating: {prediction:.2f}')
