import streamlit as st
import pandas as pd
import pickle
import os

# Initialize session state for storing the previous prediction
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False  # To check if prediction has been made

# Load the model
model_path = './lagos_pred_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error(f"Model file '{model_path}' not found. Please ensure the file is uploaded correctly.")
    model = None

# Streamlit app
st.title('Lagos Apartment Pulse')

# Sidebar for additional information and feedback
st.sidebar.header('Welcome to Lagos Apartment Pulse')
st.sidebar.write("Hi! Lagos Apartment Pulse is a web app where you can predict the estimated price of apartments in cities all over Lagos based on the features you want.")
st.sidebar.write("This app is brought to you by St. Mark Adebayo.")
st.sidebar.write("Did you find this app useful? Share your experience with a comment below:")

# Feedback form
with st.sidebar.form(key='feedback_form'):
    name = st.text_input('Your Name')
    message = st.text_area('Your Comment')
    submit_button = st.form_submit_button(label='Send')
    if submit_button:
        with open('feedback.txt', 'a') as f:
            f.write(f"Name: {name}\nComment: {message}\n\n")
        st.sidebar.write("Thank you for your feedback!")

# Define location and title clusters
location_clusters = [
    'Abule Egba', 'Agege', 'Ajah', 'Ajao Estate', 'Alimosho', 'Apapa', 'Gbagada', 'Ikeja', 
    'Ikoyi', 'Ipaja', 'Lekki', 'Ojo', 'Ogba', 'Oshodi', 'Shomolu', 'Sangotedo', 
    'Surulere', 'Victoria Island', 'Yaba'
]
title_clusters = [
    'Duplex', 'Detached Duplex', 'Semi Detached Duplex', 'Flat', 'Mini Flat', 
    'Apartment', 'Self Con', 'Terrace', 'Terrace Duplex Detached', 'Room And Parlour', 
    'Penthouse Apartment', 'Studio Apartment', 'Maisonette', 'House', 'Terrace Duplex'
]

# Side-by-side layout for Location and Transaction Type
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox('Location', location_clusters, key='location')

with col2:
    transaction_type = st.radio('Transaction Type', ['Rent', 'Purchase'], key='transaction_type')

# Side-by-side layout for Bedrooms and Title
col3, col4 = st.columns(2)

with col3:
    bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, value=1, key='bedrooms')

with col4:
    title = st.selectbox('Title', title_clusters, key='title')

# Encode transaction type
transaction_encoded = 1 if transaction_type == 'Rent' else 0

# Instead of one-hot encoding, just use the index position of the selected location and title
location_encoded = location_clusters.index(location)
title_encoded = title_clusters.index(title)

# Prepare the input data with the exact 5 features the model expects
input_data = {
    'bedrooms': [bedrooms],
    'transaction_type': [transaction_encoded],
    'location_encoded': [location_encoded],  # Assuming this is how location was encoded
    'title_encoded': [title_encoded],         # Assuming this is how title was encoded
    'missing_feature': [0]  # Replace with appropriate value if needed
}

# Reorder the columns to match the model's expected input format
input_df = pd.DataFrame(input_data)

# Predict button
if st.button('Predict Price'):
    if model:
        try:
            # Make prediction
            prediction = model.predict(input_df)
            price = prediction[0]
            formatted_price = f"{price:,.1f}"
            
            if transaction_type == 'Rent':
                rent = round((float(formatted_price) / 20), 1)
                result = f"A {bedrooms}-bedroom {title} in {location} is estimated to be around ₦{rent} million/year."
            else:  # Purchase
                result = f"A {bedrooms}-bedroom {title} detached in {location} is estimated to be around ₦{formatted_price} million."
            
            # Store the state to indicate a prediction has been made
            st.session_state.prediction_done = True
            
            # Display the result in a box with dark background and white text
            st.markdown(
                f"""
                <div style="padding: 10px; border: 1px solid #444; border-radius: 5px; background-color: #333;">
                    <p style="font-size: 18px; font-weight: bold; color: white; text-align: center;">
                        {result}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        except ValueError as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Model is not loaded. Please upload the model file.")

# Clear the prediction state if any of the input features are changed
if st.session_state.prediction_done:
    # If any of the inputs are changed, reset the prediction state
    if st.session_state.location != location or st.session_state.title != title or \
       st.session_state.bedrooms != bedrooms or st.session_state.transaction_type != transaction_type:
        st.session_state.prediction_done = False
