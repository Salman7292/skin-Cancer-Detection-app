import requests
from io import BytesIO
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.image as mpimg
import pandas as pd
import json
import base64
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import pickle




st.set_page_config(
    page_icon="Logo4.png",
    page_title="Skin Cancer Detection | app",
    layout="wide"
)



data = pd.read_csv("Training Result.csv")

# Cache the model loading function
@st.cache_resource
def load_model_once():
    # Load the model from the pickle file
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to convert an image file to base64 encoding
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convert the local images to base64 encoding
background_img1 = get_img_as_base64("bg.jpg")  # Replace with your local image path





genai.configure(api_key="AIzaSyAkgJ60JabYJzxHTDqA_VwD6M_ptR0s5XU")



def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def to_markdown2(list1):
  list2=list()
  for i in list1:
    if "*" in i:
      newword=i.replace("*"," ")
      list2.append(newword)
    elif "-" in i:
        newword=i.replace("-"," ")
        list2.append(newword)



  return list2


model = genai.GenerativeModel('gemini-pro')

# genmini api


def generate_skin_cancer_info(skin_condition_label):
    prompt = f"""
    You are given the following skin condition label:

    {skin_condition_label}

    Your task is to provide detailed information about the skin condition associated with the label. Specifically, you need to:
    1. Describe the skin condition associated with the label.
    2. List and explain the common symptoms of this skin condition.
    3. Provide a summary of the best available treatments for this condition.
    4. Suggest references or specialists who are known for treating this specific skin condition.

    The skin condition labels are:
    - 'actinic keratosis'
    - 'basal cell carcinoma'
    - 'dermatofibroma'
    - 'melanoma'
    - 'nevus'
    - 'pigmented benign keratosis'
    - 'seborrheic keratosis'
    - 'squamous cell carcinoma'
    - 'vascular lesion'

    Based on the provided label, generate the following information:

    **Condition Description:**
    [Detailed description of the skin condition]

    **Symptoms:**
    [Detailed list of symptoms associated with the skin condition]

    **Treatments:**
    [Summary of the best treatments available for this condition]

    **Specialist References:**
    [Suggestions for specialists or institutions known for treating this condition]

    Ensure that:
    - The information is accurate and up-to-date.
    - Symptoms and treatments are described in detail.
    - Specialist recommendations are relevant and based on current medical practices.

    If the label provided is 'nevus' or 'dermatofibroma', explain that these are generally benign conditions, but monitoring for changes is recommended.

    If the label provided is 'melanoma' or 'squamous cell carcinoma', emphasize the seriousness of the condition and the need for immediate medical intervention.

    """
    response = model.generate_content(prompt)
    return response










# Assuming you have class labels in order (you can map these indices to actual labels)
class_labels = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']



def load_and_preprocess_image(img_path):
    # Load the image using PIL and resize it to (256, 256)
    img = Image.open(img_path).resize((256, 256))

    # Convert the PIL image to a NumPy array
    img_array = np.array(img)

    # If the image is in grayscale (1 channel), convert it to RGB (3 channels)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Add an extra dimension to represent batch size (1 image)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image using ResNet50's preprocessing
    img_array = preprocess_input(img_array)

    return img_array





# Step 2: Predict the class of the image
def predict_image_class(img_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)

    model=load_model_once()

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the index of the class with the highest score
    predicted_class = np.argmax(predictions, axis=1)

    # Get the confidence score for the predicted class
    confidence = np.max(predictions)

    return predicted_class, predictions, confidence














def image_preprocessing_from_url(image_url, target_size=(256, 256)):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Resize the image
        img = img.resize(target_size)

        # Convert the image to a numpy array
        img_array = np.array(img)

        # If the image is in grayscale (1 channel), convert it to RGB (3 channels)
        if img_array.ndim == 2 or img_array.shape[-1] != 3:
            img_array = np.stack([img_array] * 3, axis=-1)

        # Add batch dimension (1, 256, 256, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image for the model (if required, e.g., ResNet50)
        img_array = preprocess_input(img_array)

        return img_array
    
    except Exception as e:
        return f"An error occurred during image preprocessing: {e}"

# Step 2: Predict the class of the image from URL
def predict_image_class_from_url(image_url):
    # Preprocess the image from URL
    img_array = image_preprocessing_from_url(image_url)

    # Load the model (assuming there's a function to load it once)
    model = load_model_once()

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the index of the class with the highest score
    predicted_class = np.argmax(predictions, axis=1)

    # Get the confidence score for the predicted class
    confidence = np.max(predictions)

    return predicted_class, predictions, confidence








# Function to plot training and validation loss
def plot_training_validation_loss(df):
  
    st.markdown(
                """
                <h4 style='text-align: center;'>Training and Validation Loss</h4>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
            """
            <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
            """,
            unsafe_allow_html=True
        ) 
    # Assume the DataFrame has 'epoch', 'loss', 'val_loss' columns
    epochs = df.index  # Use df.index if 'epoch' is not present in the CSV
    loss = df['loss']
    val_loss = df['val_loss']

    # Create a figure for training and validation loss
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=loss, mode='lines', name='Training Loss', line=dict(color='Yellow')))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss', line=dict(color='red')))
    fig_loss.update_layout(
        
        xaxis_title='Epoch', 
        yaxis_title='Loss', 
        legend=dict(x=0.5, y=1.15, orientation='h'),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Remove background color
        paper_bgcolor='rgba(0, 0, 0, 0)'  # Remove the outer "paper" background
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig_loss, use_container_width=True)



# Function to plot training and validation accuracy
def plot_training_validation_accuracy(df):

    st.markdown(
                """
                <h4 style='text-align: center;'>Training and Validation Accuracy</h4>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
            """
            <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
            """,
            unsafe_allow_html=True
        ) 
    
    # Assume the DataFrame has 'epoch', 'accuracy', 'val_accuracy' columns
    epochs = df.index  # Use df.index if 'epoch' is not present in the CSV
    accuracy = df['accuracy']
    val_accuracy = df['val_accuracy']

    # Create a figure for training and validation accuracy
    fig_accuracy = go.Figure()

    # Add training accuracy trace
    fig_accuracy.add_trace(go.Scatter(
        x=epochs, y=accuracy, mode='lines', name='Training Accuracy', 
        line=dict(color='blue')
    ))

    # Add validation accuracy trace
    fig_accuracy.add_trace(go.Scatter(
        x=epochs, y=val_accuracy, mode='lines', name='Validation Accuracy', 
        line=dict(color='orange')
    ))

    # Update layout to set width, height, and enable thin white gridlines
    fig_accuracy.update_layout(
        xaxis_title='Epoch', 
        yaxis_title='Accuracy',
        legend=dict(x=0.5, y=1.15, orientation='h'),

        plot_bgcolor='rgba(0, 0, 0, 0)',  # Remove background color
        paper_bgcolor='rgba(0, 0, 0, 0)'  # Remove the outer "paper" background
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig_accuracy, use_container_width=True)


def training_data_graph():
    st.markdown(
                """
                <h4 style='text-align: center;'>Distribution of Training Data In Each Class</h4>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
            """
            <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
            """,
            unsafe_allow_html=True
        ) 
   
    # Data
    categories = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 
                'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 
                'squamous cell carcinoma', 'vascular lesion']
    values = [1995, 1997, 1991, 1998, 1999, 1996, 1999, 1994, 1992]

    # Create a DataFrame
    data = pd.DataFrame({
        'Categories': categories,
        'Number of Cases': values
    })

    # Create a bar chart using Plotly Express
    fig = px.bar(data, 
                x='Categories', 
                y='Number of Cases', 
                color='Categories', 
                color_discrete_sequence=[
                    '#FF6347', '#FFD700', '#ADFF2F', '#32CD32', 
                    '#00BFFF', '#FF69B4', '#FFA500', '#4B0082', 
                    '#00FA9A'
                ]
               )

    # Update layout for better readability
    fig.update_layout(
        yaxis_title='Number of Cases',
        xaxis_title='',  # Remove x-axis title
       
        plot_bgcolor='rgba(0,0,0,0)',  # Remove plot background
        paper_bgcolor='rgba(0,0,0,0)'   # Remove paper background
    )

    st.plotly_chart(fig)




def testing_data_graph():
    st.markdown(
                """
                <h4 style='text-align: center;'>Distribution of Testing Data In Each Class</h4>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
            """
            <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
            """,
            unsafe_allow_html=True
        ) 
    
    
    # Data for skin lesion categories and their corresponding case counts
    categories = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 
                  'nevus', 'pigmented benign keratosis', 'seborrheic keratosis', 
                  'squamous cell carcinoma', 'vascular lesion']
    case_counts = [999, 1000, 954, 954, 953, 959, 953, 953, 958]

    # Create a DataFrame
    data = pd.DataFrame({
        'Categories': categories,
        'Number of Cases': case_counts
    })

    # Create a bar chart using Plotly Express
    fig = px.bar(data, 
                  x='Categories', 
                  y='Number of Cases', 
                  color='Categories', 
                  color_discrete_sequence=[
                      'yellow', 'orange', 'lime', 'cyan', 
                      'magenta', 'lightcoral', 'lightblue', 
                      'pink', 'gold'
                  ]
                 )

    # Update layout for better readability
    fig.update_layout(
        yaxis_title='Number of Cases',
        xaxis_title='',  # Remove x-axis title
       
        plot_bgcolor='rgba(0,0,0,0)',  # Remove plot background
        paper_bgcolor='rgba(0,0,0,0)'   # Remove paper background
    )

    # Display the bar chart
    st.plotly_chart(fig, use_container_width=True)





# Function to display the confusion matrix graph
def confusion_matrix_graph():

    st.markdown(
                """
                <h4 style='text-align: center;'>Confusion matrix</h4>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
            """
            <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
            """,
            unsafe_allow_html=True
        ) 
    
    # Example confusion matrix data
    confusion_matrix = np.array([
        [766, 0, 0, 0, 33, 4, 0, 1, 0],  # Actinic keratosis
        [1, 715, 1, 2, 0, 6, 1, 6, 0],   # Basal cell carcinoma
        [0, 0, 709, 0, 0, 1, 0, 1, 0],   # Dermatofibroma
        [0, 2, 2, 632, 10, 7, 88, 1, 0], # Melanoma
        [104, 4, 1, 12, 644, 5, 0, 2, 0],# Nevus
        [8, 14, 3, 2, 3, 704, 1, 15, 3], # Pigmented benign keratosis
        [0, 0, 0, 24, 0, 0, 701, 0, 0],  # Seborrheic keratosis
        [0, 7, 0, 0, 3, 8, 0, 703, 0],   # Squamous cell carcinoma
        [0, 0, 0, 0, 0, 0, 0, 0, 701]    # Vascular lesion
    ])

    labels = ["Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma", "Melanoma", "Nevus", 
            "Pigmented Benign Keratosis", "Seborrheic Keratosis", "Squamous Cell Carcinoma", "Vascular Lesion"]


    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    # Plot the confusion matrix
    fig_cm = px.imshow(df_cm, text_auto=True, aspect="auto", color_continuous_scale="Blues")

    # Plot the confusion matrix with bright colors
    fig_cm = px.imshow(df_cm, 
                       text_auto=True, 
                       aspect="auto", 
                       color_continuous_scale=[
                           "yellow", "orange", "red", "pink", "magenta"
                       ])  # Brighter color scale

    fig_cm.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        font=dict(color='black'),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
        coloraxis_showscale=True  # Show the color scale
    )
    
    # Display confusion matrix in the app
    st.plotly_chart(fig_cm)


# Function to display confusion matrix and correct vs wrong predictions graph
def Model_Predictions_graph():

    st.markdown(
        """
        <h4 style='text-align: center;'>Model Predictions Detail</h4>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(255,0,0,1) 13%, rgba(255,165,0,1) 57%, rgba(255,255,0,1) 93%); margin: 0 auto;" />
        """,
        unsafe_allow_html=True
    )

    # Example confusion matrix data
    confusion_matrix = np.array([
        [766, 0, 0, 0, 33, 4, 0, 1, 0],  # Actinic keratosis
        [1, 715, 1, 2, 0, 6, 1, 6, 0],   # Basal cell carcinoma
        [0, 0, 709, 0, 0, 1, 0, 1, 0],   # Dermatofibroma
        [0, 2, 2, 632, 10, 7, 88, 1, 0], # Melanoma
        [104, 4, 1, 12, 644, 5, 0, 2, 0],# Nevus
        [8, 14, 3, 2, 3, 704, 1, 15, 3], # Pigmented benign keratosis
        [0, 0, 0, 24, 0, 0, 701, 0, 0],  # Seborrheic keratosis
        [0, 7, 0, 0, 3, 8, 0, 703, 0],   # Squamous cell carcinoma
        [0, 0, 0, 0, 0, 0, 0, 0, 701]    # Vascular lesion
    ])

    labels = ["Actinic Keratosis", "Basal Cell Carcinoma", "Dermatofibroma", "Melanoma", "Nevus", 
              "Pigmented Benign Keratosis", "Seborrheic Keratosis", "Squamous Cell Carcinoma", "Vascular Lesion"]

    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)



    # 1. Calculate the correct predictions (diagonal elements)
    correct_predictions = np.trace(confusion_matrix)

    # 2. Calculate the wrong predictions (total sum minus diagonal)
    total_predictions = np.sum(confusion_matrix)
    wrong_predictions = total_predictions - correct_predictions

    # 3. Create a bar chart for correct and wrong predictions
    data = {
        "Type": ["Correct Predictions", "Wrong Predictions"],
        "Count": [correct_predictions, wrong_predictions]
    }

    df = pd.DataFrame(data)

    # Plot the bar chart with bright colors
    fig_bar = px.bar(df, x="Type", y="Count",
                     color="Type", text_auto=True,
                     color_discrete_map={
                         "Correct Predictions": "Orange",  # Bright lime green for correct predictions
                         "Wrong Predictions": "magenta"       # Bright magenta for wrong predictions
                     }
                     )

    fig_bar.update_layout(
        xaxis_title="Prediction Type",
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
    )

    # Display the bar chart in the Streamlit app
    st.plotly_chart(fig_bar)







# Custom CSS to style the BMI result box
st.markdown("""
    <style>
.bmiresult {
    font-family: Arial, sans-serif;
    background-color: #021531;
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    text-align: center;
    width: 465px;
    /* HEIGHT: 328PX; */
    margin: auto;
}

.bmiresult h1 {
    margin: 0 0 10px;
    color: #333;
    font-size: 34px;
}

.bmiresult h3 {
    margin: 0;
    color: #07d6ef;
    font-size: 2em;
}
            


    .bmiresult p {
        margin-top: 10px;
        color: #333;
    }
            
      
    </style>
""", unsafe_allow_html=True)


# CSS styling for the Streamlit app
page_bg_img = f"""
<style>

[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/jpeg;base64,{background_img1}");

       
    background-size: 100%;
    background-position: top left;
    # background-repeat: no-repeat;
    background-attachment: local;
    # opacity: 0.3;
    # transition: opacity 2s ease-in-out; /* 2 seconds transition */
}}



[data-testid="stSidebar"] > div:first-child {{
    background-repeat: no-repeat;
    background-attachment: fixed;
    background: rgb(18 18 18 / 0%);
}}




.st-emotion-cache-1gv3huu {{
    position: relative;
    top: 2px;
    background-color: #000;
    z-index: 999991;
    min-width: 244px;
    max-width: 550px;
    transform: none;
    transition: transform 300ms, min-width 300ms, max-width 300ms;
}}

.st-emotion-cache-1jicfl2 {{
    width: 100%;
    padding: 2rem 2rem 4rem;
    min-width: auto;
    max-width: initial;

}}


.st-emotion-cache-4uzi61 {{
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background: rgb(240 242 246);
    box-shadow: 0 5px 8px #6c757d;
}}

.st-emotion-cache-1vt4y43 {{
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    COLOR: WHITE;
    user-select: none;
    background-color: #0461f1;
    border: 1px solid rgba(49, 51, 63, 0.2);
}}

.st-emotion-cache-qcpnpn {{
    border: 1px solid rgb(163, 168, 184);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);
    background-color: rgb(38, 39, 48);
    MARGIN-TOP: 9PX;
    box-shadow: 0 5px 8px #6c757d;
}}


.st-emotion-cache-15hul6a {{
    user-select: none;
    background-color: #ffc107;
    border: 1px solid rgba(250, 250, 250, 0.2);
    
}}

.st-emotion-cache-1hskohh {{
    margin: 0px;
    padding-right: 2.75rem;
    color: rgb(250, 250, 250);
    border-radius: 0.5rem;
    background: #000;
}}

.st-emotion-cache-12pd2es {{
    margin: 0px;
    padding-right: 2.75rem;
    color: #f0f2f6;
    border-radius: 0.5rem;
    background: #000;
}}

.st-emotion-cache-1r6slb0 {{
    width: calc(33.3333% - 1rem);
    flex: 1 1 calc(33.3333% - 1rem);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}}
.st-emotion-cache-12w0qpk {{
    width: calc(25% - 1rem);
    flex: 1 1 calc(25% - 1rem);
    display: flex;
    flex-direction: row;
    justify-content: CENTER;
    ALIGN-ITEMS: CENTER;
}}



.st-emotion-cache-1kyxreq {{
    display: flex;
    flex-flow: wrap;
    row-gap: 1rem;
    align-items: center;
    justify-content: center;
}}

img {{
    vertical-align: middle;
    border-radius: 10px;
 
}}


    h5 {{
    
    font-family: "math", sans-serif;
    font-weight: 600;
    color: rgb(255 255 255 / 72%);
    padding: 0px 0px 1rem;
    margin: 0px;
    line-height: 1.2;
}}




.st-emotion-cache-12fmjuu {{
background-color:#022c54
    # display: none;
}}


.st-emotion-cache-h4xjwg {{
background-color:#022c54

    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 3.75rem;
    background: #022c54;
    outline: none;
    z-index: 999990;
    display: block;
}}


        [data-testid=stSidebar] {{
            background-color: #050A30;
        }}



.st-cb {{
    # padding-top: 1rem;
    background: #075f868c;
    padding: 20px;
    # border-radius: 23px;
    margin-top: 10px;
    box-shadow: 0px 4px 4px white; /* Adds box shadow */
}}

.js-plotly-plot .plotly .user-select-none {{
   border:1px solid white;

   background: #2a391854;
    border-radius: 8px;
 box-shadow: 0 5px 32px rgb(22 131 144 / 70%);
}}






</style>
"""

# Apply CSS styling to the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # Display logo image
    st.image("Logo4.png", use_column_width=True)

    # Adding a custom style with HTML and CSS for sidebar
    st.markdown("""
        <style>
            .custom-text {
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                color:#ffc107
            }
            .custom-text span {
                color: #04ECF0; /* Color for the word 'Recommendation' */
            }
        </style>
    """, unsafe_allow_html=True)
  

    # Displaying the subheader with custom styling
    st.markdown('<p class="custom-text"> Skin Cancer <span>Detecter</span> App</p>', unsafe_allow_html=True)

    # HTML and CSS for the GitHub button
    github_button_html = """
    <div style="text-align: center; margin-top: 50px;">
        <a class="button" href="https://github.com/Salman7292" target="_blank" rel="noopener noreferrer">Visit my GitHub</a>
    </div>

    <style>
        /* Button styles */
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #ffc107;
            color: black;
            text-decoration: none;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #000345;
            color: white;
            text-decoration: none; /* Remove underline on hover */
        }
    </style>
    """

    # Display the GitHub button in the sidebar
    st.markdown(github_button_html, unsafe_allow_html=True)
    
    # Footer HTML and CSS
    footer_html = """
    <div style="padding:10px; text-align:center;margin-top: 10px;">
        <p style="font-size:20px; color:#ffffff;">Made with ❤️ by Salman Malik</p>
    </div>
    """

    # Display footer in the sidebar
    st.markdown(footer_html, unsafe_allow_html=True)


# Define the option menu for navigation
selections = option_menu(
    menu_title=None,
options = ['Home', 'Classify MRI Scan', 'Model Results', 'About Us', 'Contact Us'],
icons = ['house-fill', 'file-earmark-medical-fill', 'graph-up', 'info-circle-fill', 'telephone-fill'],



    menu_icon="cast",
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {
            "padding": "5px 23px",
            # "background-color": "#0d6efd",
            "background-color": "#022c54",
            "border-radius": "0px",
            "border-bottom":"2px solid white",
           
            "box-shadow": "0px 4px 10px rgba(0, 0, 0, 0.25)"
        },
        "icon": {"color": "#f9fafb", "font-size": "18px"},
        "hr": {"color": "#0d6dfdbe"},
        "nav-link": {
            "color": "#f9fafb",
            "font-size": "15px",
            "text-align": "center",
            "margin": "0 10px",
            "--hover-color": "#0761e97e",
            "padding": "10px 10px",
            "border-radius": "40px"
        },
        "nav-link-selected": {"background-color": "#ffc107", "font-size": "12px"},
    }
)

if selections == "Home":
# Define HTML and CSS for the hero section
    code = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Skin Cancer Detection App</title>
<style>
    .hero-section {
        padding: 60px 20px;
        text-align: center;
        # background: black;
        font-family: Arial, sans-serif;
    }



    .hero-heading {
        font-size: 2.5rem;
        margin-bottom: 20px;
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
    }



    .hero-text {
        font-size: 1.2rem;
        line-height: 1.6;
        color: #ffffffd6;
        max-width: 800px;
        margin: 0 auto;
    }

</style>
</head>
<body>
<section class="hero-section">
    <div class="container">
        <h1 class="hero-heading">Skin Cancer Detection Using Deep Learning</h1>
        <p class="hero-text">
            Welcome to our Skin Cancer Detection app. Effortlessly analyze skin lesions with advanced deep learning techniques to determine if they are benign or malignant. Simply upload an image of the affected area or provide a URL to receive quick and accurate results. Harness the power of AI to support early skin cancer diagnosis and improve patient outcomes, designed for use by both medical professionals and patients.
        </p>
    </div>
</section>
</body>
</html>

"""





# Use Streamlit to display the HTML content
    st.markdown(code, unsafe_allow_html=True)

elif selections == "Classify MRI Scan":
    st.markdown(
        """
        <h1 style='text-align: center;'>Insert Your Skin Effected Image Here</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 2px;width: 50%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
        """,
        unsafe_allow_html=True
    )


    Browes_file,Inesrt_from_url=st.tabs(["Browse File","Insert from URL"])

    with Browes_file:
        image = None  # Initialize image variable outside the expander

        with st.expander("Loading Image..."):
            # Image uploader widget
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

            # Check if an image has been uploaded
            if uploaded_image is not None:
                try:
                    # Open the image
                    image = Image.open(uploaded_image)
                    # Resize the image
                    max_size = (256, 256)  # Set the maximum width and height
                    image = image.resize(max_size, Image.Resampling.LANCZOS)
                    st.success("Image uploaded successfully!")

                except Exception as e:
                    st.error(f"An error occurred while uploading the image: {e}")

        # Display the uploaded image outside the expander
        if image is not None:

            st.image(image, caption="Uploaded Image", use_column_width=False)
         
            classify = st.button("Classify MRI")


            if classify:
                predicted_class, predictions, confidence=predict_image_class(uploaded_image)
                # Get the predicted class label
                predicted_label = class_labels[predicted_class[0]]
                
                rounded_confidence = str(round(round(confidence, 2)*100,2))
                st.markdown(f"""
                        <div class="bmiresult">
                            <h3>Info About Skin image</h3>
                            <h5>Cancer Type: {predicted_label}</h5>
                            <h5>Model confidence: {rounded_confidence+"%"}</h5>

                           


                        </div>
                    """, unsafe_allow_html=True)
                result=generate_skin_cancer_info(predicted_label)

                if result:  
                    st.title(f"Full Detail About {predicted_label}")
                    st.markdown(result.text)
                
        
    with Inesrt_from_url:
        url = st.text_input("Enter the image URL")

        if url:
            try:
                # Fetch the image from the URL
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))

                # Resize the image
                max_size = (256, 256)  # Set the maximum width and height
                image = image.resize(max_size, Image.Resampling.LANCZOS)
                st.success("Image loaded from URL successfully!")

                # Display the image and classification button
                st.image(image, caption="Image from URL", use_column_width=False)
                classify = st.button("Classify MRI",key="url")

                if classify:
                    predicted_class, predictions, confidence= predict_image_class_from_url(url)
                                   # Get the predicted class label
                    predicted_label = class_labels[predicted_class[0]]
                    rounded_confidence = str(round(round(confidence, 2)*100,2))
                    st.markdown(f"""
                            <div class="bmiresult">
                                <h3>Info About Skin image</h3>
                                <h5>Cancer Type: {predicted_label}</h5>
                                <h5>Model confidence: {rounded_confidence+"%"}</h5>

                            


                            </div>
                        """, unsafe_allow_html=True)
                    result=generate_skin_cancer_info(predicted_label)

                    if result:  
                        st.title(f"Full Detail About {predicted_label}")
                        st.markdown(result.text)


            except Exception as e:
                st.error(f"An error occurred while fetching the image from URL: {e}")


elif selections=="Model Results":
   
    st.markdown(
        """
        <h2 style='text-align: center;'>How Our AI Detects Skin Cancer? A Deep Dive into the Model</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
        """,
        unsafe_allow_html=True
    ) 
    st.markdown(
        """
        <h5 style='text-align: center;color:white;'>Our research is to use transfer learning and retrain pre-trained models like ResNet50 or VGG16 on medical data, which are already trained on general datasets, and then compare the performance of these models on medical data.</h5>
        """,
        unsafe_allow_html=True
    )

    ResNet50,VGG16 = st.tabs(["ResNet50","VGG16"])

    with ResNet50:
      
        st.markdown(
        """
        <h1 style='text-align: center;'>ResNet50</h1>
        """,
        unsafe_allow_html=True
    )

        st.markdown(
        """
        <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
        """,
        unsafe_allow_html=True
    ) 

        st.markdown(
        """
        <p style='color:white;'>Here 
            This model is designed for skin cancer detection using the pre-trained ResNet50 architecture, fine-tuned for this task. It processes images of size 256x256x3 and leverages pre-trained weights from ImageNet. The model excludes ResNet50's fully connected layers and adds custom layers for skin cancer classification, including Batch Normalization and Dropout to improve stability and reduce overfitting. The final layer classifies images into 9 categories. The last convolutional block is unfrozen to allow fine-tuning. It is trained using the Adam optimizer and categorical cross-entropy, with early stopping to prevent overfitting..</p>
        """,
        unsafe_allow_html=True
    )
        

        st.markdown(
                    """
                    <h3 style='text-align: center;'>How we Can Distribute Dataset into Training and Testing</h3>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
                """
                <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )       
        
        col1, col2 =st.columns(2)


        with col1:

            training_data_graph()
        
        with col2:
  
            testing_data_graph()

        



        st.markdown(
                    """
                    <h3 style='text-align: center;'>Our Model Training Result</h3>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
                """
                <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )  
        
        col3, col4 =st.columns(2)


        with col3:
            plot_training_validation_loss(data)
        
        with col4:
            plot_training_validation_accuracy(data)




        st.markdown(
                    """
                    <h3 style='text-align: center;'>Model Evaluation</h3>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
                """
                <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True
            )  
        
        col5, col6 =st.columns(2)


        with col5:
            confusion_matrix_graph()
        
        with col6:
            Model_Predictions_graph()
        



        
    with VGG16:
            st.write("VGG16")


elif selections=="Contact Us":



    st.header(":mailbox: Get In Touch With Us!")


    contact_form = """
    <form action="https://formsubmit.co/salmanuom04@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css("style/style.css")



elif selections=="About Us":

    st.markdown(
        """
        <h1 style='text-align: center;'>Meet Our Team</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <hr style="border: none; height: 2px;width: 80%; background: linear-gradient(90deg, rgba(255,0,0,1) 13%, rgba(255,165,0,1) 57%, rgba(255,255,0,1) 93%); margin: 0 auto;" />
        """,
        unsafe_allow_html=True
    )

    Mohib,Salman,Adnan,New=st.columns(4,vertical_alignment="center")


    with Mohib:
        st.image("mohib2 .png",use_column_width=True)
  
        st.markdown(
        """
        <h4 style='text-align: center;'>Mohib Wadood</h4>
        """,
        unsafe_allow_html=True
    ) 

        
    with Salman:
        st.image("Salman.png",use_column_width=True)
  
        st.markdown(
        """
        <h4 style='text-align: center;'>Salman Malik</h4>
        """,
        unsafe_allow_html=True
    ) 
    
    # with Adnan:
    #     st.image("salman.png",use_column_width=True)
  
    #     st.markdown(
    #     """
    #     <h4 style='text-align: center;'>Salman Malik</h4>
    #     """,
    #     unsafe_allow_html=True
    # )  



    Salman,Mohib=st.tabs(["Salman Malik","Mohib Wadood"])


    with Salman:
        image_Section,Title=st.columns(2)
        with image_Section:
            st.image("salman3.png",use_column_width=True)

        with Title:
            st.title(" ")
            st.title(" ")
            st.markdown(
        """
        <h1 style='text-align: center;'>Salman Malik</h1>
        """,
        unsafe_allow_html=True
    )
            st.markdown(
        """
        <h4 style='text-align: center;'>Python Programmer | Computer Vision</h4>
        """,
        unsafe_allow_html=True
    )
        st.markdown(
        """
        <h1 style='text-align: left;'>About Me</h1>
        """,
        unsafe_allow_html=True
        ) 
        st.markdown(
        """
        <hr style="border: none; height: 2px;width: 20%; background: linear-gradient(90deg, rgba(255,0,0,1) 13%, rgba(255,165,0,1) 57%, rgba(255,255,0,1) 93%); margin: 0 ;" />
        """,
        unsafe_allow_html=True
        ) 

        st.markdown(
        """
        <p style='color:white;'>I'm a dedicated Python programmer with expertise in developing applications using Streamlit. My passion lies in the fields of Machine Learning and Deep Learning, where I constantly explore and implement cutting-edge techniques. I specialize in predictive analytics and enjoy creating robust models to solve real-world problems. With a strong foundation in AI, I am committed to advancing my knowledge and contributing to the open-source community.</p>
        """,
        unsafe_allow_html=True
    )
           
            
                 

       


    with Mohib:
        image_Section,Title=st.columns(2)
        with image_Section:
            st.image("m.png",use_column_width=True)

        with Title:
            st.title(" ")
            st.title(" ")
            st.markdown(
        """
        <h1 style='text-align: center;'>Mohib Wadood</h1>
        """,
        unsafe_allow_html=True
    )
            st.markdown(
        """
        <h4 style='text-align: center;'>Machine Learning | WebDevolpment</h4>
        """,
        unsafe_allow_html=True
    )
        st.markdown(
        """
        <h1 style='text-align: left;'>About Me</h1>
        """,
        unsafe_allow_html=True
        ) 
        st.markdown(
        """
        <hr style="border: none; height: 2px;width: 20%; background: linear-gradient(90deg, rgba(255,0,0,1) 13%, rgba(255,165,0,1) 57%, rgba(255,255,0,1) 93%); margin: 0 ;" />
        """,
        unsafe_allow_html=True
        ) 

        st.markdown(
        """
        <p style='color:white;'>I'm a dedicated Python programmer with expertise in developing applications using Streamlit. My passion lies in the fields of Machine Learning and Deep Learning, where I constantly explore and implement cutting-edge techniques. I specialize in predictive analytics and enjoy creating robust models to solve real-world problems. With a strong foundation in AI, I am committed to advancing my knowledge and contributing to the open-source community.</p>
        """,
        unsafe_allow_html=True
    )
