# https://docs.streamlit.io/en/stable/
# Import libraries

# !pip install streamlit
import streamlit as st

# Import local libraries
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Change title tags 
# https://docs.streamlit.io/en/stable/api.html?highlight=beta_page_config#streamlit.beta_set_page_config
# streamlit.set_page_config(page_title=None, page_icon=None, layout='centered', initial_sidebar_state='auto'
st.set_page_config(page_title='Prediction Apps')

# Add the title 
st.title("Make Prediction Web Application")

# Add instructions 
st.text("Please, upload your file in the sidebar")

# Add instruction in sidebar 
st.sidebar.text("Upload file here..")

# Add uploader file 
uploadFile = st.sidebar.file_uploader("Upload File", type=[".xlsx"])

# Add user input
userToPredict = st.sidebar.number_input("Temperature to predict")

# Add button to analysis 
startAnalysis = st.sidebar.button("Start Analysis")

# If user click start the analysis 
if startAnalysis:
    ### Copy and paste code in the local jupyter notebook  ###

    # read dataset
    dataset = pd.read_excel(uploadFile)
    # print(dataset.shape)
    
    # Add the user message
    st.text("Here is a description of your dataset")
    
    # Add a general description 
    st.write(dataset.describe())
    
    # Add a user message
    st.info("A plot to see distribution")
    
    # Display scatter matrix
    scatter_matrix(
        dataset,
        diagonal="hist",
        figsize=(8,8)
    )
    
    # hide warning
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Display the plot
    st.pyplot()
    
    # Train model section
    # Get the data
    x = dataset.loc[:,['Temperature']] # Label
    y = dataset.loc[:,['Power Electric']]
    
    # Define model
    regressor = LinearRegression()
    
    # Train model
    regressor.fit(x, y)
    
    # native test
    # print("model is ok!")
    
    # Data predict temperature to power electric result
    xPredict = np.array([[
        # 20,
        userToPredict,
    ]])
    prediction = regressor.predict(xPredict)
    # print(prediction)
    
    # User message
    st.success("with temperature : " + str(xPredict[0][0]) + " The power electric generator is : " + str(prediction[0][0]))
    
    # Display the plot
    
    # User message
    st.info('You can see your data(blue), the model(red) and prediction(yellow)')
    
    # create the figure 
    fix, ax = plt.subplots()
    
    # Add label to the plot
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Power Electric')
    
    # Add datasets
    ax.scatter(x, y)
    
    # Plot the model
    ax.plot(x, regressor.intercept_[0] + regressor.coef_[0]*x, c='r')
    
    # Add the prediction model
    ax.scatter(xPredict, prediction, linewidths=12)
    
    # Display plot
    st.pyplot(fix)