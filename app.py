import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
from streamlit_option_menu import option_menu
import base64
from pathlib import Path

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)


# UI part -->>
# create Streamlit app


if __name__ == "__main__":
    st.set_page_config(page_title="Option Menu", layout="wide")
    selected = option_menu(None, ["Home", "Summary", 'About'], 
        icons=['house', 'file-earmark-text', 'people'], 
        menu_icon="cast", default_index=0, orientation="horizontal")

    if selected == "Home":
        st.title("Credit Card Fraud Detection Model")
        st.subheader("Enter the following features to check if the transaction is legitimate or fraudulent:")

        # create input fields for user to enter feature values
        input_df = st.text_input('Input All features')
        input_df_lst = input_df.split(',')
        # create a button to submit input and get prediction
        submit = st.button("Submit")

        if submit:
            # get input feature values
            features = np.array(input_df_lst, dtype=np.float64)
            # make prediction
            prediction = model.predict(features.reshape(1,-1))
            # display result
            if prediction[0] == 0:
                st.success('Legitimate transaction', icon="✅")
            else:
                st.error('Fraudulent transaction', icon="🚨")

        # # Custom HTML and CSS for the footer
        # footer_style = f"""
        #     <style>
        #         .footer {{
        #             background: #ff4b4b;
        #             color: white;
        #             padding: 10px;
        #             position: fixed;
        #             bottom: 0;
        #             width: 90vw;
        #             text-align: center;
        #             font-size: 20px;
        #         }}
        #     </style>
        # """

        # # Display the custom HTML
        # st.markdown(footer_style, unsafe_allow_html=True)

        # # Your Streamlit app content goes here

        # # Display the footer
        # def img_to_bytes(img_path):
        #     img_bytes = Path(img_path).read_bytes()
        #     encoded = base64.b64encode(img_bytes).decode()
        #     return encoded
        # def img_to_html(img_path):
        #     img_html = "<img src='data:image/png;base64,{}' style='width: 100px; padding-left: 20px; padding-right: 20px' class='img-fluid'>".format(
        #     img_to_bytes(img_path)
        #     )
        #     return img_html

        # st.markdown(f"<div class='footer'>{img_to_html('assets/SGBAU.png')} Design and Developed By : Final Year Students, Computer Science and Engineering, H.V.P.M. Amravati. {img_to_html('assets/HVPM.png')}<div>", unsafe_allow_html=True)


        # st.image('assets/HVPM.png', width=100)
        # with st.markdown('<div class="footer">Design and Developed By : Final Year Students, Computer Science and Engineering, H.V.P.M. Amravati.</div>', unsafe_allow_html=True):
        #     pass
        # st.image('assets/SGBAU.png', width=100)


        
        
    if selected == "Summary":
        st.title("Credit Card Fraud Detection Abstract")
        st.subheader("Credit card fraud detection is presently the most frequently occurring problem in the present world. This is due to the rise in both online transactions and e-commerce platforms. Credit card fraud generally happens when the card was stolen for any of the unauthorized purposes or even when the fraudster uses the credit card information for his use. In the present world, we are facing a lot of credit card problems. To detect fraudulent activities the credit card fraud detection system was introduced. This project aims to focus mainly on machine learning algorithms. The algorithm used is the Logistic Regression. Logistic Regression is a popular machine learning algorithm used in various fields, including credit card fraud detection. In the context of credit card fraud detection, Logistic Regression can be employed to predict whether a given credit card transaction is fraudulent or not based on certain features. The methodology involves preprocessing the dataset to handle missing values, outliers, and scaling numerical features. The data is split into training and testing sets for model evaluation. The trained Logistic Regression model is assessed using metrics such as accuracy, precision, recall, F1 score, and the ROC-AUC curve. Feature importance analysis is conducted to understand the contribution of each feature in predicting fraud. The results demonstrate the effectiveness of Logistic Regression in credit card fraud detection, providing insights into its performance, interpretability, and potential for integration into broader fraud detection systems. The study concludes with recommendations for further research and the exploration of complementary techniques to enhance fraud detection capabilities. The study emphasizes the importance of continuous monitoring and model updates to adapt to evolving fraud patterns. The classification threshold is adjusted based on the specific requirements and costs associated with false positives and false negatives.")
    if selected == "About":
        st.title(f"Group Members")

        st.header("Samarpeet Nandanwar") 
        st.subheader("Roll No: 50, Final Year,  Computer Science & Engineering")
        st.divider()

        st.header("Adib Khan") 
        st.subheader("Roll No: 07, Final Year,  Computer Science & Engineering")
        st.divider()

        st.header("Suraj Pawar") 
        st.subheader("Roll No: 57, Final Year,  Computer Science & Engineering")
        st.divider()

        st.header("Deven Malekar") 
        st.subheader("Roll No: 14, Final Year,  Computer Science & Engineering")
        st.divider()

        st.header("Himanshu kadu") 
        st.subheader("Roll No: 22, Final Year,  Computer Science & Engineering")


    # Custom HTML and CSS for the footer
    footer_style = f"""
        <style>
            .footer {{
                background: #ff4b4b;
                color: white;
                padding: 10px;
                position: fixed;
                bottom: 0;
                width: 90vw;
                text-align: center;
                font-size: 20px;
            }}
        </style>
    """

    # Display the custom HTML
    st.markdown(footer_style, unsafe_allow_html=True)

    # Your Streamlit app content goes here

    # Display the footer
    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    def img_to_html(img_path):
        img_html = "<img src='data:image/png;base64,{}' style='width: 100px; padding-left: 20px; padding-right: 20px' class='img-fluid'>".format(
        img_to_bytes(img_path)
        )
        return img_html

    st.markdown(f"<div class='footer'>{img_to_html('assets/SGBAU.png')} Design and Developed By : Final Year Students, Computer Science and Engineering, H.V.P.M. Amravati. {img_to_html('assets/HVPM.png')}<div>", unsafe_allow_html=True)