import streamlit as st
from openai import OpenAI



steps_eda = "LSTM is a type of recurrent neural network (RNN) that is well-suited for time series data. It is a popular algorithm for forecasting and classification tasks. Here are the steps to implement LSTM for time series forecasting:" \
    "1. Preprocess the data: Split the data into training and testing sets, scale the features, and handle missing values if necessary." \
    "2. Build the LSTM model: Choose the optimal number of hidden layers, hidden units, and dropout rate. Use an appropriate loss function and optimizer for training the model." \
    "3. Train the model: Fit the model on the training data, monitor the validation set performance, and adjust the hyperparameters if necessary." \
    "4. Make predictions: Use the trained model to make predictions on new data points." \
    "5. Evaluate the model: Calculate the accuracy, precision, recall, and other metrics to evaluate the model's performance." \
    "6. Tune the hyperparameters: Try different combinations of hyperparameters to optimize the model's performance." \
    "7. Deploy the model: Use a production-grade framework like TensorFlow or PyTorch to deploy the trained model for real-time predictions."
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    with st.expander("What are the steps of LSTM"):
        st.caption(steps_eda)
    user_csv = st.file_uploader("Upload your file here!", type="csv")
    st.divider()
    st.caption('<p style="text-align:center">made with ❤️ by Yuan</p>', unsafe_allow_html=True)


st.title("AI Assistant for Data Science💬")
st.write("Hello 👋🏻 I am your AI Assistant, and I am here to help you with your data science problems.")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    client = OpenAI(api_key=openai_api_key)

    if user_csv is not None:
        # Add the prompt to the list of messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        data_description = "The uploaded CSV file has the following columns: " + ", ".join(df.columns) + "."
        # Update prompt with data description
        prompt_with_data = f"{data_description} {prompt}"

        # Generate response from OpenAI model
        response = client.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt_with_data,
            max_tokens=150
        )

        # Extract the content from the response
        msg = response.choices[0].text.strip()
    else:
        # If there is no CSV, just proceed with the normal conversation flow
        response = client.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150
        )
        msg = response.choices[0].text.strip()

    # Append AI's message to the session state
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)