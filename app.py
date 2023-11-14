import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as data
from keras.models import load_model 
import streamlit as st  
from datetime import datetime, timedelta


def main():
    start = datetime.now() - timedelta(days=5*365)
    end = datetime.now()
    user_input = st.text_input("Enter stock ticker")
    if user_input:
        # Perform validation or other checks on the ticker symbol
        if is_valid_ticker(user_input):
            # Display stock information or perform other actions
                df = yf.download(user_input, start, end)
                st.title("Stock Trend Prediction")
                #describing data
                st.subheader(f"Date from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
                st.write(df.describe())
                #Visualization 
                st.subheader('Closing price vs time chart with 100MA')
                ma100 = df.Close.rolling(100).mean()
                fig = plt.figure(figsize =(12, 6))
                plt.plot(ma100, 'r', label = 'ma100')
                plt.plot(df.Close, 'b', label = 'close')
                plt.legend()
                st.pyplot(fig)

                st.subheader('Closing price vs time chart with 200MA')
                ma100 = df.Close.rolling(100).mean()
                ma200 = df.Close.rolling(200).mean()
                fig = plt.figure(figsize =(12, 6))
                plt.plot(ma100, 'r', label = 'ma100')
                plt.plot(ma200, 'g', label = 'ma200')
                plt.plot(df.Close, 'b', label = 'close')
                plt.legend()
                st.pyplot(fig)

                data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
                data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): len(df)])

                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range = (0, 1))

                #splitting data into x_train and y_train
                x_train = []
                y_train = []

                data_training_array = scaler.fit_transform(data_training)

                for i in range(100, data_training_array.shape[0]):
                    x_train.append(data_training_array[i - 100:i])
                    y_train.append(data_training_array[i, 0])
                x_train, y_train = np.array(x_train), np.array(y_train)

                #Load my model 
                model = load_model('keras_model.h5')

                #Testing part
                past_100_days = data_training.tail(100)
                final_df= pd.concat([past_100_days, data_testing], ignore_index=True)
                input_data = scaler.fit_transform(final_df)

                x_test = []
                y_test = []
                x_test = []
                y_test = []
                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i - 100: i])
                    y_test.append(input_data[i, 0])

                x_test, y_test = np.array(x_test), np.array(y_test)
                y_predict = model.predict(x_test)

                #after model predict, take it back to scaled value
                scaler = scaler.scale_

                scale_factor = 1/scaler[0]
                y_predict = y_predict * scale_factor
                y_test = y_test * scale_factor

                #finalize the graph 

                st.subheader('predictions vs original')
                fig2 = plt.figure(figsize = (12, 6))
                plt.plot(y_test, 'b', label = 'Original Price')
                plt.plot(y_predict, 'r', label = 'Predicted Price')
                plt.xlabel('Time(Years)')
                plt.ylabel('Price(USD)')
                plt.legend()
                st.pyplot(fig2)
                from sklearn.metrics import r2_score
                accuracy = r2_score(y_test, y_predict)
                st.subheader("The accuracy of this model is {:.2f}".format(accuracy))
        else:
            st.markdown("Invalid ticker symbol. Please enter a valid symbol.")
    
def is_valid_ticker(ticker):
    # Perform your validation logic here, e.g., check if the ticker symbol exists
    return True
if __name__ == '__main__':
    main()