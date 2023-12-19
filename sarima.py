import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

def calculate_rmse(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def calculate_mape(y_true, y_pred):
    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    # Calculate MAPE
    mape = mae / y_true.abs().mean() * 100
    return mape

# Function to predict input using SARIMA
def predict_salary_increase_sarima(df, region, future_years):
    df_region = df[df['REGION'] == region]

    # # Code to convert 'YEAR' to datetime if needed
    # df_region['YEAR'] = pd.to_datetime(df_region['YEAR'])

    y = df_region['SALARY']

    # Fit SARIMA model
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Example SARIMA parameters
    results = model.fit()

    # Menghitung RMSE dan MAPE
    y_true = y
    y_pred = results.fittedvalues
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    # Membulatkan nilai RMSE dan MAPE
    rounded_rmse = round(rmse)  # Menggunakan 2 angka di belakang koma
    rounded_mape = round(mape, 2)  # Menggunakan 2 angka di belakang koma

    # Menampilkan hasil dengan pembulatan
    st.write(f'Perbedaan prediksi nominal sekitar Rp {rounded_rmse},00 dari nominal asli')
    st.write(f'Tingkat error prediksi: {rounded_mape}%')

    # Predict future values
    start_year = df_region['YEAR'].max() + 1
    end_year = start_year + future_years - 1
    future_years_list = pd.date_range(start=str(start_year), periods=future_years, freq='Y')
    predicted_salaries = results.get_forecast(steps=future_years).predicted_mean

    return future_years_list, predicted_salaries

# Function to predict using SARIMA for the 'INDONESIA' region
def predict_salary_increase_sarima_indonesia(df, future_years):
    df_indonesia = df[df['REGION'] == 'INDONESIA']
    y = df_indonesia['SALARY']

    # Fit SARIMA model
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Example SARIMA parameters
    results = model.fit()

    # Menghitung RMSE dan MAPE
    y_true = y
    y_pred = results.fittedvalues
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    # Membulatkan nilai RMSE dan MAPE
    rounded_rmse = round(rmse)  # Menggunakan 2 angka di belakang koma
    rounded_mape = round(mape, 2)  # Menggunakan 2 angka di belakang koma

    # Menampilkan hasil dengan pembulatan
    st.write(f'Perbedaan prediksi nominal sekitar Rp {rounded_rmse},00 dari nominal asli')
    st.write(f'Tingkat error prediksi: {rounded_mape}%')
    
    # Predict future values
    start_year = df_indonesia['YEAR'].max() + 1
    end_year = start_year + future_years - 1
    future_years_list = pd.date_range(start=str(start_year), periods=future_years, freq='Y')
    predicted_salaries = results.get_forecast(steps=future_years).predicted_mean

    return future_years_list, predicted_salaries


# Streamlit App
st.title('Prediksi Gaji UMR Indonesia')

# Input file CSV
uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('Data yang Diunggah:')
    st.write(df)

    # Input
    selected_region = st.selectbox('Pilih Kabupaten/Kota:', df['REGION'].unique())
    future_years = st.number_input('Masukkan Jumlah Tahun yang Ingin Diprediksi:', min_value=1, max_value=20, value=3)

    # Prediction based on user input
    if st.button('Prediksi Berdasarkan Kabupaten/Kota'):
        future_years_list, predicted_salaries = predict_salary_increase_sarima(df, selected_region, future_years)

        st.write(f'Prediksi Besarnya UMR di {selected_region} untuk {future_years} tahun ke depan:')
        prediction_df = pd.DataFrame({'Tahun': future_years_list, 'Prediksi UMR': predicted_salaries})
        # Konversi kolom 'Tahun' menjadi string
        prediction_df['Tahun'] = prediction_df['Tahun'].dt.strftime('%Y')

        st.write(prediction_df)

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(future_years_list, predicted_salaries, marker='o')
        ax.set_title(f'Prediksi Besarnya UMR di {selected_region}')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Prediksi UMR')
        st.pyplot(fig)

    # Prediction for 'INDONESIA' region
    if st.button('Prediksi UMR Rata-Rata INDONESIA'):
        future_years_list, predicted_salaries = predict_salary_increase_sarima_indonesia(df, future_years)

        st.write(f'Prediksi besarnya UMR Rata-Rata INDONESIA untuk {future_years} tahun ke depan')
        prediction_df = pd.DataFrame({'Tahun': future_years_list, 'Prediksi UMR Rata-Rata': predicted_salaries})
        # Konversi kolom 'Tahun' menjadi string
        prediction_df['Tahun'] = prediction_df['Tahun'].dt.strftime('%Y')
        
        st.write(prediction_df)

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(future_years_list, predicted_salaries, marker='o')
        ax.set_title('Prediksi Besarnya UMR Rata-Rata INDONESIA')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Prediksi UMR Rata-Rata')
        st.pyplot(fig)
