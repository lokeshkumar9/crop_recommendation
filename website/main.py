from flask import Flask, redirect, url_for, render_template, request, json,Response
import http.client  # Use the http.client library instead of requests
import pandas as pd
import numpy as np
import random
import base64
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from urllib.parse import quote
import io
from io import BytesIO

app = Flask(__name__)
app.secret_key = "Teams"
app.static_folder = 'static'
stateph="Andhra Pradesh"

# OpenWeatherMap API key
api_key = "7786359001637c3cbd8b1d24c68636f8"

# Function to get latitude and longitude using GeoCode API
def get_lat_lon(city, state, country):
    connection = http.client.HTTPSConnection("api.openweathermap.org", timeout=10)  # Setting a timeout of 10 seconds
    location = quote(f"{state},{city},{country}")
    endpoint = f"/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
    
    try:
        connection.request("GET", endpoint)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        data = json.loads(data.decode("utf-8"))
        if data:
            lat = data[0]['lat']
            lon = data[0]['lon']
            return lat, lon
        else:
            return None
    except:
        return None

# Function to get weather data using latitude and longitude
def get_weather_data(lat, lon):
    connection = http.client.HTTPSConnection("api.openweathermap.org", timeout=10)  # Setting a timeout of 10 seconds
    endpoint = f"/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    
    try:
        connection.request("GET", endpoint)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        data = json.loads(data.decode("utf-8"))
        if data and 'main' in data:
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            temperature = temperature - 273
            return temperature, humidity
        else:
            return None
    except:
        return None
    
def get_rainfall(state_name):
    d = pd.read_csv('data/rainfall_statewise.csv')
    average = d.loc[d['State'] == state_name, ['2020', '2021', '2022']].mean().mean()
    average /= 12
    return average

@app.route("/")
def home():
    return render_template("index.html",cont="User")

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    if request.method == 'POST':
        stateph = request.form.get('state', 'Andhra Pradesh')  # Get state name from form data or default to 'Andhra Pradesh'

        # Call the function to generate plot as base64 string
        img_str = generate_plot(stateph)
        phimg_str=generate_plot2(stateph)

        # Return the rendered template with the state name and image string
        return render_template('graph.html', state_name=stateph, img_str=img_str,img_str2=phimg_str)
    return render_template('graph.html', state_name=None)

def generate_plot(state_name):
    # Load the data from CSV
    df = pd.read_csv('data/statesrainfall.csv')
    # Filter the dataframe for the given state
    state_data = df[df['state'] == state_name]

    if state_data.empty:
        print(f"No data found for the state: {state_name}")
        return None

    # Set the years which are also the column names from 2001 to 2010
    years = [str(year) for year in range(1960, 2018)]

    # Extract the pH values for these years
    ph_values = state_data[years].values.flatten()

    # Plotting the pH values
    plt.figure(figsize=(10, 5))
    plt.plot(years, ph_values, marker='o')
    plt.title(f'Rainfall values(mm) from 1960 to 2018 for {state_name}')
    plt.xlabel('Year')
    plt.ylabel('Rainfall(in mm)')
    plt.grid(True)
    plt.xticks(rotation=270)
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')

    plt.close()  # Close the plot to free up memory

    return img_str

def generate_plot2(state_name):
    # Load the data from CSV
    df = pd.read_csv('data/statesph.csv')
    # Filter the dataframe for the given state
    state_data = df[df['state'] == state_name]

    if state_data.empty:
        print(f"No data found for the state: {state_name}")
        return None

    # Set the years which are also the column names from 2001 to 2021
    years = [str(year) for year in range(2001, 2021)]

    # Extract the pH values for these years
    ph_values = state_data[years].values.flatten()

    # Plotting the pH values
    plt.figure(figsize=(10, 5))
    plt.plot(years, ph_values, marker='o')
    plt.title(f'ph values values from 2001 to 2020 for {state_name}')
    plt.xlabel('Year')
    plt.ylabel('ph values')
    plt.grid(True)

    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')

    plt.close()  # Close the plot to free up memory

    return img_str

@app.route("/login",methods=['GET','POST'])
def login():
    if request.method=='POST':
        state_code = request.form['state_code']
        str=state_code
        city_name = request.form['city_name']
        season = request.form['season']
        state_abb=pd.read_csv('data/states.csv')
        state_list=state_abb[state_abb['State'] == state_code]
        ph_value = pd.read_csv('data/ph_values.csv')
        state_name = state_code.upper()
        state_data = ph_value[ph_value['STATE'] == state_name]
        mean_ph_min = state_data['ph_min'].mean()
        mean_ph_max = state_data['ph_max'].mean()
        PH= (mean_ph_min + mean_ph_max) / 2  
        if abs(PH - 0) <=0:
           PH = random.uniform(5, 9)
        
        p=36.17962
        k=48.92058+(3.98266)*PH
        n= random.uniform(35, 95)
        N = round(n, 2)
        P = round(p, 2)
        K = round(k, 2)
        country_code = 'IN'
        state_code= 'AP'
        if not state_list.empty and not state_list['Abb'].isnull().all():
            state_code = state_list['Abb'].values[0]
        new_string = city_name.split()
        city_name = ''.join(new_string)
        lat_lon = get_lat_lon(city_name, state_code, country_code)
        Temperature = 17
        Humidity = 60
        Rainfall = 150
        phlevel='Acidic'

        if lat_lon:
            latitude, longitude = lat_lon
            weather_data = get_weather_data(latitude, longitude)
            if weather_data:
                Temperature, Humidity= weather_data
                Rainfall=get_rainfall(state_code)

        
        # object = Crop.classifier.predict(np.array([N,P,K,Humidity,PH,Rainfall]))
        # model=pickle.load(open("model.pkl",'rb'))
        data= pd.read_csv('data/Crop_recommendation.csv')
        inputs = data.drop('label',axis='columns')
        target = data['label']
        X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)
        model = tree.DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if np.isnan(K):
            K = np.random.randint(50, 71)
        if np.isnan(Rainfall):
            Rainfall=np.random.randint(70,180)
            
        predict1=model.predict(np.array([N, P, K, Temperature, Humidity, PH, Rainfall]).reshape(1,-1) )
        dfd = pd.read_csv('data/stateandcrops.csv')
        
        crop_name = None
        if season == 'Rabi':
            crops = dfd.loc[dfd['states'] == str].iloc[:, 1:4].values.flatten()
        elif season == 'Kharif':
            crops = dfd.loc[dfd['states'] == str].iloc[:, 4:7].values.flatten()
        elif season == 'Zaid':
            crops = dfd.loc[dfd['states'] == str].iloc[:, 7:].values.flatten()
        else:
            return None
        
        crop_name = random.choice(crops)
        
        if crop_name is None:
            crop_name=predict1[0]

        if float(Humidity) >=1 and float(Humidity)<= 33 : 
            humidity_level = 'Low Humid'
        elif float(Humidity) >=34 and float(Humidity) <= 66:
            humidity_level = 'Medium Humid'
        else:
            humidity_level = 'High Humid'

        if float(Temperature) >= 0 and float(Temperature)<= 6:
            temperature_level = 'Cool'
        elif float(Temperature) >=7 and float(Temperature):
            temperature_level = 'Warm'
        else:
            temperature_level= 'Hot' 

        if float(Rainfall) >=1 and float(Rainfall) <= 100:
            rainfall_level = 'Less'
        elif float(Rainfall) >= 101 and float(Rainfall) <=200:
            rainfall_level = 'Moderate'
        elif float(Rainfall) >=201:
            rainfall_level = 'Heavy Rain'
        else:
            rainfall_level = 'No Rain'
        N_level='Less'
        P_level='Less'
        potassium_level='Less'

        if float(N) >= 1 and float(N) <= 50: 
            N_level = 'Less'
        elif float(N) >=51 and float(N) <=100:
            N_level = 'Not too less and Not to High'
        elif float(N) >=101:
            N_level = 'High'

        if float(P) >= 1 and float(P) <= 50:
            P_level = 'Less'
        elif float(P) >= 51 and float(P) <=100:
            P_level = 'Not too less and Not to High'
        elif float(P) >=101:
            P_level = 'High'

        if float(K) >= 1 and float(K) <=50: 
            potassium_level = 'Less'
        elif float(K) >= 51 and float(K) <= 100:
            potassium_level = 'Not too less and Not to High'
        elif float(K) >=101:
            potassium_level = 'High'

        if float(PH) >=0 and float(PH) <=5:             
            phlevel = 'Acidic' 
        elif float(PH) >= 6 and float(PH) <= 8:
            phlevel = 'Neutral'
        elif float(PH) >= 9 and float(PH) <= 14:
            phlevel = 'Alkaline'
        # return render_template("index.html",cont=[crop_name,humidity_level,temperature_level,rainfall_level,N_level,P_level,potassium_level,phlevel])
        
        PH=random.uniform(5,7)
        return render_template("Display.html",cont=[N_level,P_level,potassium_level,humidity_level,temperature_level,rainfall_level,phlevel],values=[N,P,K,Humidity,Temperature,Rainfall,PH],cropName=crop_name,st_name=str,season_name=season,district_name=city_name)

    return render_template("index.html")

@app.route("/user/<usr>")
def user(usr):
    return f"<h1> Hi {usr} !</h1>"

# @app.route("/user")
# def user():
#     if "user" in session:
#         user=session['user']
#         return f"<h1> hi {user} </h1>"
#     return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True, port=9090)
  