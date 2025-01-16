import paho.mqtt.client as mqtt 
import numpy as np 
import pandas as pd 
import time 
import tensorflow as tf 
from tensorflow.keras.models import load_model 
import dash 
from dash import dcc, html 
from dash.dependencies import Input, Output 
import plotly.graph_objs as go 
 
# Load the pre-trained model for predictions 
model = load_model('my_model.h5')  # Load your trained Keras model 
 
# Variables to store real-time data 
V_GS = 0 
I_D = 0 
data = [] 
 
# MQTT setup to receive data 
def on_message(client, userdata, message): 
    global V_GS, I_D 
    if message.topic == "transistor/V_GS": 
        V_GS = float(message.payload.decode()) 
    elif message.topic == "transistor/I_D": 
        I_D = float(message.payload.decode()) 
        # Store the data for plotting and analysis 
        data.append({'V_GS': V_GS, 'I_D': I_D}) 
        print(f"Received V_GS: {V_GS}, I_D: {I_D}") 
 
client = mqtt.Client("Digital Twin")
client.on_message = on_message 
#client.connect("19990da2823846f68376aadb01a9948a.s1.eu.hivemq.cloud", 8884) 
client.connect("test.mosquitto.org", 1883)
client.subscribe("transistor/#") 
client.loop_start()

def adjust_vgs(predicted_id): 
    """ Adjust V_GS based on the predicted I_D to maintain desired behavior """ 
    # Example: if predicted I_D is too high, reduce V_GS to prevent overheating. 
    target_id = 1.5  # Desired target current in Amperes 
    adjustment = 0.1 * (target_id - predicted_id)  # Simple proportional control 
    new_vgs = V_GS + adjustment 
     
    # Send the new V_GS back to the physical system via MQTT 
    client.publish("transistor/V_GS_control", new_vgs) 
    print(f"Adjusting V_GS to: {new_vgs}")
 
# Step 3: Real-Time Visualization with Dash 
app = dash.Dash(__name__) 
app.layout = html.Div([ 
    dcc.Graph(id='live-update-graph'), 
    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0) 
]) 
 
@app.callback( 
    Output('live-update-graph', 'figure'), 
    [Input('interval-component', 'n_intervals')] 
) 
def update_graph_live(n):
    global data
    df = pd.DataFrame(data)
    if not df.empty:
        # Predict using the model for future behavior
        X = df[['V_GS', 'I_D']].values[-1].reshape(1, -1)  # Include both features
        predicted_I_D = model.predict(X)[0][0]  # Extract scalar prediction
        df['Predicted_I_D'] = model.predict(df[['V_GS', 'I_D']].values)

        # Create traces for actual and predicted I_D
        trace1 = go.Scatter(x=df['V_GS'], y=df['I_D'], mode='lines+markers', name='Actual I_D')
        trace2 = go.Scatter(x=df['V_GS'], y=df['Predicted_I_D'], mode='lines', name='Predicted I_D', line=dict(dash='dash'))

        return {
            'data': [trace1, trace2],
            'layout': go.Layout(title='Real-Time Transistor Behavior', xaxis={'title': 'V_GS (V)'}, yaxis={'title': 'I_D (A)'})
        }
    return {'data': [], 'layout': go.Layout(title='Waiting for Data...')}
 
if __name__ == '__main__': 
    app.run_server(debug=True)