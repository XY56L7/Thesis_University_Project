import os
from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from .serializer import EnergyAnalysisSerializer, RegisterSerializer, UserSerializer
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import numpy as np
import pandas as pd
import json
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import random
import matplotlib
import pickle
import torch
import torch.nn as nn
from asgiref.sync import sync_to_async
matplotlib.use('Agg')

BASE_DIR = settings.BASE_DIR 
MEDIA_DIR = os.path.join(BASE_DIR, 'api', 'media')

@csrf_exempt
@require_http_methods(['POST'])
async def predict_communities_gru(request):
    try:
        data = json.loads(request.body)
        print("Parsed data:", data)

        required_columns = [
            'production', 'number_of_panels', 'panel_area_m2', 'category', 'air_temp', 
            'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'clearsky_gti', 
            'cloud_opacity', 'dhi', 'dni', 'ghi', 'gti', 
            'snow_soiling_rooftop', 'snow_soiling_ground', 'season'
        ]

        df = pd.DataFrame(data)

        missing_cols = [col for col in required_columns if col not in df.columns]

        if (missing_cols):
            return JsonResponse({'error': 'Missing required input fields'}, status=400)
        
        df['production'] = df['production'].astype(int)
        df['number_of_panels'] = df['number_of_panels'].astype(int)
        df['panel_area_m2'] = df['panel_area_m2'].astype(float)
        df['category'] = df['category'].astype(str)
        df['air_temp'] = df['air_temp'].astype(float)
        df['clearsky_dhi'] = df['clearsky_dhi'].astype(float)
        df['clearsky_dni'] = df['clearsky_dni'].astype(float)
        df['clearsky_ghi'] = df['clearsky_ghi'].astype(float)
        df['clearsky_gti'] = df['clearsky_gti'].astype(float)
        df['cloud_opacity'] = df['cloud_opacity'].astype(float)
        df['dhi'] = df['dhi'].astype(float)
        df['ghi'] = df['ghi'].astype(float)
        df['dni'] = df['dni'].astype(float)
        df['gti'] = df['gti'].astype(float)
        df['snow_soiling_rooftop'] = df['snow_soiling_rooftop'].astype(float)
        df['snow_soiling_ground'] = df['snow_soiling_ground'].astype(float)
        df['season'] = df['season'].astype(str)

        predicted = await sync_to_async(predict_form_gru)(df)

        MODEL_PATH = os.path.join(BASE_DIR, 'api', 'models', 'communities_GRU', 'Communities_GRU.pth')

        loaded_model = SolarGRU()
        state_dict = torch.load(MODEL_PATH, map_location='cpu')

        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()

        device = torch.device("cpu")

        input_tensor = torch.tensor(predicted, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction_norm = loaded_model(input_tensor).cpu().numpy()

        Y_SCALER = os.path.join(BASE_DIR, 'api', 'models', 'communities_GRU', 'GRU_scaler_y.save')

        y_scaler = joblib.load(Y_SCALER)

        prediction_real = y_scaler.inverse_transform(prediction_norm)
        
        result = max(0.0, float(prediction_real[0][0]))

        response_data = {
            'predicted_production': float(result)
        }

        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    except ValueError as e:
        return JsonResponse({'error': f'Invalid input data: {str(e)}'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@require_http_methods(['POST'])
async def predict_communities(request):
    try:
        data = json.loads(request.body)
        print("Parsed data:", data)

        if ('Timestamp' not in data or 'Number_of_panels' not in data or 'Panel_area_m2' not in data or 'Category' not in data or
             'Consumption' not in data or 'Air_temp' not in data or 'Clearsky_dhi' not in data or 'Clearsky_dni' not in data or
             'Clearsky_ghi' not in data or 'Clearsky_gti' not in data or 'Cloud_opacity' not in data or 'Dhi' not in data or
             'Dni' not in data or 'Ghi' not in data or 'Gti' not in data or 'Snow_soiling_rooftop' not in data or 
             'Snow_soiling_ground' not in data or 'Season' not in data):
            return JsonResponse({'error': 'Missing required input fields'}, status=400)

        timestamp = pd.to_datetime(data.get('Timestamp'))
        number_of_panels = int(data.get('Number_of_panels'))
        panel_area_m2 = float(data.get('Panel_area_m2'))
        category = str(data.get('Category'))
        consumption = float(data.get('Consumption'))
        air_temp = float(data.get('Air_temp'))
        clearsky_dhi = float(data.get('Clearsky_dhi'))
        clearsky_dni = float(data.get('Clearsky_dni'))
        clearsky_ghi = float(data.get('Clearsky_ghi'))
        clearsky_gti = float(data.get('Clearsky_gti'))
        cloud_opacity = float(data.get('Cloud_opacity'))
        dhi = float(data.get('Dhi'))
        ghi = float(data.get('Ghi'))
        dni = float(data.get('Dni'))
        gti = float(data.get('Gti'))
        snow_soiling_rooftop = float(data.get('Snow_soiling_rooftop'))
        snow_soiling_ground = float(data.get('Snow_soiling_ground'))
        season = str(data.get('Season'))

        predicted = await sync_to_async(predict_form)(timestamp, number_of_panels, panel_area_m2, category, consumption, air_temp,
                                                                       clearsky_dhi, clearsky_dni, clearsky_ghi, clearsky_gti, cloud_opacity, dhi, dni,
                                                                       ghi, gti, snow_soiling_rooftop, snow_soiling_ground, season)
        
        MODEL_PATH = os.path.join(BASE_DIR, 'api', 'models', 'communities', 'reg.sav')
        loaded_model = pickle.load(open(MODEL_PATH, "rb"))

        input_data_as_numpy_array = np.asarray(predicted)
        predicted_production = loaded_model.predict(input_data_as_numpy_array)

        response_data = {
            'predicted_production': float(predicted_production[0])
        }

        return JsonResponse(response_data)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    except ValueError as e:
        return JsonResponse({'error': f'Invalid input data: {str(e)}'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@require_http_methods(["POST"])
async def evaluate_model(request):
    try:        
        data = json.loads(request.body)
        print("Parsed data:", data)
        
        if 'V_rms' not in data or 'I_rms' not in data or 'S' not in data:
            return JsonResponse({'error': 'Missing required input fields'}, status=400)

        v_rms = float(data.get('V_rms'))
        i_rms = float(data.get('I_rms'))
        s = float(data.get('S'))
        device = str(data.get('Device'))

        predicted_power = await sync_to_async(predict_power_consumption)(v_rms, i_rms, s,device)

        fig, ax = plt.subplots()
        ax.plot([v_rms, i_rms, s], label='Input Data')
        ax.set_title('Predicted Power Consumption')
        ax.set_xlabel('Features')
        ax.set_ylabel('Value')
        ax.legend()

        random_number = random.randint(0, 10000)
        image_filename = f'prediction_plot{random_number}.png'
        image_path = os.path.join(MEDIA_DIR, image_filename)

        image_path = os.path.normpath(image_path)

        plt.savefig(image_path)
        plt.close(fig)

        response_data = {
            'image_path': request.build_absolute_uri('/api' + settings.MEDIA_URL + image_filename),
            'predicted_power': predicted_power
        }

        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    except ValueError as e:
        return JsonResponse({'error': f'Invalid input data: {str(e)}'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def protected(request):
    return Response({'message': 'Ez egy védett endpoint!'})

@api_view(['POST'])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def login(request):
    from django.contrib.auth import authenticate
    from rest_framework_simplejwt.tokens import RefreshToken

    username = request.data.get('username')
    password = request.data.get('password')

    user = authenticate(username=username, password=password)

    if user:
        refresh = RefreshToken.for_user(user)
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'userName': username
        })
    return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

@csrf_exempt
def predict_energy(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            year = data.get('year')
            month = data.get('month')
            day = data.get('day')
            hour = data.get('hour')
            number_of_panels = data.get('number_of_panels')
            season = data.get('season')  
            category = data.get('category') 

            season_encoded = encode_season(season)
            category_encoded = encode_category(category)

            features = [
                year, month, day, hour, number_of_panels, *season_encoded, *category_encoded
            ]

            production = predict_production(features)
            consumption = predict_consumption(features)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(['Production', 'Consumption'], [production, consumption], marker='o', label='Predictions')
            ax.set_title('Energy Predictions')
            ax.set_ylabel('Power (kW)')
            ax.set_xlabel('Category')
            ax.legend()
            ax.grid(True)

            random_number = random.randint(0, 10000)
            image_filename = f'energy_predictions_{random_number}.png'
            image_path = os.path.join(MEDIA_DIR, image_filename)
            plt.savefig(image_path)
            plt.close(fig)

            return JsonResponse({
                'production_power': production,
                'consumption_power': consumption,
                'plot_url': request.build_absolute_uri('/api' + settings.MEDIA_URL + image_filename)
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def predict_production(features):
    model_path = os.path.join(settings.BASE_DIR, 'api', 'models', 'production', 'production_model.keras')
    scaler_path = os.path.join(settings.BASE_DIR, 'api', 'models', 'production', 'scaler.pkl')
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)
    return float(prediction[0][0])

def predict_consumption(features):
    model_path = os.path.join(settings.BASE_DIR, 'api', 'models', 'consumption', 'consumption_model.keras')
    scaler_path = os.path.join(settings.BASE_DIR, 'api', 'models', 'consumption', 'scaler.pkl')
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    features_scaled = scaler.transform([features])
    
    prediction = model.predict(features_scaled)
    return float(prediction[0][0])

def predict_power_consumption(v_rms, i_rms, s,device):
    MODEL_PATH = os.path.join(BASE_DIR, 'api', 'models', f'{device}', f'{device}.keras')

    INPUT_SCALER = os.path.join(BASE_DIR, 'api', 'models', f'{device}', f'{device}_scaler_features.pkl')
    OUTPUT_SCALER = os.path.join(BASE_DIR, 'api', 'models', f'{device}', f'{device}_scaler_target.pkl')
    model = load_model(MODEL_PATH)

    input_scaler = joblib.load(INPUT_SCALER)
    output_scaler = joblib.load(OUTPUT_SCALER)

    input_data = pd.DataFrame([[v_rms, i_rms, s]], columns=['V_rms', 'I_rms', 'S'])
    
    input_data_scaled = input_scaler.transform(input_data)
    input_data_scaled = np.reshape(input_data_scaled, (1, 1, 3))

    prediction = model.predict(input_data_scaled)
    
    predicted_power_scaled = prediction[0, 0]  

    predicted_power = output_scaler.inverse_transform([[predicted_power_scaled]])[0][0]

    return abs(predicted_power)

def encode_season(season):
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    return [1 if s == season else 0 for s in seasons]

def encode_category(category):
    categories = ["kertes ház", "ikerház", "panel"]
    return [1 if c == category else 0 for c in categories]

class SolarGRU(nn.Module):
    def __init__(self, input_size=32, hidden_size=128, num_layers=2, output_size=1):
        super(SolarGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def predict_form_gru(df):
    FIXED_SEASONS = ['Autumn', 'Spring', 'Summer', 'Winter']
    FIXED_CATEGORIES = [
        'apartman', 'családi ház', 'gyár', 'hivatal', 'ikerház', 
        'irodaház', 'iskola', 'kertes ház', 'kunyhó', 'könyvtár', 
        'panel lakás', 'sorház', 'tanya'
    ]
    NUMERICAL_COLS = [
        'production', 'air_temp', 'cloud_opacity', 'dhi', 'dni', 'ghi', 'gti',
        'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'clearsky_gti',
        'snow_soiling_rooftop', 'snow_soiling_ground',
        'number_of_panels', 'panel_area_m2' 
    ]
    FINAL_FEATURE_ORDER = NUMERICAL_COLS + \
                        [f'season_{s}' for s in FIXED_SEASONS] + \
                        [f'category_{c}' for c in FIXED_CATEGORIES]
    

    X_SCALER = os.path.join(BASE_DIR, 'api', 'models', 'communities_GRU', 'GRU_scaler_X.save')

    x_scaler = joblib.load(X_SCALER)

    df['season'] = pd.Categorical(df['season'], categories=FIXED_SEASONS)
    df['category'] = pd.Categorical(df['category'], categories=FIXED_CATEGORIES)

    df = pd.get_dummies(df, columns=['season', 'category'], dummy_na=False, dtype='float32')

    df = df[FINAL_FEATURE_ORDER].values.astype('float32')
    df = x_scaler.transform(df)

    print(df)
    return df

def predict_form(timestamp, number_of_panels, panel_area_m2, category, consumption, air_temp,
                       clearsky_dhi, clearsky_dni, clearsky_ghi, clearsky_gti, cloud_opacity, dhi,
                       dni, ghi, gti, snow_soiling_rooftop, snow_soiling_ground, season):
    input_features = [
        cloud_opacity, ghi, gti, dhi, timestamp, category, clearsky_dhi, clearsky_dni, clearsky_ghi, clearsky_gti, number_of_panels, panel_area_m2, consumption,
        air_temp, dni, snow_soiling_rooftop, snow_soiling_ground, season
    ]
    columns = [
        'cloud_opacity', 'ghi', 'gti', 'dhi', 'timestamp', 'category', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'clearsky_gti', 'number_of_panels', 'panel_area_m2', 'consumption',
        'air_temp', 'dni', 'snow_soiling_rooftop', 'snow_soiling_ground', 'season'
    ]
    df = pd.DataFrame([input_features], columns=columns, dtype=object)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = add_time_features(df)
    df['ts_orig'] = df['timestamp']

    CATEGORY_SCALER = os.path.join(BASE_DIR, 'api', 'models', 'communities', 'scaler_category.pkl')
    SEASON_SCALER = os.path.join(BASE_DIR, 'api', 'models', 'communities', 'scaler_season.pkl')

    category_scaler = pickle.load(open(CATEGORY_SCALER, "rb"))
    season_scaler = pickle.load(open(SEASON_SCALER, "rb"))

    df['category'] = category_scaler.transform(df[['category']])
    df['season'] = season_scaler.transform(df[['season']])

    columns = [
        'cloud_opacity', 'ghi', 'gti', 'dhi', 'month', 'category', 'clearsky_dhi', 'year', 'clearsky_dni', 'snow_soiling_ground', 'month_sin', 
        'dni', 'panel_area_m2', 'snow_soiling_rooftop', 'day', 'season', 'consumption', 'clearsky_gti', 'clearsky_ghi', 'month_cos', 'air_temp', 
        'number_of_panels'
    ]
    input = df.loc[[df.index[0]], columns]
    print(input)
    return input

def add_time_features(df):
    df['year']  = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day']   = df['timestamp'].dt.day
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df