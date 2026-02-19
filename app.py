"""
API REST para previsão de preços de ações usando modelo LSTM
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import yfinance as yf
from datetime import datetime, timedelta
import os

app = FastAPI(
    title="LSTM Stock Prediction API",
    description="API para previsão de preços de ações da Amazon (AMZN) usando LSTM",
    version="1.0.0"
)

# Carregar modelo e scaler
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best_model.keras')
SCALER_PATH = os.getenv('SCALER_PATH', 'data/scaler.pkl')

try:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Modelo e scaler carregados com sucesso!")
except Exception as e:
    print(f"⚠ Erro ao carregar modelo: {e}")
    model = None
    scaler = None

# Configurações
SYMBOL = 'AMZN'
SEQUENCE_LENGTH = 60

class PredictionRequest(BaseModel):
    """Schema para requisição de previsão"""
    days_ahead: int = Field(default=1, ge=1, le=30, description="Número de dias para prever (1-30)")

class PredictionResponse(BaseModel):
    """Schema para resposta de previsão"""
    symbol: str
    current_price: float
    predicted_prices: List[float]
    prediction_dates: List[str]
    confidence_interval: dict

class HealthResponse(BaseModel):
    """Schema para health check"""
    status: str
    model_loaded: bool
    timestamp: str

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz"""
    return {
        "message": "LSTM Stock Prediction API",
        "symbol": SYMBOL,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Verifica status da API"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

def get_latest_data(symbol: str, days: int = 100):
    """Obtém dados mais recentes da ação"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"Não foi possível obter dados para {symbol}")
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df

def prepare_features(df):
    """Prepara features do DataFrame"""
    data = df.copy()
    
    # Features derivadas
    data['Daily_Return'] = data['Close'].pct_change()
    data['Price_Range'] = data['High'] - data['Low']
    data['MA_7'] = data['Close'].rolling(window=7).mean()
    data['MA_21'] = data['Close'].rolling(window=21).mean()
    data['Volatility'] = data['Daily_Return'].rolling(window=21).std()
    
    # Remover NaN
    data = data.dropna()
    
    # Selecionar features na ordem correta
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'Daily_Return', 'Price_Range', 'MA_7', 'MA_21', 'Volatility']
    
    return data[features]

def predict_next_days(model, scaler, last_sequence, days_ahead=1):
    """Faz previsões para os próximos dias"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Fazer previsão
        pred_normalized = model.predict(current_sequence.reshape(1, SEQUENCE_LENGTH, -1), verbose=0)
        
        # Desnormalizar
        pred_full = np.zeros((1, scaler.n_features_in_))
        pred_full[0, 3] = pred_normalized[0, 0]  # Close é a 4ª coluna (índice 3)
        pred_denormalized = scaler.inverse_transform(pred_full)[0, 3]
        
        predictions.append(float(pred_denormalized))
        
        # Atualizar sequência para próxima previsão
        # Criar novo ponto com previsão
        new_point = current_sequence[-1].copy()
        new_point[3] = pred_normalized[0, 0]  # Atualizar Close normalizado
        
        # Shift da sequência
        current_sequence = np.vstack([current_sequence[1:], new_point])
    
    return predictions

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Faz previsão de preços futuros
    
    - **days_ahead**: Número de dias para prever (1-30)
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        # Obter dados recentes
        df = get_latest_data(SYMBOL, days=150)
        
        # Preparar features
        data = prepare_features(df)
        
        # Normalizar
        scaled_data = scaler.transform(data)
        
        # Pegar últimos 60 dias
        if len(scaled_data) < SEQUENCE_LENGTH:
            raise ValueError(f"Dados insuficientes. Necessário {SEQUENCE_LENGTH} dias.")
        
        last_sequence = scaled_data[-SEQUENCE_LENGTH:]
        
        # Fazer previsões
        predictions = predict_next_days(model, scaler, last_sequence, request.days_ahead)
        
        # Gerar datas de previsão
        last_date = df.index[-1]
        prediction_dates = []
        for i in range(1, request.days_ahead + 1):
            next_date = last_date + timedelta(days=i)
            # Pular finais de semana
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            prediction_dates.append(next_date.strftime('%Y-%m-%d'))
        
        # Preço atual
        current_price = float(df['Close'].iloc[-1])
        
        # Calcular intervalo de confiança (simplificado)
        std_dev = np.std(predictions)
        confidence_interval = {
            "lower": [float(p - 1.96 * std_dev) for p in predictions],
            "upper": [float(p + 1.96 * std_dev) for p in predictions]
        }
        
        return {
            "symbol": SYMBOL,
            "current_price": current_price,
            "predicted_prices": predictions,
            "prediction_dates": prediction_dates,
            "confidence_interval": confidence_interval
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Retorna informações sobre o modelo"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    # Ler métricas se disponível
    metrics_path = 'models/metrics.csv'
    metrics = {}
    
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metrics = metrics_df.to_dict('records')[0]
    
    return {
        "symbol": SYMBOL,
        "sequence_length": SEQUENCE_LENGTH,
        "model_architecture": {
            "layers": len(model.layers),
            "total_params": model.count_params()
        },
        "metrics": metrics
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
