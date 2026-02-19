"""
Script de teste completo da API de previsão de ações
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_root():
    """Testa endpoint raiz"""
    print_section("1. Testando Endpoint Raiz")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_health():
    """Testa health check"""
    print_section("2. Testando Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))
    
    if data['status'] == 'healthy':
        print("\n✅ API está saudável!")
    else:
        print("\n❌ API com problemas!")

def test_model_info():
    """Testa informações do modelo"""
    print_section("3. Testando Informações do Modelo")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(json.dumps(data, indent=2))
    
    print(f"\n📊 Métricas do Modelo:")
    print(f"   MAE: ${data['metrics']['MAE']:.2f}")
    print(f"   RMSE: ${data['metrics']['RMSE']:.2f}")
    print(f"   MAPE: {data['metrics']['MAPE']:.2f}%")

def test_prediction(days=5):
    """Testa previsão"""
    print_section(f"4. Testando Previsão para {days} dias")
    
    payload = {"days_ahead": days}
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n📈 Previsões para {data['symbol']}:")
        print(f"   Preço Atual: ${data['current_price']:.2f}")
        print(f"\n   Previsões:")
        
        for i, (date, price, lower, upper) in enumerate(zip(
            data['prediction_dates'],
            data['predicted_prices'],
            data['confidence_interval']['lower'],
            data['confidence_interval']['upper']
        ), 1):
            change = ((price - data['current_price']) / data['current_price']) * 100
            print(f"   {i}. {date}: ${price:.2f} ({change:+.2f}%)")
            print(f"      Intervalo: ${lower:.2f} - ${upper:.2f}")
        
        # Calcular tendência
        if data['predicted_prices'][-1] > data['current_price']:
            trend = "📈 ALTA"
        elif data['predicted_prices'][-1] < data['current_price']:
            trend = "📉 BAIXA"
        else:
            trend = "➡️ ESTÁVEL"
        
        print(f"\n   Tendência: {trend}")
    else:
        print(f"❌ Erro: {response.json()}")

def test_multiple_predictions():
    """Testa múltiplas previsões"""
    print_section("5. Testando Múltiplas Previsões")
    
    for days in [1, 7, 15, 30]:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"days_ahead": days},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            last_price = data['predicted_prices'][-1]
            change = ((last_price - data['current_price']) / data['current_price']) * 100
            print(f"   {days:2d} dias: ${last_price:.2f} ({change:+.2f}%)")
        else:
            print(f"   {days:2d} dias: ❌ Erro")

def main():
    print("\n" + "🚀 " * 20)
    print("  TESTE COMPLETO DA API DE PREVISÃO DE AÇÕES LSTM")
    print("🚀 " * 20)
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_prediction(5)
        test_multiple_predictions()
        
        print_section("✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("\n📍 Acesse a documentação interativa:")
        print(f"   {BASE_URL}/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: Não foi possível conectar à API")
        print("   Certifique-se de que a API está rodando em http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()
