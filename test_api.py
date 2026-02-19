"""
Script para testar a API
"""

import requests
import json

# URL base da API
BASE_URL = "http://localhost:8000"

def test_root():
    """Testa endpoint raiz"""
    print("="*50)
    print("Testando endpoint raiz (GET /)")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_health():
    """Testa health check"""
    print("="*50)
    print("Testando health check (GET /health)")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_model_info():
    """Testa informações do modelo"""
    print("="*50)
    print("Testando informações do modelo (GET /model/info)")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction(days_ahead=5):
    """Testa previsão"""
    print("="*50)
    print(f"Testando previsão (POST /predict) - {days_ahead} dias")
    print("="*50)
    
    payload = {
        "days_ahead": days_ahead
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nSímbolo: {data['symbol']}")
        print(f"Preço Atual: ${data['current_price']:.2f}")
        print(f"\nPrevisões:")
        for date, price in zip(data['prediction_dates'], data['predicted_prices']):
            print(f"  {date}: ${price:.2f}")
    else:
        print(f"Erro: {response.text}")
    print()

def main():
    """Executa todos os testes"""
    print("\n" + "="*50)
    print("TESTANDO API DE PREVISÃO DE AÇÕES")
    print("="*50 + "\n")
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_prediction(days_ahead=1)
        test_prediction(days_ahead=5)
        test_prediction(days_ahead=10)
        
        print("="*50)
        print("TODOS OS TESTES CONCLUÍDOS!")
        print("="*50)
        
    except requests.exceptions.ConnectionError:
        print("❌ Erro: Não foi possível conectar à API.")
        print("Certifique-se de que a API está rodando em http://localhost:8000")
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")

if __name__ == "__main__":
    main()
