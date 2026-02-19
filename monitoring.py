"""
Script de monitoramento da API
Monitora performance, tempo de resposta e uso de recursos
"""

import requests
import time
import psutil
import json
from datetime import datetime
import pandas as pd
import os

API_URL = "http://localhost:8000"
LOG_FILE = "monitoring_log.csv"

class APIMonitor:
    def __init__(self, api_url=API_URL):
        self.api_url = api_url
        self.metrics = []
    
    def check_health(self):
        """Verifica saúde da API"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "status": "up" if response.status_code == 200 else "down",
                "status_code": response.status_code,
                "response_time_ms": response_time
            }
        except Exception as e:
            return {
                "status": "down",
                "status_code": 0,
                "response_time_ms": 0,
                "error": str(e)
            }
    
    def test_prediction(self):
        """Testa endpoint de previsão"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/predict",
                json={"days_ahead": 1},
                timeout=30
            )
            response_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "response_time_ms": response_time
            }
        except Exception as e:
            return {
                "status": "error",
                "status_code": 0,
                "response_time_ms": 0,
                "error": str(e)
            }
    
    def get_system_metrics(self):
        """Obtém métricas do sistema"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    
    def collect_metrics(self):
        """Coleta todas as métricas"""
        timestamp = datetime.now()
        
        health = self.check_health()
        prediction = self.test_prediction()
        system = self.get_system_metrics()
        
        metric = {
            "timestamp": timestamp.isoformat(),
            "health_status": health["status"],
            "health_response_time_ms": health["response_time_ms"],
            "prediction_status": prediction["status"],
            "prediction_response_time_ms": prediction["response_time_ms"],
            "cpu_percent": system["cpu_percent"],
            "memory_percent": system["memory_percent"],
            "disk_percent": system["disk_percent"]
        }
        
        self.metrics.append(metric)
        return metric
    
    def save_metrics(self):
        """Salva métricas em arquivo CSV"""
        if not self.metrics:
            return
        
        df = pd.DataFrame(self.metrics)
        
        if os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(LOG_FILE, index=False)
        
        self.metrics = []
    
    def print_metric(self, metric):
        """Imprime métrica formatada"""
        print(f"\n[{metric['timestamp']}]")
        print(f"  Health: {metric['health_status']} ({metric['health_response_time_ms']:.2f}ms)")
        print(f"  Prediction: {metric['prediction_status']} ({metric['prediction_response_time_ms']:.2f}ms)")
        print(f"  CPU: {metric['cpu_percent']:.1f}% | Memory: {metric['memory_percent']:.1f}% | Disk: {metric['disk_percent']:.1f}%")
    
    def monitor(self, interval=60, duration=None):
        """
        Monitora a API continuamente
        
        Args:
            interval: Intervalo entre coletas (segundos)
            duration: Duração total do monitoramento (segundos), None para infinito
        """
        print("="*60)
        print("MONITORAMENTO DA API - LSTM STOCK PREDICTION")
        print("="*60)
        print(f"URL: {self.api_url}")
        print(f"Intervalo: {interval}s")
        print(f"Log: {LOG_FILE}")
        print("Pressione Ctrl+C para parar")
        print("="*60)
        
        start_time = time.time()
        
        try:
            while True:
                metric = self.collect_metrics()
                self.print_metric(metric)
                
                # Salvar a cada 10 coletas
                if len(self.metrics) >= 10:
                    self.save_metrics()
                
                # Verificar duração
                if duration and (time.time() - start_time) >= duration:
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoramento interrompido pelo usuário")
        finally:
            self.save_metrics()
            print(f"\nMétricas salvas em {LOG_FILE}")
    
    def generate_report(self):
        """Gera relatório de monitoramento"""
        if not os.path.exists(LOG_FILE):
            print(f"Arquivo {LOG_FILE} não encontrado")
            return
        
        df = pd.read_csv(LOG_FILE)
        
        print("\n" + "="*60)
        print("RELATÓRIO DE MONITORAMENTO")
        print("="*60)
        
        print(f"\nPeríodo: {df['timestamp'].iloc[0]} até {df['timestamp'].iloc[-1]}")
        print(f"Total de coletas: {len(df)}")
        
        print("\n--- Tempo de Resposta (ms) ---")
        print(f"Health endpoint:")
        print(f"  Média: {df['health_response_time_ms'].mean():.2f}ms")
        print(f"  Mínimo: {df['health_response_time_ms'].min():.2f}ms")
        print(f"  Máximo: {df['health_response_time_ms'].max():.2f}ms")
        
        print(f"\nPrediction endpoint:")
        print(f"  Média: {df['prediction_response_time_ms'].mean():.2f}ms")
        print(f"  Mínimo: {df['prediction_response_time_ms'].min():.2f}ms")
        print(f"  Máximo: {df['prediction_response_time_ms'].max():.2f}ms")
        
        print("\n--- Uso de Recursos ---")
        print(f"CPU:")
        print(f"  Média: {df['cpu_percent'].mean():.1f}%")
        print(f"  Máximo: {df['cpu_percent'].max():.1f}%")
        
        print(f"\nMemória:")
        print(f"  Média: {df['memory_percent'].mean():.1f}%")
        print(f"  Máximo: {df['memory_percent'].max():.1f}%")
        
        print("\n--- Disponibilidade ---")
        uptime = (df['health_status'] == 'up').sum() / len(df) * 100
        print(f"Uptime: {uptime:.2f}%")
        
        success_rate = (df['prediction_status'] == 'success').sum() / len(df) * 100
        print(f"Taxa de sucesso (previsões): {success_rate:.2f}%")

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitoramento da API')
    parser.add_argument('--url', default=API_URL, help='URL da API')
    parser.add_argument('--interval', type=int, default=60, help='Intervalo entre coletas (segundos)')
    parser.add_argument('--duration', type=int, help='Duração do monitoramento (segundos)')
    parser.add_argument('--report', action='store_true', help='Gerar relatório')
    
    args = parser.parse_args()
    
    monitor = APIMonitor(api_url=args.url)
    
    if args.report:
        monitor.generate_report()
    else:
        monitor.monitor(interval=args.interval, duration=args.duration)

if __name__ == "__main__":
    main()
