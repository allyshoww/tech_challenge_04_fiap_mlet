"""
Pipeline completo de treinamento
Executa todas as etapas: coleta -> preparação -> treinamento
"""

import os
import sys

def run_pipeline():
    """Executa o pipeline completo"""
    print("="*60)
    print("PIPELINE COMPLETO DE TREINAMENTO - LSTM AMZN")
    print("="*60)
    
    # Etapa 1: Coleta de dados
    print("\n[1/3] Coletando e pré-processando dados...")
    print("-"*60)
    import data_collection
    data_collection.main()
    
    # Etapa 2: Preparar sequências
    print("\n[2/3] Preparando sequências temporais...")
    print("-"*60)
    import prepare_sequences
    prepare_sequences.prepare_data_for_lstm(sequence_length=60)
    
    # Etapa 3: Treinar modelo
    print("\n[3/3] Treinando modelo LSTM...")
    print("-"*60)
    import lstm_model
    lstm_model.main()
    
    print("\n" + "="*60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print("\nPróximos passos:")
    print("  1. Revisar métricas em 'models/metrics.csv'")
    print("  2. Analisar gráficos em 'models/'")
    print("  3. Criar API para servir o modelo")

if __name__ == "__main__":
    run_pipeline()
