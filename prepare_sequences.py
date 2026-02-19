"""
Script para preparar sequências temporais para o modelo LSTM
"""

import numpy as np
import pandas as pd
import os

def create_sequences(data, target_col='Close', sequence_length=60):
    """
    Cria sequências temporais para treinamento do LSTM
    
    Args:
        data: DataFrame com dados normalizados
        target_col: Nome da coluna alvo (default: 'Close')
        sequence_length: Número de dias anteriores para usar como entrada
    
    Returns:
        X: Array com sequências de entrada
        y: Array com valores alvo
    """
    X = []
    y = []
    
    # Converter para numpy array
    data_array = data.values
    target_idx = data.columns.get_loc(target_col)
    
    for i in range(sequence_length, len(data_array)):
        # Sequência de entrada (todos os features)
        X.append(data_array[i-sequence_length:i])
        # Valor alvo (apenas Close)
        y.append(data_array[i, target_idx])
    
    return np.array(X), np.array(y)

def prepare_data_for_lstm(sequence_length=60):
    """
    Prepara todos os conjuntos de dados para o LSTM
    """
    print("="*50)
    print("PREPARAÇÃO DE SEQUÊNCIAS PARA LSTM")
    print("="*50)
    
    # Carregar dados processados
    data_file = 'data/AMZN_processed.csv'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Arquivo {data_file} não encontrado. Execute data_collection.py primeiro.")
    
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Normalizar novamente (necessário para criar sequências)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    # Dividir dados
    n = len(scaled_df)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    train_data = scaled_df[:train_size]
    val_data = scaled_df[train_size:train_size + val_size]
    test_data = scaled_df[train_size + val_size:]
    
    print(f"\nTamanho da sequência: {sequence_length} dias")
    print(f"Features utilizadas: {df.columns.tolist()}")
    
    # Criar sequências
    print("\nCriando sequências de treino...")
    X_train, y_train = create_sequences(train_data, sequence_length=sequence_length)
    
    print("Criando sequências de validação...")
    X_val, y_val = create_sequences(val_data, sequence_length=sequence_length)
    
    print("Criando sequências de teste...")
    X_test, y_test = create_sequences(test_data, sequence_length=sequence_length)
    
    print(f"\nFormato dos dados:")
    print(f"X_train: {X_train.shape} (samples, sequence_length, features)")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Salvar sequências
    os.makedirs('data/sequences', exist_ok=True)
    
    np.save('data/sequences/X_train.npy', X_train)
    np.save('data/sequences/y_train.npy', y_train)
    np.save('data/sequences/X_val.npy', X_val)
    np.save('data/sequences/y_val.npy', y_val)
    np.save('data/sequences/X_test.npy', X_test)
    np.save('data/sequences/y_test.npy', y_test)
    
    print("\nSequências salvas em 'data/sequences/'")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    prepare_data_for_lstm(sequence_length=60)
    print("\n" + "="*50)
    print("PREPARAÇÃO CONCLUÍDA!")
    print("="*50)
