"""
Script para coleta e pré-processamento de dados de ações
Empresa: Amazon (AMZN)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configurações
SYMBOL = 'AMZN'
START_DATE = '2015-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
DATA_DIR = 'data'

# Criar diretório de dados se não existir
os.makedirs(DATA_DIR, exist_ok=True)

def collect_data():
    """Coleta dados históricos da ação"""
    print(f"Coletando dados de {SYMBOL} de {START_DATE} até {END_DATE}...")
    
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
    
    # Remover multi-index se existir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"\nDados coletados: {len(df)} registros")
    print(f"Período: {df.index[0]} até {df.index[-1]}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
    print(f"\nInformações do dataset:")
    print(df.info())
    
    # Salvar dados brutos
    raw_file = os.path.join(DATA_DIR, f'{SYMBOL}_raw.csv')
    df.to_csv(raw_file)
    print(f"\nDados brutos salvos em: {raw_file}")
    
    return df

def analyze_data(df):
    """Análise exploratória dos dados"""
    print("\n" + "="*50)
    print("ANÁLISE EXPLORATÓRIA")
    print("="*50)
    
    # Estatísticas descritivas
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    # Verificar valores faltantes
    print("\nValores faltantes:")
    print(df.isnull().sum())
    
    # Criar visualizações
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Preço de fechamento ao longo do tempo
    axes[0, 0].plot(df.index, df['Close'])
    axes[0, 0].set_title(f'{SYMBOL} - Preço de Fechamento')
    axes[0, 0].set_xlabel('Data')
    axes[0, 0].set_ylabel('Preço (USD)')
    axes[0, 0].grid(True)
    
    # Volume de negociação
    axes[0, 1].plot(df.index, df['Volume'], color='orange')
    axes[0, 1].set_title(f'{SYMBOL} - Volume de Negociação')
    axes[0, 1].set_xlabel('Data')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True)
    
    # Distribuição dos retornos diários
    returns = df['Close'].pct_change().dropna()
    axes[1, 0].hist(returns, bins=50, edgecolor='black')
    axes[1, 0].set_title('Distribuição dos Retornos Diários')
    axes[1, 0].set_xlabel('Retorno')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].grid(True)
    
    # Boxplot dos preços
    axes[1, 1].boxplot([df['Open'], df['High'], df['Low'], df['Close']])
    axes[1, 1].set_xticklabels(['Open', 'High', 'Low', 'Close'])
    axes[1, 1].set_title('Distribuição dos Preços')
    axes[1, 1].set_ylabel('Preço (USD)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = os.path.join(DATA_DIR, f'{SYMBOL}_analysis.png')
    plt.savefig(plot_file)
    print(f"\nGráficos salvos em: {plot_file}")
    plt.close()

def preprocess_data(df):
    """Pré-processamento dos dados"""
    print("\n" + "="*50)
    print("PRÉ-PROCESSAMENTO")
    print("="*50)
    
    # Criar cópia para não modificar original
    data = df.copy()
    
    # Tratar valores faltantes (se houver)
    if data.isnull().sum().sum() > 0:
        print("\nPreenchendo valores faltantes...")
        data = data.fillna(method='ffill')
    
    # Selecionar features relevantes
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[features]
    
    # Adicionar features derivadas
    data['Daily_Return'] = data['Close'].pct_change()
    data['Price_Range'] = data['High'] - data['Low']
    data['MA_7'] = data['Close'].rolling(window=7).mean()
    data['MA_21'] = data['Close'].rolling(window=21).mean()
    data['Volatility'] = data['Daily_Return'].rolling(window=21).std()
    
    # Remover linhas com NaN geradas pelas features derivadas
    data = data.dropna()
    
    print(f"\nDados após feature engineering: {len(data)} registros")
    print(f"\nFeatures criadas:")
    print(data.columns.tolist())
    
    # Normalização dos dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Criar DataFrame com dados normalizados
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
    
    print("\nDados normalizados (primeiras linhas):")
    print(scaled_df.head())
    
    # Salvar dados processados
    processed_file = os.path.join(DATA_DIR, f'{SYMBOL}_processed.csv')
    data.to_csv(processed_file)
    print(f"\nDados processados salvos em: {processed_file}")
    
    # Salvar scaler para uso futuro
    import joblib
    scaler_file = os.path.join(DATA_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_file)
    print(f"Scaler salvo em: {scaler_file}")
    
    return data, scaled_df, scaler

def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """Divide os dados em treino, validação e teste"""
    print("\n" + "="*50)
    print("DIVISÃO DOS DADOS")
    print("="*50)
    
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\nTotal de registros: {n}")
    print(f"Treino: {len(train_data)} registros ({train_ratio*100:.0f}%)")
    print(f"Validação: {len(val_data)} registros ({val_ratio*100:.0f}%)")
    print(f"Teste: {len(test_data)} registros ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    print(f"\nPeríodos:")
    print(f"Treino: {train_data.index[0]} até {train_data.index[-1]}")
    print(f"Validação: {val_data.index[0]} até {val_data.index[-1]}")
    print(f"Teste: {test_data.index[0]} até {test_data.index[-1]}")
    
    return train_data, val_data, test_data

def main():
    """Função principal"""
    print("="*50)
    print(f"COLETA E PRÉ-PROCESSAMENTO DE DADOS - {SYMBOL}")
    print("="*50)
    
    # 1. Coletar dados
    df = collect_data()
    
    # 2. Análise exploratória
    analyze_data(df)
    
    # 3. Pré-processamento
    data, scaled_data, scaler = preprocess_data(df)
    
    # 4. Divisão dos dados
    train_data, val_data, test_data = split_data(scaled_data)
    
    print("\n" + "="*50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("="*50)
    print(f"\nArquivos gerados no diretório '{DATA_DIR}/':")
    print(f"  - {SYMBOL}_raw.csv (dados brutos)")
    print(f"  - {SYMBOL}_processed.csv (dados processados)")
    print(f"  - {SYMBOL}_analysis.png (gráficos de análise)")
    print(f"  - scaler.pkl (normalizador)")

if __name__ == "__main__":
    main()
