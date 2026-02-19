"""
Sistema de Validação de Previsões
Verifica a acurácia das previsões de alta/baixa do modelo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Configurações
SYMBOL = 'AMZN'
SEQUENCE_LENGTH = 60
MODEL_PATH = 'models/best_model.keras'
SCALER_PATH = 'data/scaler.pkl'

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
    
    # Selecionar features
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'Daily_Return', 'Price_Range', 'MA_7', 'MA_21', 'Volatility']
    
    return data[features]

def get_historical_data(symbol, days=365):
    """Obtém dados históricos"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df

def backtest_predictions(model, scaler, df, days_ahead=1):
    """
    Faz backtesting das previsões
    Compara previsões com valores reais
    """
    data = prepare_features(df)
    scaled_data = scaler.transform(data)
    
    predictions = []
    actuals = []
    dates = []
    
    # Fazer previsões para cada ponto possível
    for i in range(SEQUENCE_LENGTH, len(scaled_data) - days_ahead):
        # Sequência de entrada
        sequence = scaled_data[i-SEQUENCE_LENGTH:i]
        
        # Fazer previsão
        pred_normalized = model.predict(sequence.reshape(1, SEQUENCE_LENGTH, -1), verbose=0)
        
        # Desnormalizar previsão
        pred_full = np.zeros((1, scaler.n_features_in_))
        pred_full[0, 3] = pred_normalized[0, 0]
        pred_price = scaler.inverse_transform(pred_full)[0, 3]
        
        # Valor real
        actual_full = np.zeros((1, scaler.n_features_in_))
        actual_full[0, 3] = scaled_data[i + days_ahead - 1, 3]
        actual_price = scaler.inverse_transform(actual_full)[0, 3]
        
        # Preço atual (para calcular direção)
        current_full = np.zeros((1, scaler.n_features_in_))
        current_full[0, 3] = scaled_data[i - 1, 3]
        current_price = scaler.inverse_transform(current_full)[0, 3]
        
        predictions.append({
            'date': df.index[i + days_ahead - 1],
            'current_price': current_price,
            'predicted_price': pred_price,
            'actual_price': actual_price,
            'predicted_direction': 'ALTA' if pred_price > current_price else 'BAIXA',
            'actual_direction': 'ALTA' if actual_price > current_price else 'BAIXA',
            'predicted_change': ((pred_price - current_price) / current_price) * 100,
            'actual_change': ((actual_price - current_price) / current_price) * 100
        })
    
    return pd.DataFrame(predictions)

def calculate_direction_accuracy(results_df):
    """Calcula acurácia da direção (alta/baixa)"""
    correct = (results_df['predicted_direction'] == results_df['actual_direction']).sum()
    total = len(results_df)
    accuracy = (correct / total) * 100
    
    return accuracy, correct, total

def calculate_metrics(results_df):
    """Calcula métricas detalhadas"""
    # Acurácia de direção
    direction_accuracy, correct, total = calculate_direction_accuracy(results_df)
    
    # Erro absoluto médio
    mae = np.mean(np.abs(results_df['predicted_price'] - results_df['actual_price']))
    
    # Erro percentual médio
    mape = np.mean(np.abs(
        (results_df['actual_price'] - results_df['predicted_price']) / results_df['actual_price']
    )) * 100
    
    # Matriz de confusão
    y_true = (results_df['actual_direction'] == 'ALTA').astype(int)
    y_pred = (results_df['predicted_direction'] == 'ALTA').astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Métricas por classe
    tn, fp, fn, tp = cm.ravel()
    
    # Precisão e Recall
    precision_alta = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_alta = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_alta = 2 * (precision_alta * recall_alta) / (precision_alta + recall_alta) if (precision_alta + recall_alta) > 0 else 0
    
    precision_baixa = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_baixa = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_baixa = 2 * (precision_baixa * recall_baixa) / (precision_baixa + recall_baixa) if (precision_baixa + recall_baixa) > 0 else 0
    
    return {
        'direction_accuracy': direction_accuracy,
        'correct_predictions': correct,
        'total_predictions': total,
        'mae': mae,
        'mape': mape,
        'confusion_matrix': cm,
        'precision_alta': precision_alta * 100,
        'recall_alta': recall_alta * 100,
        'f1_alta': f1_alta * 100,
        'precision_baixa': precision_baixa * 100,
        'recall_baixa': recall_baixa * 100,
        'f1_baixa': f1_baixa * 100
    }

def plot_validation_results(results_df, metrics, save_path='models/validation_results.png'):
    """Gera gráficos de validação"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Acurácia ao longo do tempo
    ax1 = axes[0, 0]
    window = 30
    rolling_accuracy = []
    for i in range(window, len(results_df)):
        subset = results_df.iloc[i-window:i]
        acc = (subset['predicted_direction'] == subset['actual_direction']).sum() / window * 100
        rolling_accuracy.append(acc)
    
    ax1.plot(results_df['date'].iloc[window:], rolling_accuracy, linewidth=2)
    ax1.axhline(y=50, color='r', linestyle='--', label='Baseline (50%)')
    ax1.axhline(y=metrics['direction_accuracy'], color='g', linestyle='--', 
                label=f'Média ({metrics["direction_accuracy"]:.1f}%)')
    ax1.set_title('Acurácia de Direção ao Longo do Tempo (Janela de 30 dias)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Acurácia (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Matriz de Confusão
    ax2 = axes[0, 1]
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['BAIXA', 'ALTA'],
                yticklabels=['BAIXA', 'ALTA'])
    ax2.set_title('Matriz de Confusão', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Real')
    ax2.set_xlabel('Previsto')
    
    # 3. Distribuição de Erros
    ax3 = axes[1, 0]
    errors = results_df['predicted_price'] - results_df['actual_price']
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_title(f'Distribuição de Erros (MAE: ${metrics["mae"]:.2f})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Erro de Previsão ($)')
    ax3.set_ylabel('Frequência')
    ax3.grid(True, alpha=0.3)
    
    # 4. Métricas por Classe
    ax4 = axes[1, 1]
    classes = ['ALTA', 'BAIXA']
    precision = [metrics['precision_alta'], metrics['precision_baixa']]
    recall = [metrics['recall_alta'], metrics['recall_baixa']]
    f1 = [metrics['f1_alta'], metrics['f1_baixa']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax4.bar(x - width, precision, width, label='Precisão', alpha=0.8)
    ax4.bar(x, recall, width, label='Recall', alpha=0.8)
    ax4.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax4.set_title('Métricas por Classe', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(classes)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráficos salvos em: {save_path}")
    
    return fig

def print_report(metrics, results_df):
    """Imprime relatório detalhado"""
    print("\n" + "="*70)
    print("  RELATÓRIO DE VALIDAÇÃO DE PREVISÕES")
    print("="*70)
    
    print(f"\n📊 ACURÁCIA DE DIREÇÃO (ALTA/BAIXA)")
    print(f"   Acertos: {metrics['correct_predictions']}/{metrics['total_predictions']}")
    print(f"   Acurácia: {metrics['direction_accuracy']:.2f}%")
    
    if metrics['direction_accuracy'] >= 60:
        print(f"   Status: ✅ EXCELENTE (>60%)")
    elif metrics['direction_accuracy'] >= 55:
        print(f"   Status: ✅ BOM (55-60%)")
    elif metrics['direction_accuracy'] >= 50:
        print(f"   Status: ⚠️  RAZOÁVEL (50-55%)")
    else:
        print(f"   Status: ❌ ABAIXO DO BASELINE (<50%)")
    
    print(f"\n💰 ERRO DE PREÇO")
    print(f"   MAE: ${metrics['mae']:.2f}")
    print(f"   MAPE: {metrics['mape']:.2f}%")
    
    print(f"\n📈 MÉTRICAS PARA PREVISÕES DE ALTA")
    print(f"   Precisão: {metrics['precision_alta']:.2f}%")
    print(f"   Recall: {metrics['recall_alta']:.2f}%")
    print(f"   F1-Score: {metrics['f1_alta']:.2f}%")
    
    print(f"\n📉 MÉTRICAS PARA PREVISÕES DE BAIXA")
    print(f"   Precisão: {metrics['precision_baixa']:.2f}%")
    print(f"   Recall: {metrics['recall_baixa']:.2f}%")
    print(f"   F1-Score: {metrics['f1_baixa']:.2f}%")
    
    print(f"\n🎯 MATRIZ DE CONFUSÃO")
    cm = metrics['confusion_matrix']
    print(f"   True Negatives (BAIXA correta): {cm[0,0]}")
    print(f"   False Positives (previu ALTA, foi BAIXA): {cm[0,1]}")
    print(f"   False Negatives (previu BAIXA, foi ALTA): {cm[1,0]}")
    print(f"   True Positives (ALTA correta): {cm[1,1]}")
    
    # Análise de tendências
    print(f"\n📊 ANÁLISE DE TENDÊNCIAS")
    alta_predictions = (results_df['predicted_direction'] == 'ALTA').sum()
    baixa_predictions = (results_df['predicted_direction'] == 'BAIXA').sum()
    print(f"   Previsões de ALTA: {alta_predictions} ({alta_predictions/len(results_df)*100:.1f}%)")
    print(f"   Previsões de BAIXA: {baixa_predictions} ({baixa_predictions/len(results_df)*100:.1f}%)")
    
    # Últimas previsões
    print(f"\n🔍 ÚLTIMAS 5 PREVISÕES")
    print(f"   {'Data':<12} {'Previsto':<8} {'Real':<8} {'Acerto':<8}")
    print(f"   {'-'*40}")
    for _, row in results_df.tail(5).iterrows():
        acerto = '✅' if row['predicted_direction'] == row['actual_direction'] else '❌'
        print(f"   {row['date'].strftime('%Y-%m-%d'):<12} {row['predicted_direction']:<8} "
              f"{row['actual_direction']:<8} {acerto:<8}")
    
    print("\n" + "="*70)

def main():
    """Função principal"""
    print("\n🔍 Iniciando Validação de Previsões...")
    
    # Carregar modelo e scaler
    print("\n📦 Carregando modelo e scaler...")
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✓ Modelo e scaler carregados")
    
    # Obter dados históricos
    print(f"\n📈 Obtendo dados históricos de {SYMBOL}...")
    df = get_historical_data(SYMBOL, days=365)
    print(f"✓ {len(df)} registros obtidos")
    
    # Fazer backtesting
    print("\n🔄 Executando backtesting...")
    results_df = backtest_predictions(model, scaler, df, days_ahead=1)
    print(f"✓ {len(results_df)} previsões testadas")
    
    # Calcular métricas
    print("\n📊 Calculando métricas...")
    metrics = calculate_metrics(results_df)
    
    # Gerar relatório
    print_report(metrics, results_df)
    
    # Gerar gráficos
    print("\n📊 Gerando gráficos...")
    plot_validation_results(results_df, metrics)
    
    # Salvar resultados
    results_path = 'models/validation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"✓ Resultados salvos em: {results_path}")
    
    print("\n✅ Validação concluída!")
    
    return results_df, metrics

if __name__ == "__main__":
    results_df, metrics = main()
