"""
Modelo LSTM para previsão de preços de ações
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

class StockLSTM:
    def __init__(self, sequence_length=60, n_features=10):
        """
        Inicializa o modelo LSTM
        
        Args:
            sequence_length: Número de dias anteriores
            n_features: Número de features de entrada
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=[128, 64], dropout_rate=0.2):
        """
        Constrói a arquitetura do modelo LSTM
        
        Args:
            lstm_units: Lista com número de unidades em cada camada LSTM
            dropout_rate: Taxa de dropout para regularização
        """
        model = Sequential()
        
        # Primeira camada LSTM
        model.add(LSTM(units=lstm_units[0], 
                      return_sequences=True,
                      input_shape=(self.sequence_length, self.n_features)))
        model.add(Dropout(dropout_rate))
        
        # Segunda camada LSTM
        model.add(LSTM(units=lstm_units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Camada densa de saída
        model.add(Dense(units=1))
        
        # Compilar modelo
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        self.model = model
        
        print("="*50)
        print("ARQUITETURA DO MODELO")
        print("="*50)
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Treina o modelo LSTM
        
        Args:
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validação
            epochs: Número de épocas
            batch_size: Tamanho do batch
        """
        print("\n" + "="*50)
        print("TREINAMENTO DO MODELO")
        print("="*50)
        
        # Criar diretório para modelos
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Treinar modelo
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        print("\nTreinamento concluído!")
        
        return self.history
    
    def evaluate(self, X_test, y_test, scaler=None):
        """
        Avalia o modelo nos dados de teste
        
        Args:
            X_test, y_test: Dados de teste
            scaler: Scaler para desnormalizar os valores
        """
        print("\n" + "="*50)
        print("AVALIAÇÃO DO MODELO")
        print("="*50)
        
        # Fazer previsões
        y_pred = self.model.predict(X_test)
        
        # Desnormalizar se scaler fornecido
        if scaler is not None:
            # Criar array com shape correto para inverse_transform
            y_test_full = np.zeros((len(y_test), scaler.n_features_in_))
            y_pred_full = np.zeros((len(y_pred), scaler.n_features_in_))
            
            # Assumindo que Close é a 4ª coluna (índice 3)
            close_idx = 3
            y_test_full[:, close_idx] = y_test
            y_pred_full[:, close_idx] = y_pred.flatten()
            
            y_test_denorm = scaler.inverse_transform(y_test_full)[:, close_idx]
            y_pred_denorm = scaler.inverse_transform(y_pred_full)[:, close_idx]
        else:
            y_test_denorm = y_test
            y_pred_denorm = y_pred.flatten()
        
        # Calcular métricas
        mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
        rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
        mape = mean_absolute_percentage_error(y_test_denorm, y_pred_denorm) * 100
        
        print(f"\nMétricas de Avaliação:")
        print(f"MAE (Mean Absolute Error): ${mae:.2f}")
        print(f"RMSE (Root Mean Square Error): ${rmse:.2f}")
        print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
        
        # Salvar métricas
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('models/metrics.csv', index=False)
        print("\nMétricas salvas em 'models/metrics.csv'")
        
        return y_pred_denorm, metrics
    
    def plot_training_history(self):
        """Plota histórico de treinamento"""
        if self.history is None:
            print("Modelo ainda não foi treinado!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Treino')
        axes[0].plot(self.history.history['val_loss'], label='Validação')
        axes[0].set_title('Loss durante o Treinamento')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Treino')
        axes[1].plot(self.history.history['val_mae'], label='Validação')
        axes[1].set_title('MAE durante o Treinamento')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        print("Histórico de treinamento salvo em 'models/training_history.png'")
        plt.close()
    
    def plot_predictions(self, y_test, y_pred, save_path='models/predictions.png'):
        """Plota previsões vs valores reais"""
        plt.figure(figsize=(15, 6))
        
        plt.plot(y_test, label='Valor Real', linewidth=2)
        plt.plot(y_pred, label='Previsão', linewidth=2, alpha=0.7)
        plt.title('Previsões vs Valores Reais - AMZN')
        plt.xlabel('Dias')
        plt.ylabel('Preço (USD)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Gráfico de previsões salvo em '{save_path}'")
        plt.close()
    
    def save_model(self, filepath='models/lstm_model.keras'):
        """Salva o modelo treinado"""
        self.model.save(filepath)
        print(f"\nModelo salvo em '{filepath}'")
    
    @staticmethod
    def load_model(filepath='models/lstm_model.keras'):
        """Carrega um modelo salvo"""
        model = keras.models.load_model(filepath)
        print(f"Modelo carregado de '{filepath}'")
        return model

def main():
    """Função principal para treinar o modelo"""
    print("="*50)
    print("TREINAMENTO DO MODELO LSTM - AMZN")
    print("="*50)
    
    # Carregar sequências
    print("\nCarregando sequências...")
    X_train = np.load('data/sequences/X_train.npy')
    y_train = np.load('data/sequences/y_train.npy')
    X_val = np.load('data/sequences/X_val.npy')
    y_val = np.load('data/sequences/y_val.npy')
    X_test = np.load('data/sequences/X_test.npy')
    y_test = np.load('data/sequences/y_test.npy')
    
    print(f"Dados carregados:")
    print(f"  Treino: {X_train.shape}")
    print(f"  Validação: {X_val.shape}")
    print(f"  Teste: {X_test.shape}")
    
    # Criar e construir modelo
    sequence_length = X_train.shape[1]
    n_features = X_train.shape[2]
    
    lstm = StockLSTM(sequence_length=sequence_length, n_features=n_features)
    lstm.build_model(lstm_units=[128, 64], dropout_rate=0.2)
    
    # Treinar modelo
    lstm.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Plotar histórico
    lstm.plot_training_history()
    
    # Carregar scaler
    scaler = joblib.load('data/scaler.pkl')
    
    # Avaliar modelo
    y_pred, metrics = lstm.evaluate(X_test, y_test, scaler=scaler)
    
    # Plotar previsões
    lstm.plot_predictions(y_test, y_pred)
    
    # Salvar modelo
    lstm.save_model('models/lstm_model.keras')
    
    print("\n" + "="*50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("="*50)
    print("\nArquivos gerados:")
    print("  - models/best_model.keras (melhor modelo)")
    print("  - models/lstm_model.keras (modelo final)")
    print("  - models/metrics.csv (métricas)")
    print("  - models/training_history.png (gráfico de treinamento)")
    print("  - models/predictions.png (gráfico de previsões)")

if __name__ == "__main__":
    main()
