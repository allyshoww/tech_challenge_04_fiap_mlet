# Tech Challenge Fase 4 - Previsão de Preços de Ações com LSTM

Projeto de previsão de preços de ações da Amazon (AMZN) utilizando redes neurais LSTM.

## Estrutura do Projeto

```
.
├── data/                      # Dados coletados e processados
│   ├── sequences/             # Sequências temporais para LSTM
│   ├── AMZN_raw.csv          # Dados brutos
│   ├── AMZN_processed.csv    # Dados processados
│   ├── AMZN_analysis.png     # Gráficos de análise
│   └── scaler.pkl            # Normalizador
├── models/                    # Modelos treinados
│   ├── best_model.keras      # Melhor modelo
│   ├── lstm_model.keras      # Modelo final
│   ├── metrics.csv           # Métricas de avaliação
│   ├── training_history.png  # Histórico de treinamento
│   └── predictions.png       # Gráfico de previsões
├── api/                       # API REST
│   ├── app.py                # Aplicação FastAPI
│   └── requirements.txt      # Dependências da API
├── data_collection.py         # Script de coleta e pré-processamento
├── prepare_sequences.py       # Preparação de sequências temporais
├── lstm_model.py             # Modelo LSTM
├── train_pipeline.py         # Pipeline completo de treinamento
├── test_api.py               # Testes da API
├── monitoring.py             # Monitoramento da API
├── Dockerfile                # Configuração Docker
├── docker-compose.yml        # Orquestração Docker
├── deploy_instructions.md    # Instruções de deploy
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação
```

## Instalação

1. Crie um ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Coleta e Pré-processamento de Dados

```bash
python data_collection.py
```

Este script irá:
- Coletar dados históricos da Amazon (AMZN) desde 2015
- Realizar análise exploratória
- Criar features derivadas (médias móveis, volatilidade, etc.)
- Normalizar os dados
- Dividir em conjuntos de treino, validação e teste

### 2. Preparar Sequências Temporais

```bash
python prepare_sequences.py
```

Cria sequências de 60 dias para alimentar o modelo LSTM.

### 3. Treinar Modelo LSTM

```bash
python lstm_model.py
```

Treina o modelo LSTM e gera:
- Modelo treinado (.keras)
- Métricas de avaliação (MAE, RMSE, MAPE)
- Gráficos de treinamento e previsões

### Pipeline Completo

Execute tudo de uma vez:

```bash
python train_pipeline.py
```

## API REST

### Iniciar API Localmente

```bash
# Opção 1: Python direto
cd api
python app.py

# Opção 2: Docker Compose (recomendado)
docker-compose up -d
```

A API estará disponível em: http://localhost:8000

### Documentação da API

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

- `GET /` - Informações da API
- `GET /health` - Health check
- `GET /model/info` - Informações do modelo
- `POST /predict` - Fazer previsões

### Exemplo de Uso

```python
import requests

# Fazer previsão para 5 dias
response = requests.post(
    "http://localhost:8000/predict",
    json={"days_ahead": 5}
)

data = response.json()
print(f"Preço atual: ${data['current_price']:.2f}")
for date, price in zip(data['prediction_dates'], data['predicted_prices']):
    print(f"{date}: ${price:.2f}")
```

### Testar API

```bash
python test_api.py
```

## Monitoramento

```bash
# Monitorar continuamente (intervalo de 60s)
python monitoring.py

# Monitorar por 5 minutos
python monitoring.py --duration 300

# Gerar relatório
python monitoring.py --report
```

## Deploy

Consulte [deploy_instructions.md](deploy_instructions.md) para instruções detalhadas de deploy em:
- AWS (Elastic Beanstalk)
- Google Cloud Platform (Cloud Run)
- Azure (Container Instances)
- Heroku

### Próximos Passos

- [x] Coleta e pré-processamento de dados
- [x] Desenvolvimento do modelo LSTM
- [x] Treinamento e avaliação
- [x] Salvamento do modelo
- [x] Criação da API
- [x] Docker e Docker Compose
- [x] Monitoramento
- [ ] Deploy em produção
- [ ] Vídeo explicativo

## Empresa Selecionada

**Amazon (AMZN)**
- Período: 2015-01-01 até presente
- Features: Open, High, Low, Close, Volume + features derivadas

## Validação do Modelo

### Métricas de Performance

**Previsão de Preço:** ✅ EXCELENTE
- MAE: $5.15
- RMSE: $6.57
- MAPE: 2.37%

**Previsão de Direção (Alta/Baixa):** ⚠️ LIMITADO
- Acurácia: 48.82%
- O modelo foi otimizado para prever PREÇOS, não direção

### Como Usar Corretamente

✅ **Use para:**
- Estimar preços futuros (alta precisão)
- Avaliar magnitude de mudanças
- Calcular intervalos de confiança
- Suporte à decisão de investimento

❌ **NÃO use para:**
- Decisões automáticas de compra/venda
- Confiar apenas na direção prevista
- Trading sem análise adicional

### Validação Completa

Execute o script de validação:
```bash
python validate_predictions.py
```

Documentação detalhada: [VALIDACAO_MODELO.md](VALIDACAO_MODELO.md)

## Tecnologias

- Python 3.x
- TensorFlow/Keras (LSTM)
- yfinance (coleta de dados)
- scikit-learn (pré-processamento)
- FastAPI (API REST)
- Docker (containerização)
