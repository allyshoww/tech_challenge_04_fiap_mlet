#!/bin/bash

echo "🚀 Iniciando API..."
echo ""
echo "   URL: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Aguarde carregar o modelo..."
echo ""

source .venv/bin/activate
python3 api/app.py
