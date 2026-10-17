#!/bin/bash
# โหลดโมเดลลง /models (volume) ครั้งแรก แล้วเปิด JEV-9B + reranker ที่ :8766
set -e
if [ ! -f /models/JEV-9B/head.safetensors ]; then
  echo "downloading autotrust/JEV-9B (~18 GB)..."
  hf download autotrust/JEV-9B --local-dir /models/JEV-9B \
    --include "*.json" "*.jinja" "model-*.safetensors" "head.safetensors" "adapter/*"
fi
if [ ! -f /models/bge-reranker-v2-m3/config.json ]; then
  echo "downloading BAAI/bge-reranker-v2-m3 (~2 GB)..."
  hf download BAAI/bge-reranker-v2-m3 --local-dir /models/bge-reranker-v2-m3
fi
exec python part2_improved/jev_local_server.py /models/JEV-9B
