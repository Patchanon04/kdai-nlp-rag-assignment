# Demo + ingestion: React UI (demo/web) + FastAPI (demo/server.py) + demo/init.py (สร้าง index จาก cache)
# JEV-9B แยกเป็น Dockerfile.jev (ต้องใช้ GPU NVIDIA)
FROM node:22-alpine AS web
WORKDIR /web
COPY demo/web/package.json demo/web/package-lock.json ./
RUN npm ci
COPY demo/web/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
# torch แบบ CPU ก่อน ไม่งั้น requirements.txt จะดึง torch+CUDA หลาย GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt reqs/ingest.txt
COPY demo/requirements.txt reqs/demo.txt
RUN pip install --no-cache-dir -r reqs/ingest.txt -r reqs/demo.txt
COPY llm.py ./
COPY corpus_input/ corpus_input/
COPY eval/ eval/
COPY part1_baseline/ part1_baseline/
COPY part2_improved/improvedRAG.py part2_improved/
COPY results/llm_cache.json results/improved_contexts.json results/
COPY demo/server.py demo/init.py demo/
COPY demo/web/public/ demo/web/public/
COPY --from=web /web/dist demo/web/dist
ENV PYTHONUNBUFFERED=1 HF_HOME=/hf HF_HUB_DISABLE_SYMLINKS_WARNING=1 LLM_CACHE=/app/results/llm_cache.json
EXPOSE 8080
CMD ["python", "demo/server.py"]
