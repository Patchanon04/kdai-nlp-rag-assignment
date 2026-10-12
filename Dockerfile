# Demo service: React UI (demo/web) + FastAPI (demo/server.py) — JEV-9B รันแยกบน GPU (part2_improved/jev_local_server.py)
FROM node:22-alpine AS web
WORKDIR /web
COPY demo/web/package.json demo/web/package-lock.json ./
RUN npm ci
COPY demo/web/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt
COPY llm.py ./
COPY corpus_input/ corpus_input/
COPY eval/questions.json eval/
COPY part1_baseline/onlysearchAuthenticRAG.py part1_baseline/
COPY part2_improved/improvedRAG.py part2_improved/
COPY demo/server.py demo/
COPY demo/web/public/ demo/web/public/
COPY --from=web /web/dist demo/web/dist
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
EXPOSE 8080
CMD ["python", "demo/server.py"]
