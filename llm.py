"""LLM เดียวที่ใช้ทั้ง Part 1 (baseline) และ Part 2 (improved) — Gemma 4 31B ผ่าน Gemini API (OpenAI-compatible)

ตั้งค่า: GEMINI_API_KEY (หรือใส่ในไฟล์ .env ที่ root ของ repo)
"""
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

_env = Path(__file__).with_name(".env")
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

MODEL = os.environ.get("LLM_MODEL", "gemma-4-31b-it")
_client = OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


# LLM_CACHE=path.json -> เก็บคำตอบไว้ รันซ้ำได้ผลเดิมโดยไม่เรียก API (ใช้ตอน ingest / วัดผล)
_cache_path = os.environ.get("LLM_CACHE")
_cache = json.loads(Path(_cache_path).read_text(encoding="utf-8")) if _cache_path and Path(_cache_path).exists() else {}
_lock = threading.Lock()


def chat(prompt, max_tokens=500, temperature=0.2, system="You are a helpful assistant."):
    key = hashlib.sha256(f"{MODEL}|{system}|{temperature}|{max_tokens}|{prompt}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    out = _chat(prompt, max_tokens, temperature, system)
    if _cache_path:
        with _lock:
            _cache[key] = out
            Path(_cache_path).write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
    return out


def _chat(prompt, max_tokens, temperature, system):
    # Gemma 4 "คิด" ก่อนตอบ (<thought>...</thought>) และ thought นับรวมใน max_tokens
    # จึงต้องเผื่อ token ให้ส่วนคิด แล้วตัดส่วนคิดทิ้ง เหลือเฉพาะคำตอบ
    for attempt in range(15):
        try:
            r = _client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=max_tokens + 4000, temperature=temperature)
            text = r.choices[0].message.content or ""
            return re.sub(r"(?s)<thought>.*?(</thought>|$)", "", text).strip()
        except (RateLimitError, InternalServerError, APIConnectionError) as e:
            # ponytail: free tier = 16k input tokens/นาที (429) หรือ server ยุ่ง (503) — รอแล้วลองใหม่
            m = re.search(r"retry in ([\d.]+)s", str(e))
            time.sleep(float(m.group(1)) + 2 if m else 30)
    raise RuntimeError("LLM rate-limited too many times")
