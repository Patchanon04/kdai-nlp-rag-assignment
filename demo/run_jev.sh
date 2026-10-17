#!/bin/bash
# JEV-9B + bge-reranker บน GPU (Linux/WSL + ROCm) — เรียกจาก demo/start.ps1 หรือรันเองก็ได้
# มี memory guard: ถ้า RAM ใน WSL เหลือน้อยกว่า 900 MB จะฆ่าตัวเองทิ้ง ดีกว่าปล่อยให้ WSL ค้างลาก container อื่นล่มด้วย
# ponytail: path ของ venv/โมเดลเป็นของเครื่องที่ทำงานนี้ — เครื่องอื่นตั้ง VENV / JEV_MODEL / RERANKER เอง
cd "$(dirname "$0")/.."
export RERANKER=${RERANKER:-$HOME/models/bge-reranker-v2-m3}
export HF_DEACTIVATE_ASYNC_LOAD=1 HSA_ENABLE_DXG_DETECTION=1 HF_HUB_OFFLINE=1
source "${VENV:-$HOME/venv-rocm}/bin/activate"
python part2_improved/jev_local_server.py "${JEV_MODEL:-$HOME/models/JEV-9B}" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt 900 ]; then echo "MEMGUARD: MemAvailable ${avail}MB, killing $pid"; kill -9 $pid; fi
  sleep 2
done
echo "EXITED"
