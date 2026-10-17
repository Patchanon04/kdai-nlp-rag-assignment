# เปิด demo ทั้งชุดบนเครื่องนี้: Docker (OpenSearch + app) + JEV-9B/reranker บน GPU ใน WSL แล้วเปิดเบราว์เซอร์
# ใช้: ดับเบิลคลิก demo\start.cmd   (หรือ powershell -ExecutionPolicy Bypass -File demo\start.ps1)
$ErrorActionPreference = 'Stop'
$root   = Split-Path $PSScriptRoot -Parent
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$distro = 'Ubuntu'

function EngineUp { & $docker info *> $null; return $LASTEXITCODE -eq 0 }
function WaitEngine($tries) { for ($i = 0; $i -lt $tries; $i++) { if (EngineUp) { return $true }; Start-Sleep 6 }; return $false }
function Wsl([string]$cmd) { (& wsl.exe -d $distro -- bash -c $cmd) -replace "`0", '' }

# 0) LDPlayer ใช้ GPU memory ร่วม — เปิดอยู่ JEV-9B มักโหลดไม่ขึ้น (HIP out of memory)
if (Get-Process Ld9BoxHeadless -ErrorAction SilentlyContinue) {
    Write-Warning 'LDPlayer เปิดอยู่ — ปิดก่อน ไม่งั้น JEV-9B อาจโหลดไม่ขึ้น (คอลัมน์ ③ จะขึ้นว่ายังไม่เชื่อมต่อ)'
}

# 1) Docker Desktop
if (-not (EngineUp)) {
    Write-Host '[1/4] เปิด Docker Desktop...'
    Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    if (-not (WaitEngine 40)) {
        # socket ค้างจากรอบก่อน ("The file cannot be accessed by the system") — ย้ายโฟลเดอร์ socket ออกแล้วเปิดใหม่
        Write-Host '      Docker ไม่ขึ้น — ย้ายโฟลเดอร์ socket ที่ค้างออกแล้วเปิดใหม่'
        Get-Process | Where-Object { $_.Name -match '^(Docker Desktop|com\.docker\..*)$' } | Stop-Process -Force
        Start-Sleep 4
        $ts = Get-Date -Format yyyyMMddHHmmss
        foreach ($d in "$env:LOCALAPPDATA\Docker\run", "$env:LOCALAPPDATA\docker-secrets-engine") {
            if (Test-Path $d) { Rename-Item $d ("{0}.old-$ts" -f (Split-Path $d -Leaf)) }
        }
        Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
        if (-not (WaitEngine 50)) { throw 'Docker ยังไม่ขึ้น — เปิด Docker Desktop ดู error เอง' }
    }
}
Write-Host '[1/4] Docker พร้อม'

# 2) JEV-9B + reranker ใน WSL (ถ้ายังไม่รัน)
$script = Wsl "wslpath -a '$($root -replace '\\','/')/demo/run_jev.sh'"
if (-not (Wsl 'pgrep -f [j]ev_local_server || true')) {
    Write-Host '[2/4] สตาร์ท JEV-9B บน GPU (โหลด ~2-3 นาที)...'
    Wsl "setsid nohup bash '$script' > ~/jev_server.log 2>&1 & sleep 2" | Out-Null
} else { Write-Host '[2/4] JEV-9B รันอยู่แล้ว' }

# 3) IP ของ WSL เปลี่ยนได้หลังรีสตาร์ท — อัปเดต .env (UTF-8 ไม่มี BOM ไม่งั้น key บรรทัดแรกพัง)
$ip = (Wsl 'hostname -I').Trim().Split(' ')[0]
$envFile = Join-Path $root '.env'
$lines = [IO.File]::ReadAllLines($envFile) | Where-Object { $_ -notmatch '^(JEV_URL|RERANK_URL)=' }
$lines += "JEV_URL=http://${ip}:8766/score", "RERANK_URL=http://${ip}:8766/rerank"
[IO.File]::WriteAllLines($envFile, $lines, (New-Object Text.UTF8Encoding $false))
Write-Host "[3/4] WSL IP = $ip -> .env"

# 4) OpenSearch + app (compose สร้าง app ใหม่เองถ้าค่าใน .env เปลี่ยน)
Push-Location $root
& $docker compose -p nlp-rag up -d 2>&1 | Out-Host
Pop-Location
Write-Host '[4/4] รอ app และ JEV-9B พร้อม...'
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod http://localhost:8091/api/health -TimeoutSec 3 | Out-Null; break } catch { Start-Sleep 5 }
}
for ($i = 0; $i -lt 60; $i++) {
    $log = Wsl "grep -aE 'smoke:|MEMGUARD|EXITED|Error:' ~/jev_server.log | tail -1"
    if ($log) { break }
    Start-Sleep 5
}
if ($log -match 'smoke:') { Write-Host "JEV-9B พร้อม ($log)" -ForegroundColor Green }
else { Write-Warning "JEV-9B ไม่ขึ้น: $log — demo ยังใช้ได้แต่คอลัมน์ ③ จะขึ้นว่ายังไม่เชื่อมต่อ (ดู ~/jev_server.log ใน WSL)" }

Start-Process 'http://localhost:8091'
Write-Host 'เปิด http://localhost:8091 แล้ว' -ForegroundColor Green
