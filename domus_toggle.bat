@echo off
title Domus Voice System
echo ============================================
echo   DOMUS VOICE SYSTEM
echo ============================================
echo.

:: === AVVIO ===
echo [1/3] Avvio DomusHarness...
nssm restart DomusHarness >nul 2>&1
timeout /t 3 /nobreak >nul
for /f "tokens=*" %%s in ('nssm status DomusHarness 2^>nul') do set HSTATUS=%%s
echo       Harness: %HSTATUS%

echo [2/3] Avvio DomusSTT (caricamento modello GPU ~40s)...
nssm restart DomusSTT >nul 2>&1
echo       STT: avviato

echo [3/3] Avvio DomusEarDaemon...
nssm restart DomusEarDaemon >nul 2>&1
timeout /t 3 /nobreak >nul
for /f "tokens=*" %%s in ('nssm status DomusEarDaemon 2^>nul') do set ESTATUS=%%s
echo       EarDaemon: %ESTATUS%

:: === WARMUP LLM ===
echo.
echo [WARMUP] Caricamento modello LLM in GPU...
powershell -Command "try { $null = Invoke-RestMethod -Uri 'http://localhost:1234/v1/chat/completions' -Method Post -ContentType 'application/json' -Headers @{Authorization='Bearer giorgio-local-manager'} -Body '{\"model\":\"mellum2-12b-a2.5b\",\"messages\":[{\"role\":\"user\",\"content\":\"Ciao\"}],\"max_tokens\":5}' -TimeoutSec 180; Write-Host '       LLM: OK' } catch { Write-Host '       LLM: timeout o errore' }"

:: === ATTESA STT ===
echo [WARMUP] Attesa caricamento modello STT...
:warmup_stt
timeout /t 5 /nobreak >nul
powershell -Command "try { $h = Invoke-RestMethod -Uri 'http://localhost:8090/health' -TimeoutSec 5; if ($h.loaded) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
    echo       STT: modello caricato!
) else (
    echo       STT: caricamento...
    goto warmup_stt
)

echo.
echo ============================================
echo   DOMUS ATTIVO - Sistema pronto
echo ============================================
echo.
echo Premi un tasto per FERMARE tutto...
pause >nul

:: === STOP ===
echo.
echo ============================================
echo   Arresto servizi...
echo ============================================

echo [1/3] Stop DomusEarDaemon...
nssm stop DomusEarDaemon >nul 2>&1
timeout /t 2 /nobreak >nul
echo       EarDaemon: fermato

echo [2/3] Stop DomusSTT...
nssm stop DomusSTT >nul 2>&1
timeout /t 2 /nobreak >nul
echo       STT: fermato

echo [3/3] Stop DomusHarness...
nssm stop DomusHarness >nul 2>&1
timeout /t 2 /nobreak >nul
echo       Harness: fermato

echo.
echo ============================================
echo   DOMUS FERMO
echo ============================================
timeout /t 3 /nobreak >nul
