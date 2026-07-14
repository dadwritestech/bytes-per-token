@echo off
rem A/B: GLM-4.5-Air UD-Q2_K_XL fully resident (VRAM + RAM) vs GLM-5.2 streamed baseline (0.9 tok/s).
rem Measured 2026-07-14: tg 19.5 tok/s. Let -fit place tensors (manual -ngl/-ts disables it and OOMs).
rem Usage: bench_air.cmd ["prompt"]
set PROMPT=%~1
if "%PROMPT%"=="" set PROMPT=The history of the Roman Empire is a story of gradual expansion followed by
"D:/Local/llama build/llama.cpp/build/bin/Release/llama-completion.exe" ^
  -m D:/Local/models/GLM-4.5-Air/GLM-4.5-Air-UD-Q2_K_XL.gguf ^
  --no-mmap -c 4096 -n 128 -t 4 --temp 0 ^
  -p "%PROMPT%"
