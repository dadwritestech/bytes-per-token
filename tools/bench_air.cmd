@echo off
rem A/B: GLM-4.5-Air UD-Q2_K_XL fully resident (VRAM + RAM) vs GLM-5.2 streamed baseline (0.9 tok/s).
rem Usage: bench_air.cmd [N_CPU_MOE]   (default 24; raise if CUDA OOM, lower if RAM-tight)
set NCM=%1
if "%NCM%"=="" set NCM=24
"D:/Local/llama build/llama.cpp/build/bin/Release/llama-completion.exe" ^
  -m D:/Local/models/GLM-4.5-Air/GLM-4.5-Air-UD-Q2_K_XL.gguf ^
  -ngl 99 --n-cpu-moe %NCM% -ts 1,1 --no-mmap ^
  -c 4096 -n 128 -t 4 --temp 0 ^
  -p "The history of the Roman Empire is a story of gradual expansion followed by"
