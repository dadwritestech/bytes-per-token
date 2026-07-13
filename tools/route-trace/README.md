# route-trace — per-(step,candidate,layer) expert-ID tracer

Lives in the llama.cpp oracle fork as `examples/route-trace` (built via its CMake;
`add_subdirectory(route-trace)` in `examples/CMakeLists.txt`). Snapshotted here for provenance.

Captures the `ffn_moe_topk-<il>` tensor via a sched eval callback. At each generation step it
seeds K scratch KV sequences from the shared prefix and decodes the top-K next tokens as one
batch (a Thesis-A verify batch), dumping each candidate's per-layer routed expert set to CSV.

Env knobs: `RT_K` (candidates), `RT_STEPS` (gen steps), `RT_WARM` (greedy warmup),
`RT_OUT` (csv path), `RT_SAMETOK`=1 (debug: all candidates=rank0, must give A(K)=K),
`RT_DBG`=1 (dump raw tensor layout for layer 0).

Run (Qwopus): `RT_K=16 RT_STEPS=32 RT_WARM=8 RT_OUT=t.csv llama-route-trace -m <gguf> -ngl 99 -cmoe -c 8704 -p "<prompt>"`
Run (GLM): add `GGML_MOE_STREAM_GB=12 --override-kv glm-dsa.expert_used_count=int:4`.
Analyze: `python analysis/union_growth.py t.csv [more.csv ...]`.
