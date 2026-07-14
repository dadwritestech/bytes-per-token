# corpus/ (artifacts not distributed)

Two files used by the Thesis-C experiments are intentionally not in this repo:

- `qwopus.imatrix` (192 MB) — importance matrix; regenerate with
  `llama-imatrix -m <Qwopus Q6_K> -f <calibration text> -o corpus/qwopus.imatrix`.
- `ptb.test.txt` — Penn Treebank test split (LDC-licensed; obtain via LDC99T42 or use
  wikitext-2 instead; the log's PPL deltas, not absolute values, carry the conclusions).
