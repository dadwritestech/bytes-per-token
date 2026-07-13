// route-trace: dump per-(context-step, candidate-rank, layer) routed expert IDs.
//
// Measures Thesis A (amortize expert I/O across a *breadth* of candidate tokens):
// at each real generation step we take the top-K most likely next tokens (the natural
// draft set), and for each candidate we run one decode step at the SAME context and
// capture that candidate's per-layer routed expert set (the `ffn_moe_topk-<il>` tensor,
// == final global expert IDs, group-mask already applied for glm-dsa/deepseek routing).
//
// Output CSV: step,rank,token,layer,n_used,eids
// Analysis (union growth vs K) lives in the new-inference-engine repo.
//
// Custom knobs via env (keeps common arg parser happy):
//   RT_K      candidates per step               (default 8)
//   RT_STEPS  real generation steps to sample    (default 16)
//   RT_WARM   greedy tokens generated before first sample (default 0)
//   RT_OUT    output CSV path                     (default route_trace.csv)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---- capture state --------------------------------------------------------
// For a captured "ffn_moe_topk-<il>" tensor of shape [n_used, n_cand] we keep the
// flattened i32 ids plus the two dims, so the main loop can split by candidate column.
struct layer_cap { int n_used = 0; int n_cand = 0; std::vector<int32_t> ids; };

static bool                             g_capture = false;   // true only during candidate decodes
static std::map<int, layer_cap>         g_layer_cap;         // layer -> captured ids for current decode
static std::vector<uint8_t>             g_scratch;

static int env_int(const char * k, int def) {
    const char * v = getenv(k);
    return v ? atoi(v) : def;
}

// scheduler node callback: grab every "ffn_moe_topk-<il>" tensor while capturing.
static bool rt_capture_cb(struct ggml_tensor * t, bool ask, void * /*user*/) {
    if (!g_capture) {
        return false; // not interested; do not request data
    }
    const char * name = t->name;
    if (strncmp(name, "ffn_moe_topk-", 13) != 0) {
        return false;
    }
    if (ask) {
        return true; // yes, give me the computed data
    }
    // ask == false: data is ready. Parse layer index from the name suffix.
    int il = atoi(name + 13);
    const int64_t n_used = t->ne[0]; // n_expert_used
    const int64_t n_cand = t->ne[1]; // one column per candidate token in the batch
    if (t->type != GGML_TYPE_I32) {
        return true;
    }
    const size_t n_bytes = ggml_nbytes(t);
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const int32_t * ids;
    if (is_host) {
        ids = (const int32_t *) t->data;
    } else {
        if (g_scratch.size() < n_bytes) g_scratch.resize(n_bytes);
        ggml_backend_tensor_get(t, g_scratch.data(), 0, n_bytes);
        ids = (const int32_t *) g_scratch.data();
    }
    if (getenv("RT_DBG") && il == 0) {
        fprintf(stderr, "[DBG] %s type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] nbytes=%zu host=%d\n",
                name, (int)t->type,
                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                t->nb[0], t->nb[1], t->nb[2], t->nb[3], n_bytes, (int)is_host);
        fprintf(stderr, "[DBG] first %d int32: ", (int)std::min<int64_t>(40, n_bytes/4));
        for (int i = 0; i < std::min<int64_t>(40, n_bytes/4); ++i) fprintf(stderr, "%d ", ids[i]);
        fprintf(stderr, "\n");
    }
    // ffn_moe_topk is a top-k VIEW of a full [n_expert, n_cand] argsort, so columns are
    // strided by nb[1] (e.g. 256 int32), NOT n_used. Index candidate r, slot j accordingly.
    const size_t col_stride = t->nb[1] / sizeof(int32_t);
    layer_cap lc;
    lc.n_used = (int) n_used;
    lc.n_cand = (int) n_cand;
    lc.ids.resize((size_t)(n_used * n_cand));
    for (int64_t r = 0; r < n_cand; ++r) {
        for (int64_t j = 0; j < n_used; ++j) {
            lc.ids[(size_t)(r * n_used + j)] = ids[(size_t)(r * col_stride + j)];
        }
    }
    g_layer_cap[il] = std::move(lc);
    return true;
}

// decode a single token in seq 0 at pos (used for warmup + greedy commit). no capture.
static bool decode_commit(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens     = 1;
    b.token[0]     = tok;
    b.pos[0]       = pos;
    b.n_seq_id[0]  = 1;
    b.seq_id[0][0] = 0;
    b.logits[0]    = 1;
    g_capture = false;
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// decode one token into a specific sequence at a specific pos, WITH capture on.
// fills g_layer_cap (n_cand==1). Returns rc==0 on success.
static bool decode_capture_seq(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens     = 1;
    b.token[0]     = tok;
    b.pos[0]       = pos;
    b.n_seq_id[0]  = 1;
    b.seq_id[0][0] = seq;
    b.logits[0]    = 1;
    g_layer_cap.clear();
    g_capture = true;
    int rc = llama_decode(ctx, b);
    g_capture = false;
    llama_batch_free(b);
    return rc == 0;
}

// top-w token ids from the current last-decode logits
static std::vector<llama_token> topw(llama_context * ctx, int n_vocab, int w) {
    const float * lg = llama_get_logits_ith(ctx, -1);
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;
    const int ww = std::min(w, n_vocab);
    std::partial_sort(idx.begin(), idx.begin() + ww, idx.end(),
                      [&](int a, int b){ return lg[a] > lg[b]; });
    return std::vector<llama_token>(idx.begin(), idx.begin() + ww);
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    const int    K     = env_int("RT_K", 8);
    const int    STEPS = env_int("RT_STEPS", 16);
    const int    WARM  = env_int("RT_WARM", 0);
    const char * OUT   = getenv("RT_OUT") ? getenv("RT_OUT") : "route_trace.csv";

    // TREE mode (RT_TREE_D>0): expand a real w-ary depth-D speculative tree and dump every
    // node's routed expert set, so union_tree(w,D) I/O cost can be measured (Thesis A2).
    const int TREE_W = env_int("RT_TREE_W", 2);
    const int TREE_D = env_int("RT_TREE_D", 0);
    // upper bound on live sequences = total tree nodes + slack
    long tree_nodes = 0; { long p = 1; for (int d = 0; d < TREE_D; ++d) { p *= TREE_W; tree_nodes += p; } }

    // each candidate needs its own KV sequence seeded from the prefix (seq 0),
    // so candidates share context but never reuse a position (M-RoPE safe).
    const int need_seq = (TREE_D > 0) ? (int)std::min<long>(tree_nodes + 2, 4096) : (K + 1);
    params.n_parallel = std::max(params.n_parallel, need_seq);
    params.n_ctx      = std::max<uint32_t>(params.n_ctx, (uint32_t)(need_seq * 256));

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = rt_capture_cb;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    auto   init  = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("failed to init model/ctx\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, add_bos, true);
    if (toks.empty()) { LOG_ERR("no input tokens (use -p)\n"); return 1; }

    // Thesis B working-set mode (RT_WS_GEN>0): capture routing for prompt tokens AND a greedy
    // generation, labeled by phase, so per-prompt working-set size and prompt→generation expert
    // overlap can be measured.
    const int WS_GEN = env_int("RT_WS_GEN", 0);

    FILE * out = fopen(OUT, "w");
    if (!out) { LOG_ERR("cannot open RT_OUT=%s\n", OUT); return 1; }
    fprintf(out, (WS_GEN > 0) ? "phase,pos,layer,n_used,eids\n"
                : (TREE_D > 0) ? "node,parent,depth,layer,n_used,eids\n"
                              : "step,rank,token,layer,n_used,eids\n");

    // --- prefill the prompt (seq 0, positions 0..P-1); capture iff WS mode ---
    g_capture = (WS_GEN > 0);
    g_layer_cap.clear();
    {
        llama_batch pb = llama_batch_get_one(toks.data(), (int32_t) toks.size());
        if (llama_decode(ctx, pb)) { LOG_ERR("prefill decode failed\n"); return 1; }
    }
    g_capture = false;
    llama_pos n_past = (llama_pos) toks.size();

    auto greedy_from_last = [&](void) -> llama_token {
        const float * lg = llama_get_logits_ith(ctx, -1);
        llama_token best = 0; float bv = -1e30f;
        for (int i = 0; i < n_vocab; ++i) { if (lg[i] > bv) { bv = lg[i]; best = i; } }
        return best;
    };

    // ---- WS mode: dump prompt routing (captured above) + greedy generation routing ----
    if (WS_GEN > 0) {
        // prompt tokens: g_layer_cap holds [n_used, P] per layer
        for (auto & kv : g_layer_cap) {
            const int il = kv.first; const layer_cap & lc = kv.second;
            for (int c = 0; c < lc.n_cand; ++c) {
                fprintf(out, "0,%d,%d,%d,", c, il, lc.n_used);
                for (int j = 0; j < lc.n_used; ++j) fprintf(out, "%d%s", lc.ids[(size_t)c*lc.n_used+j], (j+1<lc.n_used)?" ":"");
                fprintf(out, "\n");
            }
        }
        // greedy generation with capture, committed to seq 0
        for (int t = 0; t < WS_GEN; ++t) {
            llama_token g = greedy_from_last();
            llama_batch b = llama_batch_init(1, 0, 1);
            b.n_tokens=1; b.token[0]=g; b.pos[0]=n_past; b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=1;
            g_layer_cap.clear(); g_capture = true;
            int rc = llama_decode(ctx, b); g_capture = false; llama_batch_free(b);
            if (rc != 0) { LOG_ERR("ws gen decode failed\n"); return 1; }
            for (auto & kv : g_layer_cap) {
                const int il = kv.first; const layer_cap & lc = kv.second;
                fprintf(out, "1,%d,%d,%d,", (int)n_past, il, lc.n_used);
                for (int j = 0; j < lc.n_used; ++j) fprintf(out, "%d%s", lc.ids[j], (j+1<lc.n_used)?" ":"");
                fprintf(out, "\n");
            }
            n_past++;
            if ((t % 16) == 0) LOG_INF("ws gen %d/%d\n", t, WS_GEN);
        }
        fclose(out);
        LOG_INF("route-trace WS: wrote %s (prompt=%d gen=%d)\n", OUT, (int)toks.size(), WS_GEN);
        llama_backend_free();
        return 0;
    }

    // optional warmup: advance greedily to reach a mid-generation context
    for (int w = 0; w < WARM; ++w) {
        llama_token g = greedy_from_last();
        if (!decode_commit(ctx, g, n_past)) { LOG_ERR("warmup decode failed\n"); return 1; }
        n_past++;
    }

    llama_memory_t mem0 = llama_get_memory(ctx);

    // ---- TREE mode: expand a real w-ary depth-D tree, dump every node's routing ----
    if (TREE_D > 0) {
        LOG_INF("tree mode: W=%d D=%d (~%ld nodes), context P=%d\n", TREE_W, TREE_D, tree_nodes, (int)n_past);
        int next_seq = 1;
        long emitted = 0;
        auto emit_node = [&](int node_id, int parent_id, int depth) {
            for (auto & kv : g_layer_cap) {
                const int il = kv.first; const layer_cap & lc = kv.second;
                fprintf(out, "%d,%d,%d,%d,%d,", node_id, parent_id, depth, il, lc.n_used);
                for (int j = 0; j < lc.n_used; ++j) fprintf(out, "%d%s", lc.ids[j], (j+1<lc.n_used)?" ":"");
                fprintf(out, "\n");
            }
        };
        // DFS: 'src_seq' holds context [0..pos-1]; current valid logits are src_seq's last decode.
        std::function<void(llama_seq_id,llama_pos,int,int)> expand =
            [&](llama_seq_id src_seq, llama_pos pos, int depth, int parent_id) {
            if (depth <= 0) return;
            std::vector<llama_token> kids = topw(ctx, n_vocab, TREE_W); // snapshot BEFORE any decode
            for (llama_token ct : kids) {
                if (next_seq + 1 >= params.n_parallel) { LOG_INF("seq budget hit; truncating tree\n"); return; }
                llama_seq_id cs = next_seq++;
                llama_memory_seq_rm(mem0, cs, -1, -1);
                llama_memory_seq_cp(mem0, src_seq, cs, 0, -1);       // inherit parent context
                if (!decode_capture_seq(ctx, ct, pos, cs)) { LOG_ERR("tree decode failed\n"); return; }
                int node_id = (int) ++emitted;                       // 1-based node id
                emit_node(node_id, parent_id, TREE_D - depth);       // depth 0 = root's children
                expand(cs, pos + 1, depth - 1, node_id);             // recurse (uses child's logits)
            }
        };
        expand(0, n_past, TREE_D, 0);                                 // root = prefix context, parent 0
        fclose(out);
        LOG_INF("route-trace TREE: wrote %s (%ld node-rows across layers)\n", OUT, emitted);
        llama_backend_free();
        return 0;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    const int kk = std::min(K, n_vocab);

    // --- sampling loop: at each step capture routing for the top-K candidates ---
    // The K candidates are decoded in ONE batch, each in its own KV sequence (1..K)
    // seeded from the shared prefix (seq 0). This is exactly a Thesis-A verify batch.
    for (int step = 0; step < STEPS; ++step) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        std::vector<int> idx(n_vocab);
        for (int i = 0; i < n_vocab; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b){ return lg[a] > lg[b]; });

        // seed one scratch sequence per candidate from the prefix (seq 0, pos 0..n_past-1)
        for (int r = 0; r < kk; ++r) {
            llama_memory_seq_rm(mem, r + 1, -1, -1);
            llama_memory_seq_cp(mem, 0, r + 1, 0, -1); // full-sequence copy (this KV cache only supports full)
        }

        // one batch: K candidate tokens, each at pos n_past in its own sequence
        llama_batch b = llama_batch_init(kk, 0, 1);
        b.n_tokens = kk;
        const bool sametok = getenv("RT_SAMETOK") != nullptr; // debug: all candidates = rank0 token
        for (int r = 0; r < kk; ++r) {
            b.token[r]     = sametok ? idx[0] : idx[r];
            b.pos[r]       = n_past;
            b.n_seq_id[r]  = 1;
            b.seq_id[r][0] = r + 1;
            b.logits[r]    = 1;
        }
        g_layer_cap.clear();
        g_capture = true;
        int rc = llama_decode(ctx, b);
        g_capture = false;
        llama_batch_free(b);
        if (rc != 0) { LOG_ERR("candidate batch decode failed (rc=%d)\n", rc); return 1; }

        // emit: one CSV row per (layer, candidate rank)
        for (auto & kv : g_layer_cap) {
            const int         il = kv.first;
            const layer_cap & lc = kv.second;
            for (int r = 0; r < lc.n_cand && r < kk; ++r) {
                fprintf(out, "%d,%d,%d,%d,%d,", step, r, (int)idx[r], il, lc.n_used);
                for (int j = 0; j < lc.n_used; ++j) {
                    // element (j, r) of an [n_used, n_cand] i32 tensor
                    fprintf(out, "%d%s", lc.ids[(size_t)r * lc.n_used + j], (j+1<lc.n_used) ? " " : "");
                }
                fprintf(out, "\n");
            }
        }

        // clear scratch sequences and commit the greedy (rank-0) token in seq 0
        for (int r = 0; r < kk; ++r) llama_memory_seq_rm(mem, r + 1, -1, -1);
        if (!decode_commit(ctx, idx[0], n_past)) { LOG_ERR("commit decode failed\n"); return 1; }
        n_past++;

        if ((step % 4) == 0) LOG_INF("step %d/%d done (n_past=%d)\n", step, STEPS, (int)n_past);
        fflush(out);
    }

    fclose(out);
    LOG_INF("route-trace: wrote %s (K=%d STEPS=%d WARM=%d)\n", OUT, K, STEPS, WARM);
    llama_backend_free();
    return 0;
}
