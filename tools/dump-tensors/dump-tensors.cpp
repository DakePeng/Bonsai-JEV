// dump-tensors: run one prompt through the PrismML llama.cpp fork and write selected graph
// tensors (inp_embd, l_out-N, result_norm, result_output) as raw little-endian f32 files.
// Output dir from env DUMP_DIR (default "."). Token ids go to DUMP_DIR/tokens.txt.
// Built from examples/eval-callback; only the callback differs.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct dump_ctx {
    std::string dir;
};

// DUMP_LAYERS="1,3" additionally dumps every tensor whose name ends in "-1" or "-3" (per-layer intermediates)
static std::vector<std::string> g_layer_suffixes;

static bool wanted(const char * name) {
    std::string n(name);
    if (n == "inp_embd" || n == "embd" || n == "result_norm" || n == "result_output" || n.rfind("l_out-", 0) == 0) {
        return true;
    }
    for (const auto & suf : g_layer_suffixes) {
        if (n.size() > suf.size() && n.compare(n.size() - suf.size(), suf.size(), suf) == 0) {
            return true;
        }
    }
    return false;
}

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * d = (dump_ctx *) user_data;
    if (ask) {
        return wanted(t->name);
    }
    if (!wanted(t->name)) {
        return true;
    }
    const size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    if (!ggml_is_contiguous(t)) {
        LOG_ERR("skip non-contiguous %s\n", t->name);
        return true;
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        ggml_get_type_traits(t->type)->to_float(raw.data(), buf.data(), n);
    }
    std::ofstream f(d->dir + "/" + t->name + ".f32", std::ios::binary);
    f.write((const char *) buf.data(), n * sizeof(float));
    LOG_INF("dumped %-16s ne=[%lld,%lld] %s\n", t->name, (long long) t->ne[0], (long long) t->ne[1], ggml_type_name(t->type));
    return true;
}

static bool run(llama_context * ctx, const common_params & params, const std::string & dir) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("no input tokens\n");
        return false;
    }
    std::ofstream tf(dir + "/tokens.txt");
    for (auto tok : tokens) {
        tf << tok << "\n";
    }
    LOG_INF("number of input tokens = %zu\n", tokens.size());

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("failed to eval\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    dump_ctx d;
    const char * env = std::getenv("DUMP_DIR");
    d.dir = env ? env : ".";
    if (const char * ls = std::getenv("DUMP_LAYERS")) {
        std::string cur;
        for (const char * p = ls; ; ++p) {
            if (*p == ',' || *p == 0) {
                if (!cur.empty()) g_layer_suffixes.push_back("-" + cur);
                cur.clear();
                if (*p == 0) break;
            } else {
                cur += *p;
            }
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = dump_cb;
    params.cb_eval_user_data = &d;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }

    if (!run(ctx, params, d.dir)) {
        return 1;
    }
    llama_backend_free();
    return 0;
}
