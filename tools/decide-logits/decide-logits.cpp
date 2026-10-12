// decide-logits: for each JSONL line {"prompt": "..."} on stdin, run the prompt through the model
// (no special tokens added, like JEV's add_special_tokens=False) and print one JSON line with the
// next-token logits for the candidate token ids given in env CANDS="id,id,...":
//   {"logits":[...], "n_tokens": N, "argmax": id}
// Usage: CANDS=... ./llama-decide-logits -m model.gguf -ngl 99 -c 4096 < prompts.jsonl > out.jsonl
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <clocale>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::vector<llama_token> cands;
    if (const char * c = std::getenv("CANDS")) {
        std::string cur;
        for (const char * p = c; ; ++p) {
            if (*p == ',' || *p == 0) {
                if (!cur.empty()) cands.push_back(std::stoi(cur));
                cur.clear();
                if (*p == 0) break;
            } else {
                cur += *p;
            }
        }
    }
    if (cands.empty()) {
        fprintf(stderr, "CANDS env var (comma-separated token ids) is required\n");
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }
    const int n_ctx = llama_n_ctx(ctx);
    llama_memory_t mem = llama_get_memory(ctx);

    std::string line;
    size_t done = 0, skipped = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json in = json::parse(line);
        std::string prompt = in["prompt"].get<std::string>();
        std::vector<llama_token> toks = common_tokenize(ctx, prompt, /*add_special*/ false, /*parse_special*/ false);
        json out;
        if ((int) toks.size() >= n_ctx || toks.empty()) {
            out["error"] = "too long";
            out["n_tokens"] = toks.size();
            std::cout << out.dump() << "\n";
            ++skipped;
            continue;
        }
        llama_memory_clear(mem, true);
        llama_batch batch = llama_batch_get_one(toks.data(), toks.size());
        if (llama_decode(ctx, batch)) {
            out["error"] = "decode failed";
            std::cout << out.dump() << "\n";
            ++skipped;
            continue;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        std::vector<float> sel;
        for (auto id : cands) sel.push_back(logits[id]);
        const int am = (int) (std::max_element(logits, logits + n_vocab) - logits);
        out["logits"] = sel;
        out["n_tokens"] = toks.size();
        out["argmax"] = am;
        std::cout << out.dump() << "\n";
        if (++done % 500 == 0) {
            std::cout.flush();
            LOG_INF("%zu done, %zu skipped\n", done, skipped);
        }
    }
    std::cout.flush();
    LOG_INF("finished: %zu done, %zu skipped\n", done, skipped);
    llama_backend_free();
    return 0;
}
