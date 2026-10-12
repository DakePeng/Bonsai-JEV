// decide-logits-mm: decide-logits with images. Each stdin JSONL line is
//   {"prompt": "...<__media__>...", "images": ["/path/a.jpg", ...]}
// where each <__media__> marker (mtmd_default_marker) is replaced by the next image, in order.
// Prints {"logits":[...], "n_tokens": N, "argmax": id} with next-token logits for env CANDS ids.
// Usage: CANDS=32,33 ./llama-decide-logits-mm -m model.gguf --mmproj mmproj.gguf -ngl 99 -c 8192 < in.jsonl > out.jsonl
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
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
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * lctx  = llama_init->context();
    if (model == nullptr || lctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu          = params.mmproj_use_gpu;
    mparams.device           = params.mmproj_device;
    mparams.print_timings    = false;
    mparams.n_threads        = params.cpuparams.n_threads;
    mparams.flash_attn_type  = params.flash_attn_type;
    mparams.warmup           = false;
    mparams.image_min_tokens = params.image_min_tokens;
    mparams.image_max_tokens = params.image_max_tokens;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!ctx_vision) {
        LOG_ERR("failed to load mmproj %s\n", params.mmproj.path.c_str());
        return 1;
    }

    const int n_ctx = llama_n_ctx(lctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(lctx);

    std::string line;
    size_t done = 0, skipped = 0;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json in = json::parse(line);
        std::string prompt = in["prompt"].get<std::string>();
        json out;

        mtmd::bitmaps bitmaps;
        bool ok = true;
        if (in.contains("images")) {
            for (const auto & p : in["images"]) {
                auto res = mtmd_helper_bitmap_init_from_file(ctx_vision.get(), p.get<std::string>().c_str(), false);
                if (!res.bitmap) {
                    ok = false;
                    break;
                }
                bitmaps.entries.emplace_back(res.bitmap);
            }
        }
        if (!ok) {
            out["error"] = "image load failed";
            std::cout << out.dump() << "\n";
            ++skipped;
            continue;
        }

        mtmd_input_text text;
        text.text          = prompt.data();
        text.text_len      = prompt.size();
        text.add_special   = false;
        text.parse_special = false;
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        auto bptr = bitmaps.c_ptr();
        if (mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &text, bptr.data(), bptr.size()) != 0) {
            out["error"] = "tokenize failed";
            std::cout << out.dump() << "\n";
            ++skipped;
            continue;
        }
        const size_t n_tokens = mtmd_helper_get_n_tokens(chunks.ptr.get());
        if ((int) n_tokens >= n_ctx) {
            out["error"] = "too long";
            out["n_tokens"] = n_tokens;
            std::cout << out.dump() << "\n";
            ++skipped;
            continue;
        }
        llama_memory_clear(mem, true);
        llama_pos new_n_past = 0;
        if (mtmd_helper_eval_chunks(ctx_vision.get(), lctx, chunks.ptr.get(), 0, 0, params.n_batch, true, &new_n_past) != 0) {
            out["error"] = "eval failed";
            std::cout << out.dump() << "\n";
            ++skipped;
            continue;
        }
        const float * logits = llama_get_logits_ith(lctx, -1);
        std::vector<float> sel;
        for (auto id : cands) sel.push_back(logits[id]);
        out["logits"] = sel;
        out["n_tokens"] = n_tokens;
        out["argmax"] = (int) (std::max_element(logits, logits + n_vocab) - logits);
        std::cout << out.dump() << "\n";
        if (++done % 100 == 0) {
            std::cout.flush();
            LOG_INF("%zu done, %zu skipped\n", done, skipped);
        }
    }
    std::cout.flush();
    LOG_INF("finished: %zu done, %zu skipped\n", done, skipped);
    llama_backend_free();
    return 0;
}
