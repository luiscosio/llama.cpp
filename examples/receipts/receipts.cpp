// Inference receipts with a Merkle-committed activation trace.
//
// `llama-receipts -m model.gguf -p "..." -n 32 --seed 42 [--trace --openings 32] --out receipt.json`
//   generates text with a replayable sampler and writes a receipt that commits to the model file
//   (whole-file SHA-256 and a Merkle root over its tensors), the exact tokens, the sampler settings,
//   the uniform draw and a hash of the logits at every position. With --trace it also hashes every
//   tensor the forward pass computes into one Merkle tree, puts the root in the receipt, and writes a
//   sidecar with every leaf plus openings for a Fiat-Shamir sample of nodes.
//
// `llama-receipts -m model.gguf --replay receipt.json [--report out.json]`
//   verifies a receipt by re-execution: feeds the claimed tokens back, compares logits hashes, re-runs
//   the seeded sampler at every position and judges the result with a two-regime rule.
//
// The trace sidecar is verified without running the model by verify_trace.py next to this file.
//
// The receipt is not signed here; sign the file with your own key tooling (see README.md).

#include "arg.h"
#include "base64.hpp"
#include "build-info.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json; // std::map keys: dump() is canonical (sorted keys, no whitespace)

static const char * RECEIPT_VERSION   = "0.2";
static const char * TRACE_VERSION     = "trace/v2";
static const char * SAMPLER_DOMAIN    = "llama-receipts/u/v1/";
static const char * CHALLENGE_DOMAIN  = "llama-receipts/trace-challenge/v1/";
static const int    PRODUCER_INPUT    = -1;
static const int    PRODUCER_INITIAL  = -2;

//
// hashing
//

static std::string to_hex(const unsigned char * d, size_t n) {
    static const char * digits = "0123456789abcdef";
    std::string s(n * 2, '0');
    for (size_t i = 0; i < n; ++i) {
        s[2 * i]     = digits[d[i] >> 4];
        s[2 * i + 1] = digits[d[i] & 15];
    }
    return s;
}

static std::string from_hex(const std::string & h) {
    std::string out(h.size() / 2, '\0');
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = (char) std::stoi(h.substr(2 * i, 2), nullptr, 16);
    }
    return out;
}

static std::string sha256_raw(const void * p, size_t n) {
    unsigned char d[SHA256_DIGEST_SIZE];
    sha256_hash(d, (const unsigned char *) p, n);
    return std::string((const char *) d, SHA256_DIGEST_SIZE);
}

static std::string sha256_raw(const std::string & s) { return sha256_raw(s.data(), s.size()); }

static std::string sha256_hex(const void * p, size_t n) {
    std::string raw = sha256_raw(p, n);
    return to_hex((const unsigned char *) raw.data(), raw.size());
}

static std::string sha256_hex(const std::string & s) { return sha256_hex(s.data(), s.size()); }

static void remove_tensor_hashes(json & value) {
    if (value.is_array()) {
        for (auto & item : value) {
            remove_tensor_hashes(item);
        }
        return;
    }
    if (!value.is_object()) {
        return;
    }
    for (auto it = value.begin(); it != value.end();) {
        if (it.key() == "sha256") {
            it = value.erase(it);
        } else {
            remove_tensor_hashes(it.value());
            ++it;
        }
    }
}

static std::string topology_sha256(const json & leaves) {
    json topology = leaves;
    remove_tensor_hashes(topology);
    return sha256_hex(topology.dump());
}

static std::string sha256_file(const std::string & path, std::streamoff offset = 0, std::streamoff length = -1) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    f.seekg(offset);
    sha256_t ctx;
    sha256_init(&ctx);
    std::vector<unsigned char> buf(1 << 22);
    std::streamoff left = length;
    while (left != 0) {
        size_t want = left < 0 ? buf.size() : (size_t) std::min<std::streamoff>(left, (std::streamoff) buf.size());
        f.read((char *) buf.data(), want);
        std::streamsize got = f.gcount();
        if (got <= 0) {
            break;
        }
        sha256_update(&ctx, buf.data(), (size_t) got);
        if (left > 0) {
            left -= got;
        }
    }
    unsigned char d[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, d);
    return to_hex(d, SHA256_DIGEST_SIZE);
}

//
// Merkle tree: sha256(0x00 || leaf), sha256(0x01 || left || right), odd levels duplicate the last node
//

static std::string mk_leaf(const std::string & data) { return sha256_raw(std::string(1, '\0') + data); }

static std::string mk_node(const std::string & l, const std::string & r) { return sha256_raw(std::string(1, '\1') + l + r); }

static std::vector<std::vector<std::string>> mk_levels(const std::vector<std::string> & leaves) {
    std::vector<std::vector<std::string>> levels;
    std::vector<std::string> level;
    level.reserve(leaves.size());
    for (const auto & x : leaves) {
        level.push_back(mk_leaf(x));
    }
    levels.push_back(level);
    while (level.size() > 1) {
        if (level.size() % 2) {
            level.push_back(level.back());
        }
        std::vector<std::string> next;
        next.reserve(level.size() / 2);
        for (size_t i = 0; i + 1 < level.size(); i += 2) {
            next.push_back(mk_node(level[i], level[i + 1]));
        }
        level = std::move(next);
        levels.push_back(level);
    }
    return levels;
}

static std::string mk_root_hex(const std::vector<std::string> & leaves) {
    if (leaves.empty()) {
        return sha256_hex(std::string());
    }
    const std::string root = mk_levels(leaves).back()[0];
    return to_hex((const unsigned char *) root.data(), root.size());
}

static json mk_path(const std::vector<std::vector<std::string>> & levels, size_t index) {
    json path = json::array();
    size_t i = index;
    for (size_t l = 0; l + 1 < levels.size(); ++l) {
        std::vector<std::string> padded = levels[l];
        if (padded.size() % 2) {
            padded.push_back(padded.back());
        }
        const size_t sibling = i ^ 1;
        path.push_back({ sibling < i ? "L" : "R", to_hex((const unsigned char *) padded[sibling].data(), padded[sibling].size()) });
        i /= 2;
    }
    return path;
}

//
// replayable sampler: u_t = SHA-256(domain || seed || t), ties broken by token id
//

struct sampler_cfg {
    double   temperature = 0.7;
    int      top_k       = 40;
    double   top_p       = 0.95;
    uint64_t seed        = 0;

    json to_json() const {
        return { { "type", "seeded-inverse-cdf/v1" }, { "temperature", temperature }, { "top_k", top_k }, { "top_p", top_p }, { "seed", seed } };
    }

    static sampler_cfg from_json(const json & j) {
        sampler_cfg c;
        c.temperature = j.at("temperature").get<double>();
        c.top_k       = j.at("top_k").get<int>();
        c.top_p       = j.at("top_p").get<double>();
        c.seed        = j.at("seed").get<uint64_t>();
        return c;
    }
};

static void put_be64(std::string & s, uint64_t v) {
    for (int i = 7; i >= 0; --i) {
        s.push_back((char) ((v >> (8 * i)) & 0xff));
    }
}

static double uniform_at(uint64_t seed, uint64_t position) {
    std::string msg = SAMPLER_DOMAIN;
    put_be64(msg, seed);
    put_be64(msg, position);
    const std::string d = sha256_raw(msg);
    uint64_t x = 0;
    for (int i = 0; i < 8; ++i) {
        x = (x << 8) | (unsigned char) d[i];
    }
    return (double) x / 18446744073709551616.0; // 2^64
}

struct sampling_dist {
    std::vector<int>    order; // token ids, most likely first
    std::vector<double> probs; // after temperature, top-k, top-p
};

static std::vector<int> ordered_ids(const std::vector<double> & x) {
    std::vector<int> ids(x.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = (int) i;
    }
    std::sort(ids.begin(), ids.end(), [&](int a, int b) { return x[a] > x[b] || (x[a] == x[b] && a < b); });
    return ids;
}

static sampling_dist sampling_distribution(const float * logits, int n_vocab, const sampler_cfg & cfg) {
    std::vector<double> x(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        x[i] = (double) logits[i] / cfg.temperature;
    }
    sampling_dist d;
    d.order = ordered_ids(x);
    if (cfg.top_k > 0 && (int) d.order.size() > cfg.top_k) {
        d.order.resize(cfg.top_k);
    }
    const double m = x[d.order[0]];
    double sum = 0.0;
    d.probs.resize(d.order.size());
    for (size_t i = 0; i < d.order.size(); ++i) {
        d.probs[i] = std::exp(x[d.order[i]] - m);
        sum += d.probs[i];
    }
    for (auto & p : d.probs) {
        p /= sum;
    }
    if (cfg.top_p > 0.0 && cfg.top_p < 1.0) {
        double cum = 0.0;
        size_t keep = d.probs.size();
        for (size_t i = 0; i < d.probs.size(); ++i) {
            cum += d.probs[i];
            if (cum >= cfg.top_p) {
                keep = i + 1;
                break;
            }
        }
        d.order.resize(keep);
        d.probs.resize(keep);
        double s2 = 0.0;
        for (auto p : d.probs) {
            s2 += p;
        }
        for (auto & p : d.probs) {
            p /= s2;
        }
    }
    return d;
}

struct sample_result {
    int    token;
    double logprob;       // under the distribution actually sampled from
    double logprob_full;  // under the full temperature-scaled softmax
    double u;             // the uniform draw, NaN for greedy
    double boundary;      // distance of u to the nearest CDF edge, NaN for greedy
    std::vector<std::pair<int, double>> candidates;
};

static double log_softmax_at(const float * logits, int n_vocab, double temperature, int token) {
    double m = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        m = std::max(m, (double) logits[i] / temperature);
    }
    double z = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        z += std::exp((double) logits[i] / temperature - m);
    }
    return (double) logits[token] / temperature - (std::log(z) + m);
}

static sample_result sample(const float * logits, int n_vocab, const sampler_cfg & cfg, uint64_t position, int top_n = 10) {
    sample_result r;
    if (cfg.temperature <= 0.0) {
        std::vector<double> x(logits, logits + n_vocab);
        const auto order = ordered_ids(x);
        r.token = order[0];
        r.logprob = 0.0;
        r.logprob_full = log_softmax_at(logits, n_vocab, 1.0, r.token);
        r.u = r.boundary = NAN;
        for (int i = 0; i < std::min(top_n, n_vocab); ++i) {
            r.candidates.push_back({ order[i], std::exp(log_softmax_at(logits, n_vocab, 1.0, order[i])) });
        }
        return r;
    }
    const sampling_dist d = sampling_distribution(logits, n_vocab, cfg);
    const double u = uniform_at(cfg.seed, position);
    double cum = 0.0;
    size_t idx = d.probs.size() - 1;
    double lower = 0.0;
    for (size_t i = 0; i < d.probs.size(); ++i) {
        const double next = cum + d.probs[i];
        if (u < next) { // searchsorted(cum, u, side="right")
            idx = i;
            lower = cum;
            cum = next;
            break;
        }
        cum = next;
        lower = cum;
    }
    if (idx == d.probs.size() - 1 && !(u < cum)) { // fell off the end through rounding
        lower = cum - d.probs[idx];
    }
    r.token = d.order[idx];
    r.logprob = std::log(d.probs[idx]);
    r.logprob_full = log_softmax_at(logits, n_vocab, cfg.temperature, r.token);
    r.u = u;
    r.boundary = std::min(cum - u, u - lower);
    for (size_t i = 0; i < std::min((size_t) top_n, d.order.size()); ++i) {
        r.candidates.push_back({ d.order[i], d.probs[i] });
    }
    return r;
}

// [lo, hi) of the unit interval that draws `token`; false if truncated away
static bool cdf_interval(const sampling_dist & d, int token, double & lo, double & hi) {
    double cum = 0.0;
    for (size_t i = 0; i < d.order.size(); ++i) {
        if (d.order[i] == token) {
            lo = cum;
            hi = cum + d.probs[i];
            return true;
        }
        cum += d.probs[i];
    }
    return false;
}

//
// model commitment: whole-file SHA-256, per-tensor SHA-256 read from the GGUF, Merkle root over tensors
//

struct model_commit {
    std::string file_name;
    uint64_t    file_size = 0;
    std::string file_sha256;
    std::string tensor_merkle_root;
    int64_t     n_tensors = 0;
    json        metadata;
    std::unordered_map<std::string, std::string> weight_hash; // tensor name -> sha256 hex
    std::unordered_map<std::string, std::string> weight_type; // tensor name -> ggml type name

    json summary() const {
        return { { "file_name", file_name }, { "file_size", file_size }, { "file_sha256", file_sha256 },
                 { "tensor_merkle_root", tensor_merkle_root }, { "n_tensors", n_tensors }, { "metadata", metadata } };
    }
};

static std::string upper(std::string s) {
    for (auto & c : s) {
        c = (char) toupper((unsigned char) c);
    }
    return s;
}

static model_commit commit_model(const std::string & path) {
    model_commit mc;
    const size_t slash = path.find_last_of("/\\");
    mc.file_name = slash == std::string::npos ? path : path.substr(slash + 1);

    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &ctx_meta };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gp);
    if (!gctx) {
        throw std::runtime_error("cannot read GGUF " + path);
    }
    const size_t data_offset = gguf_get_data_offset(gctx);
    mc.n_tensors = gguf_get_n_tensors(gctx);

    struct tinfo { int64_t i; size_t offset; size_t n_bytes; };
    std::vector<tinfo> order;
    for (int64_t i = 0; i < mc.n_tensors; ++i) {
        const ggml_tensor * cur = ggml_get_tensor(ctx_meta, gguf_get_tensor_name(gctx, i));
        order.push_back({ i, data_offset + gguf_get_tensor_offset(gctx, i), ggml_nbytes(cur) });
    }
    std::sort(order.begin(), order.end(), [](const tinfo & a, const tinfo & b) { return a.offset < b.offset; });

    // one pass over the file: the whole-file hash sees every byte, each tensor hash sees its range
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<unsigned char> buf(1 << 22);
    sha256_t file_ctx;
    sha256_init(&file_ctx);
    size_t pos = 0;
    auto pump = [&](size_t n, sha256_t * tensor_ctx) {
        while (n > 0) {
            const size_t want = std::min(n, buf.size());
            f.read((char *) buf.data(), want);
            const std::streamsize got = f.gcount();
            if (got <= 0) {
                throw std::runtime_error("short read in " + path);
            }
            sha256_update(&file_ctx, buf.data(), (size_t) got);
            if (tensor_ctx) {
                sha256_update(tensor_ctx, buf.data(), (size_t) got);
            }
            n -= (size_t) got;
            pos += (size_t) got;
        }
    };
    std::vector<std::string> tensor_hash(mc.n_tensors);
    for (const auto & t : order) {
        pump(t.offset - pos, nullptr); // header, metadata, alignment padding
        sha256_t tctx;
        sha256_init(&tctx);
        pump(t.n_bytes, &tctx);
        unsigned char d[SHA256_DIGEST_SIZE];
        sha256_final(&tctx, d);
        tensor_hash[t.i] = to_hex(d, SHA256_DIGEST_SIZE);
    }
    for (;;) { // trailing bytes, if any
        f.read((char *) buf.data(), buf.size());
        const std::streamsize got = f.gcount();
        if (got <= 0) {
            break;
        }
        sha256_update(&file_ctx, buf.data(), (size_t) got);
        pos += (size_t) got;
    }
    mc.file_size = pos;
    {
        unsigned char d[SHA256_DIGEST_SIZE];
        sha256_final(&file_ctx, d);
        mc.file_sha256 = to_hex(d, SHA256_DIGEST_SIZE);
    }

    std::vector<std::string> leaves;
    leaves.reserve(mc.n_tensors);
    for (int64_t i = 0; i < mc.n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        const ggml_tensor * cur = ggml_get_tensor(ctx_meta, name);
        json shape = json::array();
        for (int d = 0; d < ggml_n_dims(cur); ++d) {
            shape.push_back(cur->ne[d]);
        }
        json leaf = { { "n_bytes", ggml_nbytes(cur) }, { "name", name }, { "shape", shape }, { "sha256", tensor_hash[i] },
                      { "type", upper(ggml_type_name(cur->type)) } };
        leaves.push_back(leaf.dump());
        mc.weight_hash[name] = tensor_hash[i];
        mc.weight_type[name] = ggml_type_name(cur->type);
    }
    mc.tensor_merkle_root = mk_root_hex(leaves);

    static const char * keys[] = { "general.architecture", "general.name", "general.basename", "general.size_label",
                                   "general.file_type", "general.quantization_version" };
    mc.metadata = json::object();
    for (const char * key : keys) {
        const int64_t kid = gguf_find_key(gctx, key);
        if (kid < 0) {
            continue;
        }
        switch (gguf_get_kv_type(gctx, kid)) {
            case GGUF_TYPE_STRING: mc.metadata[key] = gguf_get_val_str(gctx, kid); break;
            case GGUF_TYPE_UINT32: mc.metadata[key] = gguf_get_val_u32(gctx, kid); break;
            case GGUF_TYPE_INT32:  mc.metadata[key] = gguf_get_val_i32(gctx, kid); break;
            default: break;
        }
    }
    gguf_free(gctx);
    ggml_free(ctx_meta);
    return mc;
}

//
// tracer: one leaf per computing GGML node, Merkle tree over the whole generation
//

static bool is_layout_op(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_RESHAPE || op == GGML_OP_VIEW || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static const ggml_tensor * base_of(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static json ne4(const ggml_tensor * t) { return { t->ne[0], t->ne[1], t->ne[2], t->ne[3] }; }
static json nb4(const ggml_tensor * t) { return { t->nb[0], t->nb[1], t->nb[2], t->nb[3] }; }

static std::string params_hex(const ggml_tensor * t) {
    const unsigned char * p = (const unsigned char *) t->op_params;
    size_t n = GGML_MAX_OP_PARAMS;
    while (n > 0 && p[n - 1] == 0) {
        --n;
    }
    return to_hex(p, n);
}

struct tracer {
    struct write_rec {
        std::string sha256;
        int         leaf;
        size_t      nbytes;
    };
    struct captured {
        std::string out;
        std::vector<std::optional<std::string>> srcs;
    };
    struct stats_t {
        int64_t graphs = 0, nodes = 0, layout_skipped = 0, bytes_hashed = 0, reads = 0, cache_hits = 0, unlinked_node_srcs = 0;
        double  seconds = 0.0;
        json to_json() const {
            return { { "graphs", graphs }, { "nodes", nodes }, { "layout_skipped", layout_skipped }, { "bytes_hashed", bytes_hashed },
                     { "reads", reads }, { "cache_hits", cache_hits }, { "unlinked_node_srcs", unlinked_node_srcs },
                     { "seconds", std::round(seconds * 1000.0) / 1000.0 } };
        }
    };

    const model_commit * commit = nullptr; // weight hashes and types the leaves describe

    std::vector<json>        leaves;
    std::vector<std::string> leaf_bytes;
    std::set<int>            capture;
    std::map<int, captured>  cap;
    std::vector<std::string> errors;
    stats_t                  stats;
    bool                     enabled = true;
    int                      graph = -1;
    int                      node_in_graph = 0;

    std::unordered_map<const void *, write_rec> last_write;
    std::unordered_set<const void *>            persistent_ptrs;
    std::unordered_map<std::string, int>        by_hash;
    json                                        pending = json::array();
    std::vector<std::optional<std::string>>     pending_blobs;

    void reset(std::set<int> capture_ = {}) {
        leaves.clear();
        leaf_bytes.clear();
        cap.clear();
        errors.clear();
        stats = stats_t();
        enabled = true;
        graph = -1;
        node_in_graph = 0;
        last_write.clear();
        persistent_ptrs.clear();
        by_hash.clear();
        pending = json::array();
        pending_blobs.clear();
        capture = std::move(capture_);
    }

    void begin_graph() {
        graph++;
        node_in_graph = 0;
        stats.graphs++;
        // compute memory is reused between graphs; only the KV cache carries state across
        for (auto it = last_write.begin(); it != last_write.end();) {
            it = persistent_ptrs.count(it->first) ? std::next(it) : last_write.erase(it);
        }
        by_hash.clear();
    }

    static bool is_persistent(const char * name) { return strncmp(name, "cache_", 6) == 0; }

    std::string read(const ggml_tensor * b) {
        if (!b->buffer) {
            throw std::runtime_error(std::string("tensor without buffer: ") + ggml_get_name(b));
        }
        const size_t n = ggml_nbytes(b);
        std::string s(n, '\0');
        ggml_backend_tensor_get(b, s.data(), 0, n);
        stats.reads++;
        stats.bytes_hashed += (int64_t) n;
        return s;
    }

    static bool cb(ggml_tensor * t, bool ask, void * user_data) { return static_cast<tracer *>(user_data)->on(t, ask); }

    bool on(const ggml_tensor * t, bool ask) {
        if (!enabled || graph < 0) {
            return !ask; // decline to observe, never abort
        }
        const auto t0 = std::chrono::steady_clock::now();
        bool ret = true;
        try {
            if (is_layout_op(t->op)) {
                if (ask) {
                    stats.layout_skipped++;
                }
                ret = !ask;
            } else if (ask) {
                collect_srcs(t);
            } else {
                record(t);
            }
        } catch (const std::exception & e) {
            errors.push_back("graph " + std::to_string(graph) + " node " + std::to_string(node_in_graph) + ": " + e.what());
        }
        stats.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return ret;
    }

    void collect_srcs(const ggml_tensor * t) {
        pending = json::array();
        pending_blobs.clear();
        const bool capturing = capture.count((int) leaves.size()) > 0;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const ggml_tensor * s = t->src[i];
            if (!s) {
                continue;
            }
            const ggml_tensor * b = base_of(s);
            const size_t offset = (size_t) ((const char *) s->data - (const char *) b->data);
            json entry = { { "name", ggml_get_name(s) }, { "type", ggml_type_name(s->type) }, { "ne", ne4(s) }, { "nb", nb4(s) }, { "offset", offset } };
            json base  = { { "name", ggml_get_name(b) }, { "type", ggml_type_name(b->type) }, { "ne", ne4(b) }, { "nb", nb4(b) }, { "nbytes", ggml_nbytes(b) } };
            const bool persistent = is_persistent(ggml_get_name(b));
            const bool leaf = b->op == GGML_OP_NONE;
            if (leaf && !persistent) {
                auto w = commit->weight_hash.find(ggml_get_name(b));
                if (w != commit->weight_hash.end()) {
                    entry["kind"] = "weight";
                    entry["sha256"] = w->second;
                    entry["type"] = base["type"] = commit->weight_type.at(w->first);
                    entry["base"] = base;
                    pending.push_back(entry);
                    pending_blobs.push_back(std::nullopt);
                    continue;
                }
            }
            std::optional<std::string> blob;
            std::string h;
            int producer;
            auto lw = last_write.find(b->data);
            if (lw != last_write.end() && lw->second.nbytes == ggml_nbytes(b) && (persistent || !leaf)) {
                h = lw->second.sha256;
                producer = lw->second.leaf;
                stats.cache_hits++;
                if (capturing) {
                    blob = read(b);
                }
            } else {
                blob = read(b);
                h = sha256_hex(*blob);
                if (persistent) {
                    producer = PRODUCER_INITIAL;
                    persistent_ptrs.insert(b->data);
                    last_write[b->data] = { h, PRODUCER_INITIAL, ggml_nbytes(b) };
                } else {
                    auto bh = by_hash.find(h);
                    producer = bh == by_hash.end() ? PRODUCER_INPUT : bh->second;
                    if (!leaf) {
                        stats.unlinked_node_srcs++;
                    }
                }
            }
            base["sha256"] = h;
            entry["kind"] = "data";
            entry["base"] = base;
            entry["producer"] = producer;
            pending.push_back(entry);
            pending_blobs.push_back(capturing ? blob : std::nullopt);
        }
    }

    void record(const ggml_tensor * t) {
        const ggml_tensor * b = base_of(t);
        const std::string blob = read(b);
        const std::string h = sha256_hex(blob);
        const int idx = (int) leaves.size();
        json leaf = {
            { "v", 1 }, { "g", graph }, { "i", node_in_graph }, { "name", ggml_get_name(t) }, { "op", ggml_op_desc(t) },
            { "op_name", ggml_op_name(t->op) }, { "params", params_hex(t) },
            { "out", { { "type", ggml_type_name(t->type) }, { "ne", ne4(t) }, { "nb", nb4(t) },
                       { "offset", (size_t) ((const char *) t->data - (const char *) b->data) },
                       { "base", { { "name", ggml_get_name(b) }, { "type", ggml_type_name(b->type) }, { "ne", ne4(b) }, { "nb", nb4(b) },
                                   { "nbytes", ggml_nbytes(b) }, { "sha256", h } } } } },
            { "srcs", pending },
        };
        leaf_bytes.push_back(leaf.dump());
        leaves.push_back(std::move(leaf));
        last_write[b->data] = { h, idx, ggml_nbytes(b) };
        if (is_persistent(ggml_get_name(b))) {
            persistent_ptrs.insert(b->data);
        }
        by_hash[h] = idx;
        if (capture.count(idx)) {
            cap[idx] = { blob, pending_blobs };
        }
        pending = json::array();
        pending_blobs.clear();
        node_in_graph++;
        stats.nodes++;
    }

    std::string root() const { return mk_root_hex(leaf_bytes); }

    json summary() const {
        return { { "version", TRACE_VERSION }, { "root", root() }, { "n_leaves", leaves.size() }, { "n_graphs", graph + 1 },
                 { "topology_sha256", topology_sha256(leaves) },
                 { "leaf_hash", "sha256(0x00 || canonical_json(leaf))" },
                 { "layout_ops_resolved", { "PERMUTE", "RESHAPE", "TRANSPOSE", "VIEW" } }, { "stats", stats.to_json() } };
    }
};

//
// Fiat-Shamir challenge: k distinct leaf indices from the signed root and the receipt
//

static std::vector<int> derive_indices(const std::string & root_hex, const std::string & binding, size_t n_leaves, size_t k,
                                       const std::string & seed = std::string()) {
    std::vector<int> out;
    if (n_leaves == 0) {
        return out;
    }
    k = std::min(k, n_leaves);
    std::set<int> seen;
    uint64_t counter = 0;
    const std::string prefix = std::string(CHALLENGE_DOMAIN) + from_hex(root_hex) + binding + seed;
    while (out.size() < k) {
        std::string msg = prefix;
        put_be64(msg, counter++);
        const std::string d = sha256_raw(msg);
        uint64_t x = 0;
        for (int i = 0; i < 8; ++i) {
            x = (x << 8) | (unsigned char) d[i];
        }
        const int idx = (int) (x % n_leaves);
        if (seen.insert(idx).second) {
            out.push_back(idx);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static std::string challenge_binding(const json & receipt) {
    return from_hex(receipt.at("commitments").at("tokens_sha256").get<std::string>()) +
           from_hex(receipt.at("commitments").at("content_sha256").get<std::string>()) +
           from_hex(receipt.at("model").at("file_sha256").get<std::string>());
}

//
// generation
//

struct engine {
    llama_context *     ctx;
    const llama_model * model;
    const llama_vocab * vocab;
    tracer *            tr;
    int                 n_vocab;
    int                 n_batch;

    void reset() {
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    const float * feed(std::vector<llama_token> tokens) {
        if ((int) tokens.size() > n_batch) {
            throw std::runtime_error("traced prompts must fit in one batch: " + std::to_string(tokens.size()) + " tokens > n_batch " + std::to_string(n_batch));
        }
        if (tr) {
            tr->begin_graph();
        }
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
            throw std::runtime_error("llama_decode failed");
        }
        return llama_get_logits_ith(ctx, -1);
    }
};

struct token_record {
    int           position;
    llama_token   token;
    sample_result res;
    std::string   logits_sha256;

    json to_json() const {
        json cands = json::array();
        for (const auto & c : res.candidates) {
            cands.push_back({ c.first, std::round(c.second * 1e6) / 1e6 });
        }
        json j = { { "position", position }, { "token", token },
                   { "logprob", std::round(res.logprob * 1e6) / 1e6 }, { "logprob_full", std::round(res.logprob_full * 1e6) / 1e6 },
                   { "logits_sha256", logits_sha256 }, { "candidates", cands } };
        if (std::isnan(res.u)) {
            j["u"] = nullptr;
            j["boundary_distance"] = nullptr;
        } else {
            j["u"] = res.u;
            j["boundary_distance"] = std::round(res.boundary * 1e9) / 1e9;
        }
        return j;
    }
};

static std::vector<token_record> generate(engine & e, const std::vector<llama_token> & context, const sampler_cfg & cfg, int n_predict) {
    e.reset();
    if (e.tr) {
        e.tr->reset();
    }
    const float * logits = e.feed(context);
    std::vector<token_record> out;
    for (int i = 0; i < n_predict; ++i) {
        const int position = (int) context.size() + i;
        token_record r;
        r.position = position;
        r.res = sample(logits, e.n_vocab, cfg, (uint64_t) position);
        r.token = r.res.token;
        r.logits_sha256 = sha256_hex(logits, (size_t) e.n_vocab * sizeof(float));
        out.push_back(r);
        if (llama_vocab_is_eog(e.vocab, r.token)) {
            break;
        }
        logits = e.feed({ r.token });
    }
    return out;
}

static std::string iso_now() {
    char buf[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

static json engine_info(const common_params & params) {
    return { { "name", "llama.cpp llama-receipts" }, { "llama_build", llama_build_number() }, { "llama_commit", llama_commit() },
             { "system_info", llama_print_system_info() }, { "backend", params.n_gpu_layers != 0 ? "gpu" : "cpu" },
             { "options", { { "n_gpu_layers", params.n_gpu_layers }, { "n_threads", params.cpuparams.n_threads }, { "n_ctx", params.n_ctx },
                            { "n_batch", params.n_batch }, { "n_ubatch", params.n_ubatch } } } };
}

static json build_receipt(engine & e, const common_params & params, const model_commit & mc, const std::string & prompt_text,
                          const std::vector<llama_token> & prompt_tokens, const sampler_cfg & cfg, const std::vector<token_record> & records,
                          int n_predict) {
    json tokens = json::array();
    json per_token = json::array();
    std::string text;
    for (const auto & r : records) {
        tokens.push_back(r.token);
        per_token.push_back(r.to_json());
        text += common_token_to_piece(e.ctx, r.token, true);
    }
    const json commit_doc = { { "prompt", prompt_tokens }, { "response", tokens } };
    const json request = { { "prompt_text", prompt_text }, { "prompt_tokens", prompt_tokens }, { "sampler", cfg.to_json() }, { "n_predict", n_predict } };
    const json response = { { "tokens", tokens }, { "text", text }, { "per_token", per_token } };
    const json content_doc = { { "prompt_text", prompt_text }, { "response_text", text } };
    json doc = {
        { "receipt_version", RECEIPT_VERSION }, { "created_at", iso_now() }, { "engine", engine_info(params) }, { "model", mc.summary() },
        { "request", request }, { "response", response },
        { "commitments", { { "tokens_sha256", sha256_hex(commit_doc.dump()) }, { "content_sha256", sha256_hex(content_doc.dump()) } } },
    };
    if (e.tr) {
        if (!e.tr->errors.empty()) {
            throw std::runtime_error("trace errors: " + e.tr->errors[0]);
        }
        doc["trace"] = e.tr->summary();
    }
    return doc;
}

// pass two: replay the exact generation, capture the sampled leaves, build the openings
static json open_trace(engine & e, const json & doc, const std::vector<llama_token> & context, int k, const std::string & seed_hex) {
    const json & tr = doc.at("trace");
    const std::string root = tr.at("root").get<std::string>();
    const std::vector<int> indices = derive_indices(root, challenge_binding(doc), tr.at("n_leaves").get<size_t>(), (size_t) k, from_hex(seed_hex));
    e.tr->reset(std::set<int>(indices.begin(), indices.end()));
    e.reset();
    e.feed(context);
    const int n_graphs = tr.at("n_graphs").get<int>();
    const auto & resp = doc.at("response").at("tokens");
    for (int g = 0; g + 1 < n_graphs && g < (int) resp.size(); ++g) {
        e.feed({ resp[g].get<llama_token>() });
    }
    if (!e.tr->errors.empty()) {
        throw std::runtime_error("trace errors during opening replay: " + e.tr->errors[0]);
    }
    if (e.tr->root() != root) {
        throw std::runtime_error("opening replay diverged from the committed trace: " + e.tr->root().substr(0, 16) + " vs " + root.substr(0, 16));
    }
    const auto levels = mk_levels(e.tr->leaf_bytes);
    json openings = json::array();
    for (int idx : indices) {
        auto c = e.tr->cap.find(idx);
        if (c == e.tr->cap.end()) {
            throw std::runtime_error("leaf " + std::to_string(idx) + " was not captured");
        }
        json srcs = json::array();
        for (const auto & s : c->second.srcs) {
            if (s) {
                srcs.push_back(base64::encode(s->data(), s->size()));
            } else {
                srcs.push_back(nullptr);
            }
        }
        openings.push_back({ { "index", idx }, { "path", mk_path(levels, (size_t) idx) }, { "enc", "base64" },
                             { "out", base64::encode(c->second.out.data(), c->second.out.size()) }, { "srcs", srcs } });
    }
    return { { "version", TRACE_VERSION }, { "root", root }, { "n_graphs", n_graphs }, { "topology_sha256", topology_sha256(e.tr->leaves) },
             { "challenge", { { "mode", seed_hex.empty() ? "fiat-shamir" : "interactive" }, { "k", indices.size() }, { "indices", indices }, { "seed", seed_hex } } },
             { "stats", e.tr->stats.to_json() }, { "leaves", e.tr->leaves }, { "openings", openings } };
}

//
// replay verification (re-execution with a trusted reference)
//

struct replay_thresholds {
    double min_token_match       = 0.75;
    double max_mean_abs_dlogprob = 0.25;
    int    max_claimed_truncated = 0;
    double max_mean_gap          = 0.03;
    double max_gap               = 0.45;
};

static int cmd_replay(engine & e, const common_params & params, const std::string & receipt_path, const std::string & report_path) {
    std::ifstream f(receipt_path);
    if (!f) {
        LOG_ERR("cannot open %s\n", receipt_path.c_str());
        return 2;
    }
    json rec;
    f >> rec;

    json report;
    const std::string file_hash = sha256_file(params.model.path);
    const bool model_ok = file_hash == rec.at("model").at("file_sha256").get<std::string>();
    report["model"] = { { "file_sha256", file_hash }, { "file_sha256_match", model_ok } };

    const std::vector<llama_token> prompt = rec.at("request").at("prompt_tokens").get<std::vector<llama_token>>();
    const std::vector<llama_token> response = rec.at("response").at("tokens").get<std::vector<llama_token>>();
    const json & per_token = rec.at("response").at("per_token");
    const sampler_cfg cfg = sampler_cfg::from_json(rec.at("request").at("sampler"));
    const json token_doc = { { "prompt", prompt }, { "response", response } };
    const json content_doc = { { "prompt_text", rec.at("request").at("prompt_text") }, { "response_text", rec.at("response").at("text") } };
    const bool tokens_commitment_ok = sha256_hex(token_doc.dump()) == rec.at("commitments").at("tokens_sha256").get<std::string>();
    const bool content_commitment_ok = sha256_hex(content_doc.dump()) == rec.at("commitments").at("content_sha256").get<std::string>();

    std::string verdict = "accept", reason;
    if (!tokens_commitment_ok || !content_commitment_ok) {
        verdict = "reject";
        reason = "receipt content commitment mismatch";
    } else if (!model_ok) {
        verdict = "reject";
        reason = "model hash mismatch";
    } else if (response.empty() || per_token.size() != response.size()) {
        verdict = "reject";
        reason = "malformed receipt";
    }

    json rows = json::array();
    int n = (int) response.size(), n_match = 0, n_hash = 0, n_trunc = 0, n_mismatch = 0;
    double sum_dlp = 0.0, max_dlp = 0.0, sum_gap = 0.0, max_gap = 0.0;
    int n_dlp = 0;
    if (verdict == "accept") {
        e.reset();
        const float * logits = e.feed(prompt);
        for (int i = 0; i < n; ++i) {
            const json & claimed = per_token[i];
            const int position = (int) prompt.size() + i;
            const sample_result res = sample(logits, e.n_vocab, cfg, (uint64_t) position);
            const bool match = res.token == response[i];
            const bool hash_match = sha256_hex(logits, (size_t) e.n_vocab * sizeof(float)) == claimed.at("logits_sha256").get<std::string>();
            double lp = -INFINITY, gap = 0.0;
            bool in_support = true;
            if (cfg.temperature > 0.0) {
                const sampling_dist d = sampling_distribution(logits, e.n_vocab, cfg);
                double lo, hi;
                in_support = cdf_interval(d, response[i], lo, hi);
                if (in_support) {
                    for (size_t j = 0; j < d.order.size(); ++j) {
                        if (d.order[j] == response[i]) {
                            lp = std::log(d.probs[j]);
                        }
                    }
                    gap = (res.u >= lo && res.u < hi) ? 0.0 : (res.u < lo ? lo - res.u : res.u - hi);
                }
            } else {
                in_support = match;
                lp = match ? 0.0 : -INFINITY;
            }
            n_match += match;
            n_hash += hash_match;
            if (!in_support) {
                n_trunc++;
            } else {
                const double dlp = std::fabs(lp - claimed.at("logprob").get<double>());
                sum_dlp += dlp;
                max_dlp = std::max(max_dlp, dlp);
                n_dlp++;
            }
            if (!match) {
                n_mismatch++;
                sum_gap += gap;
                max_gap = std::max(max_gap, gap);
            }
            rows.push_back({ { "i", i }, { "claimed", response[i] }, { "replayed", res.token }, { "match", match }, { "logits_hash_match", hash_match },
                             { "replay_logprob_of_claimed", std::isinf(lp) ? json(nullptr) : json(lp) }, { "claimed_gap", gap } });
            logits = e.feed({ response[i] });
        }
        const double token_match = (double) n_match / n;
        const double hash_rate = (double) n_hash / n;
        const double mean_dlp = n_dlp ? sum_dlp / n_dlp : 0.0;
        const double mean_gap = sum_gap / n;
        replay_thresholds th;
        char buf[256];
        if (n_trunc > th.max_claimed_truncated) {
            verdict = "reject";
            snprintf(buf, sizeof(buf), "%d claimed tokens outside the sampler's support", n_trunc);
            reason = buf;
        } else if (hash_rate == 1.0) {
            if (n_match < n) {
                verdict = "reject";
                snprintf(buf, sizeof(buf), "logits are bit-identical but the sampler disagrees at %d positions", n - n_match);
                reason = buf;
            } else {
                reason = "bit-exact replay";
            }
        } else if (max_gap > th.max_gap) {
            verdict = "reject";
            snprintf(buf, sizeof(buf), "a draw sits %.2f of CDF mass from the claimed token (> %.2f)", max_gap, th.max_gap);
            reason = buf;
        } else if (mean_gap > th.max_mean_gap) {
            verdict = "reject";
            snprintf(buf, sizeof(buf), "mean draw gap %.4f > %.2f", mean_gap, th.max_mean_gap);
            reason = buf;
        } else if (token_match < th.min_token_match) {
            verdict = "reject";
            snprintf(buf, sizeof(buf), "token match %.3f < %.2f", token_match, th.min_token_match);
            reason = buf;
        } else if (mean_dlp > th.max_mean_abs_dlogprob) {
            verdict = "reject";
            snprintf(buf, sizeof(buf), "mean |dlogprob| %.3f > %.2f", mean_dlp, th.max_mean_abs_dlogprob);
            reason = buf;
        } else {
            reason = "within backend drift";
        }
        report["summary"] = { { "n", n }, { "token_match_rate", token_match }, { "logits_hash_match_rate", hash_rate },
                              { "n_claimed_truncated", n_trunc }, { "mean_abs_dlogprob", mean_dlp }, { "max_abs_dlogprob", max_dlp },
                              { "mean_gap_all", mean_gap }, { "mismatch_gap_max", n_mismatch ? json(max_gap) : json(nullptr) } };
        LOG("tokens: %d  match rate: %.3f  logits-hash match: %.3f  mean|dlogprob|: %.4f  truncated: %d  mean gap: %.4f  max gap: %.3f\n",
            n, token_match, hash_rate, mean_dlp, n_trunc, mean_gap, max_gap);
    }
    report["verdict"] = verdict;
    report["reason"] = reason;
    report["rows"] = rows;
    LOG("verdict: %s (%s)\n", verdict.c_str(), reason.c_str());
    if (!report_path.empty()) {
        std::ofstream o(report_path);
        o << report.dump(1) << "\n";
    }
    return verdict == "accept" ? 0 : 1;
}

//
// main
//

struct extra_args {
    bool        trace = false;
    int         openings = 32;
    std::string out;
    std::string replay;
    std::string report;
    std::string challenge_seed; // hex; empty means Fiat-Shamir from the root and receipt
    std::string claim_model;    // test aid: describe this other GGUF in the receipt and leaves
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  prove:  %s -m model.gguf -p \"prompt\" -n 32 --seed 42 --temp 0.7 --top-k 40 --top-p 0.95 [--trace] [--openings 32] --out receipt.json\n", argv[0]);
    LOG("\n  replay: %s -m model.gguf --replay receipt.json [--report report.json]\n", argv[0]);
    LOG("\n  --trace writes <out minus .json>.trace.json next to the receipt; verify it with verify_trace.py\n");
    LOG("  --challenge-seed HEX derives the openings from a verifier-supplied seed instead of the root alone\n");
    LOG("  --claim-model other.gguf (test aid) writes a receipt that claims another model was run, to exercise verifiers\n\n");
}

static bool take_flag(std::vector<char *> & args, size_t & i, const char * name, std::string * value) {
    if (strcmp(args[i], name) != 0) {
        return false;
    }
    if (value) {
        if (i + 1 >= args.size()) {
            throw std::runtime_error(std::string(name) + " needs a value");
        }
        *value = args[i + 1];
        args.erase(args.begin() + i, args.begin() + i + 2);
    } else {
        args.erase(args.begin() + i);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    extra_args ex;
    std::vector<char *> args(argv, argv + argc);
    try {
        std::string v;
        for (size_t i = 1; i < args.size();) {
            if (take_flag(args, i, "--trace", nullptr)) {
                ex.trace = true;
            } else if (take_flag(args, i, "--openings", &v)) {
                ex.openings = std::stoi(v);
            } else if (take_flag(args, i, "--out", &v)) {
                ex.out = v;
            } else if (take_flag(args, i, "--replay", &v)) {
                ex.replay = v;
            } else if (take_flag(args, i, "--report", &v)) {
                ex.report = v;
            } else if (take_flag(args, i, "--challenge-seed", &v)) {
                ex.challenge_seed = v;
            } else if (take_flag(args, i, "--claim-model", &v)) {
                ex.claim_model = v;
            } else {
                ++i;
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }
    if (ex.replay.empty() && ex.out.empty()) {
        LOG_ERR("--out receipt.json is required when proving\n");
        return 1;
    }
    if (ex.trace && ex.openings <= 0) {
        LOG_ERR("--openings must be positive\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // one graph per decode call keeps the tracer's graph index exact
    params.n_ubatch = params.n_batch;
    params.warmup = false;

    tracer tr;
    model_commit mc;
    try {
        const auto t0 = ggml_time_us();
        mc = commit_model(ex.claim_model.empty() ? params.model.path : ex.claim_model);
        LOG_INF("model committed: %s, %" PRId64 " tensors, %.2f s\n", mc.file_sha256.substr(0, 16).c_str(), mc.n_tensors, (ggml_time_us() - t0) / 1e6);
        if (!ex.claim_model.empty()) {
            LOG_INF("TEST AID: the receipt will claim %s while running %s\n", ex.claim_model.c_str(), params.model.path.c_str());
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        return 1;
    }
    tr.commit = &mc;
    if (ex.trace) {
        params.cb_eval = tracer::cb;
        params.cb_eval_user_data = &tr;
    }

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    engine e = { ctx, model, vocab, ex.trace ? &tr : nullptr, llama_vocab_n_tokens(vocab), (int) llama_n_batch(ctx) };

    int ret = 0;
    try {
        if (!ex.replay.empty()) {
            ret = cmd_replay(e, params, ex.replay, ex.report);
        } else {
            sampler_cfg cfg;
            cfg.temperature = params.sampling.temp;
            cfg.top_k = params.sampling.top_k;
            cfg.top_p = params.sampling.top_p;
            cfg.seed = params.sampling.seed == LLAMA_DEFAULT_SEED ? (uint64_t) std::time(nullptr) : params.sampling.seed;
            const int n_predict = params.n_predict < 0 ? 64 : params.n_predict;

            const std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
            const auto t_gen = ggml_time_us();
            const std::vector<token_record> records = generate(e, prompt_tokens, cfg, n_predict);
            LOG_INF("generated %zu tokens in %.2f s%s\n", records.size(), (ggml_time_us() - t_gen) / 1e6, ex.trace ? " with tracing" : "");
            json doc = build_receipt(e, params, mc, params.prompt, prompt_tokens, cfg, records, n_predict);
            {
                std::ofstream o(ex.out);
                if (!o) {
                    throw std::runtime_error("cannot write " + ex.out);
                }
                o << doc.dump(1) << "\n";
            }
            LOG("%s\n", doc["response"]["text"].get<std::string>().c_str());
            LOG("\n[receipt] %zu tokens, sha256 %s, saved to %s\n", records.size(), sha256_hex(doc.dump()).substr(0, 16).c_str(), ex.out.c_str());
            if (ex.trace) {
                const auto t_open = ggml_time_us();
                const json tdoc = open_trace(e, doc, prompt_tokens, ex.openings, ex.challenge_seed);
                LOG_INF("opening replay in %.2f s\n", (ggml_time_us() - t_open) / 1e6);
                std::string tpath = ex.out;
                if (tpath.size() > 5 && tpath.compare(tpath.size() - 5, 5, ".json") == 0) {
                    tpath.resize(tpath.size() - 5);
                }
                tpath += ".trace.json";
                std::ofstream o(tpath);
                o << tdoc.dump() << "\n";
                LOG("[trace] root %s, %s leaves over %d graphs, %d openings, saved to %s\n", doc["trace"]["root"].get<std::string>().substr(0, 16).c_str(),
                    std::to_string(doc["trace"]["n_leaves"].get<size_t>()).c_str(), doc["trace"]["n_graphs"].get<int>(), ex.openings, tpath.c_str());
            }
        }
    } catch (const std::exception & err) {
        LOG_ERR("%s\n", err.what());
        ret = 1;
    }

    llama_backend_free();
    return ret;
}
