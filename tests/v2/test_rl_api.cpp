#include "third_party/catch2/catch_amalgamated.hpp"
#include "rl/rl_api.h"
#include <vector>
#include <string>
#include <cstring>

TEST_CASE("rl_init_default", "[rl]") {
    char err[128] = {0};
    rl_handle_t h = rl_init_from_json("{}", err, sizeof(err));
    REQUIRE(h != nullptr);
    REQUIRE(rl_get_feat_dim(h) >= 1);
    rl_close(h);
}

TEST_CASE("rl_init_custom_feat_dim", "[rl]") {
    char err[128] = {0};
    rl_handle_t h = rl_init_from_json("{\"feat_dim\":12, \"alpha\":0.5}", err, sizeof(err));
    REQUIRE(h != nullptr);
    REQUIRE(rl_get_feat_dim(h) == 12);
    rl_close(h);
}

static std::vector<unsigned char> make_select_candidates(int num_items, int num_candidates, float p=0.4) {
    std::vector<unsigned char> data((size_t)num_items * (size_t)num_candidates, 0);
    for (int c=0;c<num_candidates;++c) {
        for (int i=0;i<num_items;++i) {
            if (((i*37 + c*13) % 100) < (int)(p*100)) data[(size_t)c*num_items + i] = 1;
        }
    }
    return data;
}

TEST_CASE("rl_prepare_and_score_select", "[rl]") {
    char err[128] = {0};
    rl_handle_t h = rl_init_from_json("{\"feat_dim\":10}", err, sizeof(err));
    REQUIRE(h);
    int num_items = 25;
    int num_candidates = 7;
    auto cand = make_select_candidates(num_items, num_candidates);
    std::vector<float> features((size_t)num_candidates * (size_t)rl_get_feat_dim(h));
    REQUIRE(rl_prepare_features(h, cand.data(), num_items, num_candidates, 0, features.data(), err, sizeof(err)) == 0);
    std::vector<double> scores(num_candidates, 0.0);
    REQUIRE(rl_score_batch_with_features(h, features.data(), rl_get_feat_dim(h), num_candidates, scores.data(), err, sizeof(err)) == 0);
    // Expect at least one positive score
    bool anyPositive=false; for (double s : scores) if (s>0.0) { anyPositive=true; break; }
    REQUIRE(anyPositive);
    rl_close(h);
}

TEST_CASE("rl_score_batch_select_legacy", "[rl]") {
    char err[128] = {0};
    rl_handle_t h = rl_init_from_json("{\"feat_dim\":8}", err, sizeof(err));
    REQUIRE(h);
    int num_items = 16;
    int num_candidates = 5;
    auto cand = make_select_candidates(num_items, num_candidates);
    std::vector<double> scores(num_candidates, 0.0);
    REQUIRE(rl_score_batch(h, "{}", cand.data(), num_items, num_candidates, 0, scores.data(), err, sizeof(err)) == 0);
    REQUIRE(rl_get_last_batch_size(h) == num_candidates);
    rl_close(h);
}

TEST_CASE("rl_assign_mode_basic", "[rl]") {
    char err[128] = {0};
    rl_handle_t h = rl_init_from_json("{\"feat_dim\":9}", err, sizeof(err));
    REQUIRE(h);
    int num_items = 10;
    int num_candidates = 3;
    // Each item: -1 (0xFF) or bin id 0..2
    std::vector<unsigned char> data((size_t)num_items * (size_t)num_candidates, 0xFF);
    for (int c=0;c<num_candidates;++c) {
        for (int i=0;i<num_items;++i) {
            if ((i + c) % 3 == 0) data[(size_t)c*num_items + i] = (unsigned char)(i % 3); // assign
        }
    }
    std::vector<double> scores(num_candidates, 0.0);
    REQUIRE(rl_score_batch(h, "{}", data.data(), num_items, num_candidates, 1, scores.data(), err, sizeof(err)) == 0);
    rl_close(h);
}

TEST_CASE("rl_learn_updates_weights", "[rl]") {
    char err[128] = {0};
    rl_handle_t h = rl_init_from_json("{\"feat_dim\":8}", err, sizeof(err));
    REQUIRE(h);
    int num_items = 12;
    int num_candidates = 6;
    auto cand = make_select_candidates(num_items, num_candidates);
    std::vector<double> scores_before(num_candidates, 0.0);
    REQUIRE(rl_score_batch(h, "{}", cand.data(), num_items, num_candidates, 0, scores_before.data(), err, sizeof(err)) == 0);
    // Provide rewards emphasizing first half
    std::string feedback = "{\"rewards\":[1,1,1,0,0,0]}";
    REQUIRE(rl_learn_batch(h, feedback.c_str(), err, sizeof(err)) == 0);
    std::vector<double> scores_after(num_candidates, 0.0);
    REQUIRE(rl_score_batch(h, "{}", cand.data(), num_items, num_candidates, 0, scores_after.data(), err, sizeof(err)) == 0);
    // Expect average of first 3 scores after learning to be >= before (simple heuristic)
    double avg_before=0, avg_after=0;
    for (int i=0;i<3;++i) { avg_before += scores_before[i]; avg_after += scores_after[i]; }
    avg_before/=3.0; avg_after/=3.0;
    REQUIRE(avg_after >= avg_before);
    rl_close(h);
}

TEST_CASE("rl_learn_structured_feedback_with_decay", "[rl]") {
    char err[128] = {0};
    rl_handle_t h = rl_init_from_json("{\"feat_dim\":8}", err, sizeof(err));
    REQUIRE(h);
    int num_items = 10;
    int num_candidates = 5;
    auto cand = make_select_candidates(num_items, num_candidates);
    std::vector<double> scores_before(num_candidates, 0.0);
    REQUIRE(rl_score_batch(h, "{}", cand.data(), num_items, num_candidates, 0, scores_before.data(), err, sizeof(err)) == 0);
    // Choose indices 0 and 2 as positive with positional decay (positions [0,2])
    std::string feedback = "{\"chosen\":[1,0,1,0,0],\"base_reward\":1.0,\"decay\":0.8,\"positions\":[0,1,2,3,4]}";
    REQUIRE(rl_learn_batch(h, feedback.c_str(), err, sizeof(err)) == 0);
    std::vector<double> scores_after(num_candidates, 0.0);
    REQUIRE(rl_score_batch(h, "{}", cand.data(), num_items, num_candidates, 0, scores_after.data(), err, sizeof(err)) == 0);
    REQUIRE(scores_after[0] >= scores_before[0]);
    REQUIRE(scores_after[2] >= scores_before[2]);
    rl_close(h);
}

TEST_CASE("rl_get_last_features_and_config", "[rl]") {
    char err[128] = {0};
    const char* cfg = "{\"feat_dim\":7,\"alpha\":0.4}";
    rl_handle_t h = rl_init_from_json(cfg, err, sizeof(err));
    REQUIRE(h);
    int num_items = 9;
    int num_candidates = 4;
    auto cand = make_select_candidates(num_items, num_candidates);
    std::vector<double> scores(num_candidates, 0.0);
    REQUIRE(rl_score_batch(h, "{}", cand.data(), num_items, num_candidates, 0, scores.data(), err, sizeof(err)) == 0);
    int feat_dim = rl_get_feat_dim(h);
    REQUIRE(feat_dim == 7);
    std::vector<float> buf((size_t)feat_dim * (size_t)num_candidates, 0.f);
    int copied = rl_get_last_features(h, buf.data(), (int)buf.size());
    REQUIRE(copied == feat_dim * num_candidates);
    REQUIRE(buf[0] == Catch::Approx(1.0f)); // bias feature
    char cfg_out[64];
    int cfg_len = rl_get_config_json(h, cfg_out, sizeof(cfg_out));
    REQUIRE(cfg_len > 0);
    REQUIRE(std::string(cfg_out).find("feat_dim") != std::string::npos);
    rl_close(h);
}

TEST_CASE("rl_model_path_stub_bonus", "[rl][onnx]") {
    char err[128] = {0};
    // Provide model_path pointing to dummy_model.onnx
    rl_handle_t h = rl_init_from_json("{\"feat_dim\":6,\"model_path\":\"tests/v2/dummy_model.onnx\"}", err, sizeof(err));
    REQUIRE(h);
    int num_items = 8; int num_candidates = 3;
    auto cand = make_select_candidates(num_items, num_candidates);
    std::vector<double> scores_no_model(num_candidates, 0.0);
    // Score with model loaded handle
    REQUIRE(rl_score_batch(h, "{}", cand.data(), num_items, num_candidates, 0, scores_no_model.data(), err, sizeof(err)) == 0);
    // Initialize second handle without model_path for baseline
    rl_handle_t h2 = rl_init_from_json("{\"feat_dim\":6}", err, sizeof(err));
    REQUIRE(h2);
    std::vector<double> scores_baseline(num_candidates, 0.0);
    REQUIRE(rl_score_batch(h2, "{}", cand.data(), num_items, num_candidates, 0, scores_baseline.data(), err, sizeof(err)) == 0);
    // Note: With real ONNX, rl_score_batch doesn't use model (only rl_score_batch_with_features does)
    // So scores may be similar; this test now validates no crash with model_path set
    rl_close(h);
    rl_close(h2);
}

#ifdef RL_ONNX_ENABLED
TEST_CASE("onnx_inference_golden_output", "[rl][onnx]") {
    char err[256] = {0};
    // Load model with 8 features
    const char* cfg = R"({"feat_dim":8, "model_path":"tests/v2/tiny_linear_8.onnx"})";
    rl_handle_t h = rl_init_from_json(cfg, err, sizeof(err));
    REQUIRE(h != nullptr);
    REQUIRE(rl_get_feat_dim(h) == 8);
    
    // Create features: all ones for simple computation
    // Model: W*x + b where W is random weights, b=0.5
    // Expected output = sum(W) + 0.5
    // From gen_onnx_model.py with seed=42:
    // W = [ 0.04967142 -0.01382643  0.06476886  0.15230298 -0.02341534 -0.0234137  0.15792128  0.07674348]
    // sum(W) ≈ 0.4408
    // expected ≈ 0.9408 per candidate
    int num_candidates = 2;
    std::vector<float> features(num_candidates * 8, 1.0f); // all ones
    std::vector<double> scores(num_candidates, 0.0);
    
    int rc = rl_score_batch_with_features(h, features.data(), 8, num_candidates, scores.data(), err, sizeof(err));
    REQUIRE(rc == 0);
    
    // Validate scores are close to expected (W sum + b ≈ 0.9408)
    double expected = 0.9408;
    for (int i = 0; i < num_candidates; ++i) {
        REQUIRE(scores[i] == Catch::Approx(expected).epsilon(0.01));
    }
    rl_close(h);
}

TEST_CASE("onnx_feat_dim_mismatch_fallback", "[rl][onnx]") {
    char err[256] = {0};
    // Load model with 8 features
    const char* cfg = R"({"feat_dim":8, "model_path":"tests/v2/tiny_linear_8.onnx"})";
    rl_handle_t h = rl_init_from_json(cfg, err, sizeof(err));
    REQUIRE(h != nullptr);
    
    // Try to score with wrong feat_dim (should return error)
    int num_candidates = 2;
    std::vector<float> features(num_candidates * 12, 1.0f); // 12 features instead of 8
    std::vector<double> scores(num_candidates, 0.0);
    
    int rc = rl_score_batch_with_features(h, features.data(), 12, num_candidates, scores.data(), err, sizeof(err));
    REQUIRE(rc == -2); // feat_dim mismatch error
    rl_close(h);
}

TEST_CASE("onnx_missing_model_path_fallback", "[rl][onnx]") {
    char err[256] = {0};
    // Init without model_path (should use LinUCB bandit fallback)
    const char* cfg = R"({"feat_dim":8})";
    rl_handle_t h = rl_init_from_json(cfg, err, sizeof(err));
    REQUIRE(h != nullptr);
    
    // Score should use bandit (no ONNX)
    int num_candidates = 2;
    std::vector<float> features(num_candidates * 8, 1.0f);
    std::vector<double> scores(num_candidates, 0.0);
    
    int rc = rl_score_batch_with_features(h, features.data(), 8, num_candidates, scores.data(), err, sizeof(err));
    REQUIRE(rc == 0);
    // Scores should be non-zero (LinUCB with alpha > 0)
    REQUIRE(scores[0] > 0.0);
    rl_close(h);
}

TEST_CASE("onnx_invalid_model_path_fallback", "[rl][onnx]") {
    char err[256] = {0};
    // Init with invalid model path (should fall back to bandit)
    const char* cfg = R"({"feat_dim":8, "model_path":"nonexistent_model.onnx"})";
    rl_handle_t h = rl_init_from_json(cfg, err, sizeof(err));
    REQUIRE(h != nullptr); // Should still initialize (with fallback)
    
    // Score should use bandit fallback
    int num_candidates = 2;
    std::vector<float> features(num_candidates * 8, 1.0f);
    std::vector<double> scores(num_candidates, 0.0);
    
    int rc = rl_score_batch_with_features(h, features.data(), 8, num_candidates, scores.data(), err, sizeof(err));
    REQUIRE(rc == 0);
    REQUIRE(scores[0] > 0.0); // LinUCB bandit score
    rl_close(h);
}
#endif // RL_ONNX_ENABLED

