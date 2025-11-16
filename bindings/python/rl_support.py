"""Minimal ctypes binding for rl_support shared library.
Usage:
    from rl_support import RL
    rl = RL(json_cfg='{"alpha":0.3,"feat_dim":8}')
    scores = rl.score_select(candidates_bytes, num_items, num_candidates)
    rl.learn([1.0,0.0,0.5])

candidates_bytes: bytes object of length num_items * num_candidates (row-major)
"""
import ctypes, os, sys, json

# Attempt to locate shared library in build directory relative to repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BUILD = os.path.join(ROOT, 'build')
LIB_NAMES = [
    'librl_support.dylib',  # macOS
    'rl_support.dll',       # Windows (future)
    'librl_support.so',     # Linux
]
_lib_path = None
for name in LIB_NAMES:
    cand = os.path.join(BUILD, name)
    if os.path.exists(cand):
        _lib_path = cand
        break
if _lib_path is None:
    raise RuntimeError(f"rl_support shared library not found in {BUILD}; build target rl_support_shared first.")

_lib = ctypes.CDLL(_lib_path)

# Function signatures
_lib.rl_init_from_json.restype = ctypes.c_void_p
_lib.rl_init_from_json.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
_lib.rl_prepare_features.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_lib.rl_score_batch_with_features.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
_lib.rl_learn_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
_lib.rl_close.argtypes = [ctypes.c_void_p]

class RL:
    def __init__(self, json_cfg='{}'):
        err = ctypes.create_string_buffer(256)
        self._h = _lib.rl_init_from_json(json_cfg.encode('utf-8'), err, len(err))
        if not self._h:
            raise RuntimeError(f"rl_init_from_json failed: {err.value.decode('utf-8')}")
        self.feat_dim = 8  # until we expose a getter

    def prepare_features(self, candidates_bytes: bytes, num_items: int, num_candidates: int, mode: int = 0):
        if len(candidates_bytes) != num_items * num_candidates:
            raise ValueError("candidates_bytes length mismatch")
        out = (ctypes.c_float * (num_candidates * self.feat_dim))()
        err = ctypes.create_string_buffer(128)
        rc = _lib.rl_prepare_features(self._h, ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(candidates_bytes))), num_items, num_candidates, mode, out, err, len(err))
        if rc != 0:
            raise RuntimeError(f"rl_prepare_features failed: {err.value.decode('utf-8')}")
        return [out[i] for i in range(num_candidates * self.feat_dim)]

    def score_with_features(self, features, num_candidates: int):
        if len(features) != num_candidates * self.feat_dim:
            raise ValueError("features length mismatch")
        arr = (ctypes.c_float * len(features))(*features)
        out = (ctypes.c_double * num_candidates)()
        err = ctypes.create_string_buffer(128)
        rc = _lib.rl_score_batch_with_features(self._h, arr, self.feat_dim, num_candidates, out, err, len(err))
        if rc != 0:
            raise RuntimeError(f"rl_score_batch_with_features failed: {err.value.decode('utf-8')}")
        return [out[i] for i in range(num_candidates)]

    def score_select(self, candidates_bytes: bytes, num_items: int, num_candidates: int):
        feats = self.prepare_features(candidates_bytes, num_items, num_candidates, 0)
        return self.score_with_features(feats, num_candidates)

    def learn(self, rewards):
        feedback = json.dumps({"rewards": rewards})
        err = ctypes.create_string_buffer(128)
        rc = _lib.rl_learn_batch(self._h, feedback.encode('utf-8'), err, len(err))
        if rc != 0:
            raise RuntimeError(f"rl_learn_batch failed: {err.value.decode('utf-8')}")

    def close(self):
        if self._h:
            _lib.rl_close(self._h)
            self._h = None

    def __del__(self):
        self.close()

__all__ = ["RL"]
