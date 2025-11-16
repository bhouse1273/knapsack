// Package rl provides Go bindings to the rl_support library.
// Build instructions:
// 1) Ensure CMake builds shared library: rl_support_shared (librl_support.dylib / .so)
// 2) go build (CGO_ENABLED=1)
package rl

/*
#cgo CFLAGS: -I../../../rl
#cgo darwin LDFLAGS: -L../../../build -lrl_support
#cgo linux LDFLAGS: -L../../../build -lrl_support
#include "rl_api.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

type Handle struct{ h unsafe.Pointer }

// InitFromJSON initializes an RL context from a JSON configuration string.
func InitFromJSON(cfg string) (*Handle, error) {
	errBuf := make([]byte, 256)
	cHandle := C.rl_init_from_json(C.CString(cfg), (*C.char)(unsafe.Pointer(&errBuf[0])), C.int(len(errBuf)))
	if cHandle == nil {
		return nil, errors.New(string(errBuf))
	}
	return &Handle{h: cHandle}, nil
}

// PrepareFeatures extracts features for select-mode candidates.
// candidates is a flat byte slice: numCandidates * numItems.
func (h *Handle) PrepareFeatures(candidates []byte, numItems, numCandidates, mode int) ([]float32, error) {
	if h == nil || h.h == nil {
		return nil, errors.New("nil handle")
	}
	featDim := 8 // until we expose a getter
	out := make([]float32, numCandidates*featDim)
	errBuf := make([]byte, 128)
	rc := C.rl_prepare_features(h.h,
		(*C.uchar)(unsafe.Pointer(&candidates[0])),
		C.int(numItems), C.int(numCandidates), C.int(mode),
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.char)(unsafe.Pointer(&errBuf[0])), C.int(len(errBuf)))
	if rc != 0 {
		return nil, errors.New(string(errBuf))
	}
	return out, nil
}

// ScoreWithFeatures scores candidates given pre-computed features.
func (h *Handle) ScoreWithFeatures(features []float32, featDim, numCandidates int) ([]float64, error) {
	if h == nil || h.h == nil {
		return nil, errors.New("nil handle")
	}
	out := make([]float64, numCandidates)
	errBuf := make([]byte, 128)
	rc := C.rl_score_batch_with_features(h.h,
		(*C.float)(unsafe.Pointer(&features[0])),
		C.int(featDim), C.int(numCandidates),
		(*C.double)(unsafe.Pointer(&out[0])),
		(*C.char)(unsafe.Pointer(&errBuf[0])), C.int(len(errBuf)))
	if rc != 0 {
		return nil, errors.New(string(errBuf))
	}
	return out, nil
}

// Learn updates the model from feedback JSON containing rewards array.
func (h *Handle) Learn(feedbackJSON string) error {
	if h == nil || h.h == nil {
		return errors.New("nil handle")
	}
	errBuf := make([]byte, 128)
	rc := C.rl_learn_batch(h.h, C.CString(feedbackJSON), (*C.char)(unsafe.Pointer(&errBuf[0])), C.int(len(errBuf)))
	if rc != 0 {
		return errors.New(string(errBuf))
	}
	return nil
}

// Close releases resources.
func (h *Handle) Close() {
	if h != nil && h.h != nil {
		C.rl_close(h.h)
		h.h = nil
	}
}
