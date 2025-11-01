//go:build darwin && arm64

package metal

/*
#cgo LDFLAGS: -L${SRCDIR}/../../../kernels/metal/build -lknapsack_metal -lc++ -framework Metal -framework Foundation
#include <stdlib.h>
#include <string.h>
#include "../../../kernels/metal/metal_api.h"
*/
import "C"

import (
	_ "embed"
	"errors"
	"unsafe"
)

// Build the static library (no metallib required); runtime compiles shader source.
//go:generate make -C ../../../kernels/metal lib
// Copy shader locally for embedding (go:embed doesn't allow .. in patterns).
//go:generate cp -f ../../../kernels/metal/shaders/eval_block_candidates.metal .

// Embed the Metal Shading Language source so we can compile at runtime.
// This avoids depending on the external 'metal' CLI.
//
//go:embed eval_block_candidates.metal
var shaderMSL []byte

// EvalIn mirrors the C struct MetalEvalIn.
type EvalIn struct {
	Candidates    *byte
	NumItems      int32
	NumCandidates int32
	ItemValues    *float32
	ItemWeights   *float32
	VanCaps       *float32
	NumVans       int32
	PenaltyCoeff  float32
}

// EvalOut mirrors the C struct MetalEvalOut.
type EvalOut struct {
	Obj         *float32
	SoftPenalty *float32
}

func init() {
	if len(shaderMSL) == 0 {
		panic("Metal: embedded shader source missing; ensure path ../../../kernels/metal/shaders/eval_block_candidates.metal exists")
	}
	rc := C.knapsack_metal_init_from_source((*C.char)(unsafe.Pointer(&shaderMSL[0])), C.size_t(len(shaderMSL)), nil, 0)
	if rc != 0 {
		panic("Metal: init-from-source failed")
	}
}

// Evaluate runs the Metal evaluator on the provided inputs and writes results into out.
func Evaluate(in EvalIn, out EvalOut) error {
	// Allocate C memory for candidate lanes to satisfy cgo pointer rules.
	// bytes_per_candidate = ceil(num_items/4). 2 bits per item => 4 items per byte.
	if in.NumItems < 0 || in.NumCandidates < 0 {
		return errors.New("invalid sizes")
	}
	bytesPerCand := (int(in.NumItems) + 3) / 4
	totalCand := bytesPerCand * int(in.NumCandidates)
	var candCPtr unsafe.Pointer
	if totalCand > 0 {
		candCPtr = C.malloc(C.size_t(totalCand))
		if candCPtr == nil {
			return errors.New("malloc failed for candidates")
		}
		defer C.free(candCPtr)
		if in.Candidates != nil {
			C.memcpy(candCPtr, unsafe.Pointer(in.Candidates), C.size_t(totalCand))
		} else {
			// zero-initialize if nil provided
			C.memset(candCPtr, 0, C.size_t(totalCand))
		}
	}

	// Copy attribute arrays to C memory if provided
	var valsCPtr, wgtsCPtr, capsCPtr unsafe.Pointer
	if in.NumItems > 0 && in.ItemValues != nil {
		valsCPtr = C.malloc(C.size_t(int(in.NumItems) * 4))
		if valsCPtr == nil {
			return errors.New("malloc failed for item values")
		}
		defer C.free(valsCPtr)
		C.memcpy(valsCPtr, unsafe.Pointer(in.ItemValues), C.size_t(int(in.NumItems)*4))
	}
	if in.NumItems > 0 && in.ItemWeights != nil {
		wgtsCPtr = C.malloc(C.size_t(int(in.NumItems) * 4))
		if wgtsCPtr == nil {
			return errors.New("malloc failed for item weights")
		}
		defer C.free(wgtsCPtr)
		C.memcpy(wgtsCPtr, unsafe.Pointer(in.ItemWeights), C.size_t(int(in.NumItems)*4))
	}
	if in.NumVans > 0 && in.VanCaps != nil {
		capsCPtr = C.malloc(C.size_t(int(in.NumVans) * 4))
		if capsCPtr == nil {
			return errors.New("malloc failed for van caps")
		}
		defer C.free(capsCPtr)
		C.memcpy(capsCPtr, unsafe.Pointer(in.VanCaps), C.size_t(int(in.NumVans)*4))
	}

	cin := C.MetalEvalIn{
		candidates:     (*C.uchar)(candCPtr),
		num_items:      C.int(in.NumItems),
		num_candidates: C.int(in.NumCandidates),
		item_values:    (*C.float)(valsCPtr),
		item_weights:   (*C.float)(wgtsCPtr),
		van_capacities: (*C.float)(capsCPtr),
		num_vans:       C.int(in.NumVans),
		penalty_coeff:  C.float(in.PenaltyCoeff),
	}
	// Allocate C buffers for outputs and copy back after the call to avoid passing Go pointers to C.
	n := int(in.NumCandidates)
	var objCPtr, penCPtr unsafe.Pointer
	if n > 0 {
		objCPtr = C.malloc(C.size_t(n) * C.size_t(4))
		penCPtr = C.malloc(C.size_t(n) * C.size_t(4))
		if objCPtr == nil || penCPtr == nil {
			if objCPtr != nil {
				C.free(objCPtr)
			}
			if penCPtr != nil {
				C.free(penCPtr)
			}
			return errors.New("malloc failed for outputs")
		}
		defer C.free(objCPtr)
		defer C.free(penCPtr)
	}

	cout := C.MetalEvalOut{
		obj:          (*C.float)(objCPtr),
		soft_penalty: (*C.float)(penCPtr),
	}
	if rc := C.knapsack_metal_eval(&cin, &cout, nil, 0); rc != 0 {
		return errors.New("Metal evaluate failed")
	}

	// Copy results back into provided Go buffers.
	if n > 0 {
		outObj := unsafe.Slice(out.Obj, n)
		outPen := unsafe.Slice(out.SoftPenalty, n)
		cObj := unsafe.Slice((*float32)(objCPtr), n)
		cPen := unsafe.Slice((*float32)(penCPtr), n)
		copy(outObj, cObj)
		copy(outPen, cPen)
	}
	return nil
}
