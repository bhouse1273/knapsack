//go:build darwin && arm64

package v2

/*
#cgo CFLAGS: -I${SRCDIR}/../../../knapsack-library/include
#cgo LDFLAGS: -L${SRCDIR}/../../../build-lib -lknapsack -framework Metal -framework Foundation -lc++
#include <stdlib.h>
#include <string.h>
#include "knapsack_c.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Solution matches KnapsackSolutionV2.
type Solution struct {
	NumItems  int
	Select    []int
	Objective float64
	Penalty   float64
	Total     float64
}

// SolveJSON runs the V2 solver using a JSON config string and an optional options JSON string.
// options JSON supports keys: beam_width, iters, seed, debug.
func SolveJSON(configJSON string, optsJSON string) (*Solution, error) {
	if configJSON == "" {
		return nil, errors.New("config JSON is empty")
	}
	cCfg := C.CString(configJSON)
	defer C.free(unsafe.Pointer(cCfg))
	var cOpts *C.char
	if optsJSON != "" {
		cOpts = C.CString(optsJSON)
		defer C.free(unsafe.Pointer(cOpts))
	}
	var out *C.KnapsackSolutionV2
	rc := C.solve_knapsack_v2_from_json(cCfg, cOpts, &out)
	if rc != 0 {
		return nil, errors.New("solve_knapsack_v2_from_json failed")
	}
	defer C.free_knapsack_solution_v2(out)

	n := int(out.num_items)
	sel := make([]int, n)
	if n > 0 && out._select != nil {
		// Create a Go slice view of the C array and copy
		slice := (*[1 << 30]C.int)(unsafe.Pointer(out._select))[:n:n]
		for i := 0; i < n; i++ {
			sel[i] = int(slice[i])
		}
	}
	return &Solution{
		NumItems:  n,
		Select:    sel,
		Objective: float64(out.objective),
		Penalty:   float64(out.penalty),
		Total:     float64(out.total),
	}, nil
}
