//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestEvaluate_Smoke(t *testing.T) {
	numItems := 8
	numCands := 4
	bytesPerCand := (numItems + 3) / 4 // 2 bits per item

	// Build candidates: assign every item to van 0 (lane=1) => byte pattern 0b01010101 = 0x55
	cand := make([]byte, bytesPerCand*numCands)
	for c := 0; c < numCands; c++ {
		for b := 0; b < bytesPerCand; b++ {
			cand[c*bytesPerCand+b] = 0x55
		}
	}
	obj := make([]float32, numCands)
	pen := make([]float32, numCands)

	// Simple attributes: value=1 per item, weight=0 so no penalties. One van with large cap.
	values := make([]float32, numItems)
	weights := make([]float32, numItems)
	for i := range values {
		values[i] = 1
	}
	caps := []float32{1000}

	in := EvalIn{
		Candidates:    &cand[0],
		NumItems:      int32(numItems),
		NumCandidates: int32(numCands),
		ItemValues:    &values[0],
		ItemWeights:   &weights[0],
		VanCaps:       &caps[0],
		NumVans:       1,
		PenaltyCoeff:  1.0,
	}
	out := EvalOut{
		Obj:         &obj[0],
		SoftPenalty: &pen[0],
	}

	if err := Evaluate(in, out); err != nil {
		t.Fatalf("Evaluate returned error: %v", err)
	}

	for i := 0; i < numCands; i++ {
		if obj[i] != float32(numItems) {
			t.Fatalf("obj[%d] = %f, want %d", i, obj[i], numItems)
		}
		if pen[i] != 0 {
			t.Fatalf("pen[%d] = %f, want 0", i, pen[i])
		}
	}
}
