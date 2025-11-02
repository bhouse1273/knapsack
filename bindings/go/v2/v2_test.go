//go:build darwin && arm64

package v2

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSolveJSON_Smoke(t *testing.T) {
	// Load the example config
	repoRoot, err := filepath.Abs("../../../")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	cfgPath := filepath.Join(repoRoot, "docs", "v2", "example_select.json")
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		t.Fatalf("read config: %v", err)
	}

	opts := `{"beam_width":16,"iters":3,"seed":7,"debug":false}`
	sol, err := SolveJSON(string(data), opts)
	if err != nil {
		t.Fatalf("SolveJSON error: %v", err)
	}
	if sol == nil {
		t.Fatalf("nil solution")
	}
	if sol.NumItems == 0 || len(sol.Select) != sol.NumItems {
		t.Fatalf("unexpected sizes: num=%d, len(sel)=%d", sol.NumItems, len(sol.Select))
	}
	if sol.Objective-sol.Penalty != sol.Total {
		t.Fatalf("inconsistent totals: obj=%f pen=%f total=%f", sol.Objective, sol.Penalty, sol.Total)
	}
	// On the example config we expect a feasible solution with zero penalty and 2 items selected.
	if sol.Penalty != 0 {
		t.Fatalf("expected zero penalty on example, got %f", sol.Penalty)
	}
	selected := 0
	for _, v := range sol.Select {
		if v != 0 {
			selected++
		}
	}
	if selected != 2 {
		t.Fatalf("expected 2 selected on example, got %d", selected)
	}
}
