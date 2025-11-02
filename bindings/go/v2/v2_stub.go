//go:build !darwin || !arm64

package v2

import "errors"

type Solution struct{}

func SolveJSON(_ string, _ string) (*Solution, error) {
	return nil, errors.New("v2 solver is only available on darwin/arm64 in this binding")
}
