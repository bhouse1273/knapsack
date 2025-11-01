//go:build !darwin || !arm64

package metal

import "errors"

type EvalIn struct{}
type EvalOut struct{}

func Evaluate(_ EvalIn, _ EvalOut) error {
	return errors.New("Metal backend is only available on darwin/arm64")
}
