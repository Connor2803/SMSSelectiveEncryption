package lattigo

import (
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/ring"
)

// Name of the different self-defined parameter sets
var (
	PN10QP27CI = ckks.ParametersLiteral{
		LogN:         10,
		LogQ:         []int{13},
		LogP:         []int{14},
		RingType:     ring.ConjugateInvariant,
		DefaultScale: 13,
	}

	PN11QP54CI = ckks.ParametersLiteral{
		LogN:         11,
		LogQ:         []int{18, 18}, // 18 + 18
		LogP:         []int{18},     // 18
		RingType:     ring.ConjugateInvariant,
		DefaultScale: 18,
	}
)
