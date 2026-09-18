//go:build arm64

#include "textflag.h"

TEXT ·dotVector(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V2.B16, V2.B16, V2.B16
	LSR $1, R2, R3
	CBZ R3, dot_tail
dot_loop:
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R1), [V1.D2]
	WORD $0x6e61dc00 // FMUL V0.2D, V0.2D, V1.2D
	WORD $0x4e60d442 // FADD V2.2D, V2.2D, V0.2D
	SUB $1, R3
	CBNZ R3, dot_loop
dot_tail:
	WORD $0x6e62d442 // FADDP V2.2D, V2.2D, V2.2D
	VST1 [V2.D2], (RSP)
	FMOVD 0(RSP), F2
	TBZ $0, R2, dot_done
	FMOVD (R0), F0
	FMOVD (R1), F1
	FMULD F1, F0, F0
	FADDD F0, F2, F2
dot_done:
	FMOVD F2, ret+24(FP)
	RET

TEXT ·squaredEuclideanVector(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V2.B16, V2.B16, V2.B16
	LSR $1, R2, R3
	CBZ R3, dist_tail
dist_loop:
	VLD1.P 16(R0), [V0.D2]
	VLD1.P 16(R1), [V1.D2]
	WORD $0x4ee1d400 // FSUB V0.2D, V0.2D, V1.2D
	WORD $0x6e60dc00 // FMUL V0.2D, V0.2D, V0.2D
	WORD $0x4e60d442 // FADD V2.2D, V2.2D, V0.2D
	SUB $1, R3
	CBNZ R3, dist_loop
dist_tail:
	WORD $0x6e62d442 // FADDP V2.2D, V2.2D, V2.2D
	VST1 [V2.D2], (RSP)
	FMOVD 0(RSP), F2
	TBZ $0, R2, dist_done
	FMOVD (R0), F0
	FMOVD (R1), F1
	FSUBD F1, F0, F0
	FMULD F0, F0, F0
	FADDD F0, F2, F2
dist_done:
	FMOVD F2, ret+24(FP)
	RET
