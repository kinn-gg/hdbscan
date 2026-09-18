//go:build amd64 && amd64.v3

#include "textflag.h"

TEXT ·dotVector(SB), NOSPLIT, $0-32
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPD Y0, Y0, Y0
	MOVQ CX, DX
	SHRQ $2, DX
	JZ dot_tail
dot_loop:
	VMOVUPD (AX), Y1
	VMULPD (BX), Y1, Y1
	VADDPD Y1, Y0, Y0
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ DX
	JNZ dot_loop
dot_tail:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	VMOVAPD X0, X2
	ANDQ $3, CX
	JZ dot_done
dot_scalar:
	VMOVSD (AX), X1
	VMULSD (BX), X1, X1
	VADDSD X1, X2, X2
	ADDQ $8, AX
	ADDQ $8, BX
	DECQ CX
	JNZ dot_scalar
dot_done:
	VMOVSD X2, ret+24(FP)
	VZEROUPPER
	RET

TEXT ·squaredEuclideanVector(SB), NOSPLIT, $0-32
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPD Y0, Y0, Y0
	MOVQ CX, DX
	SHRQ $2, DX
	JZ dist_tail
dist_loop:
	VMOVUPD (AX), Y1
	VSUBPD (BX), Y1, Y1
	VMULPD Y1, Y1, Y1
	VADDPD Y1, Y0, Y0
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ DX
	JNZ dist_loop
dist_tail:
	VEXTRACTF128 $1, Y0, X1
	VADDPD X1, X0, X0
	VHADDPD X0, X0, X0
	VMOVAPD X0, X2
	ANDQ $3, CX
	JZ dist_done
dist_scalar:
	VMOVSD (AX), X1
	VSUBSD (BX), X1, X1
	VMULSD X1, X1, X1
	VADDSD X1, X2, X2
	ADDQ $8, AX
	ADDQ $8, BX
	DECQ CX
	JNZ dist_scalar
dist_done:
	VMOVSD X2, ret+24(FP)
	VZEROUPPER
	RET
