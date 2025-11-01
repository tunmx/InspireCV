.macro asm_function fname
.globl _\fname
_\fname:
.endm

.text
.align 5
// _TaskSamplerC4BilinearOpt
// void _TaskSamplerC4BilinearOpt(const unsigned char* source, unsigned char* dest,
//                                 float* points, size_t count, size_t iw, size_t ih, size_t yStride)
// x0: source, x1: dest, x2: points, x3: count, w4: xMax, w5: yMax, w6: yStride
asm_function _TaskSamplerC4BilinearOpt
movi v19.4s, #0
ld1 {v0.2s, v1.2s}, [x2]
sampler_c4_bi_L1:
cmp x3, #0
beq sampler_c4_bi_End
mov v16.s[0], w4
mov v16.s[1], w5
mov w12, #4
mov v7.s[0], w12
mov v7.s[1], w6
dup v20.2d, x0
sampler_c4_bi_L1Loop:
fcvtms v2.2s, v0.2s
frintm v4.2s, v0.2s
smax v2.2s, v2.2s, v19.2s
fcvtps v3.2s, v0.2s
fabd v4.2s, v0.2s, v4.2s
smax v3.2s, v3.2s, v19.2s
smin v2.2s, v2.2s, v16.2s
smin v3.2s, v3.2s, v16.2s
mul v2.2s, v2.2s, v7.2s
mul v3.2s, v3.2s, v7.2s
mov v2.s[2], v3.s[0]
mov v3.s[2], v2.s[0]
mov v2.s[3], v2.s[1]
mov v3.s[3], v3.s[1]
uaddlp v2.2d, v2.4s
uaddlp v3.2d, v3.4s
add v2.2d, v20.2d, v2.2d
add v3.2d, v20.2d, v3.2d
mov x4, v2.d[0]
mov x5, v2.d[1]
ld1 {v5.s}[0], [x4]
ld1 {v5.s}[1], [x5]
mov x4, v3.d[0]
uxtl v5.8h, v5.8b
mov x5, v3.d[1]
ld1 {v6.s}[0], [x4]
ld1 {v6.s}[1], [x5]
uxtl v6.8h, v6.8b
uxtl v2.4s, v5.4h
uxtl2 v3.4s, v5.8h
ucvtf v2.4s, v2.4s
uxtl v5.4s, v6.4h
ucvtf v3.4s, v3.4s
uxtl2 v6.4s, v6.8h
ucvtf v5.4s, v5.4s
ucvtf v6.4s, v6.4s
fsub v3.4s, v3.4s, v2.4s
fsub v6.4s, v6.4s, v5.4s
fmla v2.4s, v3.4s, v4.s[0]
fmla v5.4s, v6.4s, v4.s[0]
fsub v5.4s, v5.4s, v2.4s
fmla v2.4s, v5.4s, v4.s[1]
fcvtns v2.4s, v2.4s
uqxtn v2.4h, v2.4s
uqxtn v2.8b, v2.8h
fadd v0.2s, v0.2s, v1.2s
subs x3, x3, #1
st1 {v2.s}[0], [x1], #4
bne sampler_c4_bi_L1Loop
sampler_c4_bi_End:
ret
.text
.align 5
// _TaskSamplerC4NearestOpt
// void _TaskSamplerC4NearestOpt(const unsigned char* source, unsigned char* dest,
//                                float* points, size_t count, size_t iw, size_t ih, size_t yStride)
// x0: source, x1: dest, x2: points, x3: count, w4: xMax, w5: yMax, w6: yStride
asm_function _TaskSamplerC4NearestOpt
movi v19.4s, #0
ld1 {v0.2s, v1.2s}, [x2]
sampler_c4_near_L4:
cmp x3, #4
blt sampler_c4_near_L1
dup v16.4s, w4
dup v17.4s, w5
movi v3.2s, #4
scvtf v3.2s, v3.2s
fmul v3.2s, v3.2s, v1.2s
dup v25.4s, v3.s[0]
dup v26.4s, v3.s[1]
fadd v2.2s, v0.2s, v1.2s
mov v4.s[0], v0.s[0]
fadd v3.2s, v2.2s, v1.2s
mov v5.s[0], v0.s[1]
mov v4.s[1], v2.s[0]
mov v5.s[1], v2.s[1]
mov v4.s[2], v3.s[0]
fadd v2.2s, v3.2s, v1.2s
mov v5.s[2], v3.s[1]
mov v4.s[3], v2.s[0]
mov v5.s[3], v2.s[1]
dup v23.4s, w6
movi v24.4s, #4
dup v22.2d, x0
sampler_c4_near_L4Loop:
fcvtns v6.4s, v4.4s
fcvtns v7.4s, v5.4s
smin v6.4s, v6.4s, v16.4s
smin v7.4s, v7.4s, v17.4s
smax v6.4s, v6.4s, v19.4s
smax v7.4s, v7.4s, v19.4s
mul v7.4s, v7.4s, v23.4s
mla v7.4s, v6.4s, v24.4s
uxtl v6.2d, v7.2s
uxtl2 v7.2d, v7.4s
add v6.2d, v6.2d, v22.2d
add v7.2d, v7.2d, v22.2d
mov x12, v6.d[0]
mov x13, v6.d[1]
ld1 {v3.s}[0], [x12]
mov x12, v7.d[0]
ld1 {v3.s}[1], [x13]
fadd v5.4s, v26.4s, v5.4s
mov x13, v7.d[1]
ld1 {v3.s}[2], [x12]
fadd v4.4s, v25.4s, v4.4s
ld1 {v3.s}[3], [x13]
st1 {v3.4s}, [x1], #16
sub x3, x3, #4
cmp x3, #4
bge sampler_c4_near_L4Loop
mov v0.s[0], v4.s[0]
mov v0.s[1], v5.s[0]
sampler_c4_near_L1:
cmp x3, #0
beq sampler_c4_near_End
mov v6.s[0], w4
mov v6.s[1], w5
mov w12, #4
sampler_c4_near_L1Loop:
fcvtns v2.2s, v0.2s
smax v2.2s, v2.2s, v19.2s
smin v2.2s, v2.2s, v6.2s
mov w4, v2.s[0]
mov w5, v2.s[1]
umull x5, w5, w6
umull x4, w4, w12
add x7, x0, x5
add x4, x4, x7
ld1 {v2.s}[0], [x4]
fadd v0.2s, v0.2s, v1.2s
subs x3, x3, #1
st1 {v2.s}[0], [x1], #4
bne sampler_c4_near_L1Loop
sampler_c4_near_End:
ret
.text
.align 5
// _TaskSamplerC1BilinearOpt
// void _TaskSamplerC1BilinearOpt(const unsigned char* source, unsigned char* dest,
//                                float* points, size_t count, size_t iw, size_t ih, size_t yStride)
// x0: source, x1: dest, x2: points, x3: count, w4: xMax, w5: yMax, w6: yStride
asm_function _TaskSamplerC1BilinearOpt
movi v23.4s, #0
ld1 {v30.2s, v31.2s}, [x2]
sampler_c1_bi_L8:
cmp x3, #8
blt sampler_c1_bi_L1
dup v25.2d, x0
dup v24.4s, w6
dup v22.4s, w5
dup v21.4s, w4
ucvtf v22.4s, v22.4s
ucvtf v21.4s, v21.4s
movi v3.2s, #4
scvtf v3.2s, v3.2s
fmul v3.2s, v3.2s, v31.2s
dup v26.4s, v3.s[0]
dup v27.4s, v3.s[1]
fadd v2.2s, v30.2s, v31.2s
mov v28.s[0], v30.s[0]
fadd v3.2s, v2.2s, v31.2s
mov v29.s[0], v30.s[1]
mov v28.s[1], v2.s[0]
mov v29.s[1], v2.s[1]
mov v28.s[2], v3.s[0]
fadd v2.2s, v3.2s, v31.2s
mov v29.s[2], v3.s[1]
mov v28.s[3], v2.s[0]
mov v29.s[3], v2.s[1]
sampler_c1_bi_LoopL8:
    sub x3, x3, #8
    mov v4.16b, v28.16b
    mov v5.16b, v29.16b
    fmin v4.4s, v4.4s, v21.4s
    fadd v6.4s, v28.4s, v26.4s
    fadd v7.4s, v29.4s, v27.4s
    fmax v4.4s, v4.4s, v23.4s
    fmin v5.4s, v5.4s, v22.4s
    fadd v28.4s, v6.4s, v26.4s
    fadd v29.4s, v7.4s, v27.4s
    fmin v6.4s, v6.4s, v21.4s
    fmin v7.4s, v7.4s, v22.4s
    fmax v5.4s, v5.4s, v23.4s
    fmax v6.4s, v6.4s, v23.4s
    fmax v7.4s, v7.4s, v23.4s
.macro COMPUTE_FOUR
    fcvtms v16.4s, v4.4s
    fcvtps v17.4s, v4.4s
    frintm v30.4s, v4.4s
    fcvtms v18.4s, v5.4s
    fcvtps v19.4s, v5.4s
    frintm v31.4s, v5.4s
    fabd v0.4s, v30.4s, v4.4s
    fabd v1.4s, v5.4s, v31.4s
    mul v30.4s, v18.4s, v24.4s
    mul v31.4s, v19.4s, v24.4s
    add v18.4s, v31.4s, v16.4s
    add v19.4s, v31.4s, v17.4s
    add v16.4s, v16.4s, v30.4s
    add v17.4s, v17.4s, v30.4s
    uxtl v31.2d, v16.2s
    uxtl2 v30.2d, v16.4s
    uqadd v31.2d, v25.2d, v31.2d
    uqadd v30.2d, v25.2d, v30.2d
    mov x7, v31.d[0]
    ld1 {v2.b}[0], [x7]
    mov x7, v31.d[1]
    ld1 {v2.b}[1], [x7]
    mov x7, v30.d[0]
    ld1 {v2.b}[2], [x7]
    mov x7, v30.d[1]
    ld1 {v2.b}[3], [x7]
    uxtl v31.2d, v17.2s
    uxtl2 v30.2d, v17.4s
    uqadd v31.2d, v25.2d, v31.2d
    uqadd v30.2d, v25.2d, v30.2d
    mov x7, v31.d[0]
    ld1 {v2.b}[4], [x7]
    mov x7, v31.d[1]
    ld1 {v2.b}[5], [x7]
    mov x7, v30.d[0]
    ld1 {v2.b}[6], [x7]
    mov x7, v30.d[1]
    ld1 {v2.b}[7], [x7]
    uxtl v31.2d, v18.2s
    uxtl2 v30.2d, v18.4s
    uqadd v31.2d, v25.2d, v31.2d
    uqadd v30.2d, v25.2d, v30.2d
    mov x7, v31.d[0]
    ld1 {v3.b}[0], [x7]
    mov x7, v31.d[1]
    ld1 {v3.b}[1], [x7]
    mov x7, v30.d[0]
    ld1 {v3.b}[2], [x7]
    mov x7, v30.d[1]
    ld1 {v3.b}[3], [x7]
    uxtl v31.2d, v19.2s
    uxtl2 v30.2d, v19.4s
    uqadd v31.2d, v25.2d, v31.2d
    uqadd v30.2d, v25.2d, v30.2d
    mov x7, v31.d[0]
    ld1 {v3.b}[4], [x7]
    mov x7, v31.d[1]
    ld1 {v3.b}[5], [x7]
    mov x7, v30.d[0]
    ld1 {v3.b}[6], [x7]
    mov x7, v30.d[1]
    ld1 {v3.b}[7], [x7]
    uxtl v2.8h, v2.8b
    uxtl v3.8h, v3.8b
    uxtl v16.4s, v2.4h
    uxtl2 v17.4s, v2.8h
    uxtl v18.4s, v3.4h
    uxtl2 v19.4s, v3.8h
    ucvtf v16.4s, v16.4s
    ucvtf v17.4s, v17.4s
    ucvtf v18.4s, v18.4s
    ucvtf v19.4s, v19.4s
    fsub v17.4s, v17.4s, v16.4s
    fsub v19.4s, v19.4s, v18.4s
    fmla v16.4s, v17.4s, v0.4s
    fmla v18.4s, v19.4s, v0.4s
    fsub v18.4s, v18.4s, v16.4s
    fmla v16.4s, v18.4s, v1.4s
    fcvtzs v0.4s, v16.4s
    uqxtn v0.4h, v0.4s
    uqxtn v0.8b, v0.8h
    st1 {v0.s}[0], [x1], #4
.endm
    COMPUTE_FOUR
    mov v4.16b, v6.16b
    mov v5.16b, v7.16b
    COMPUTE_FOUR
    cmp x3, #8
    bge sampler_c1_bi_LoopL8
mov v30.s[0], v28.s[0]
mov v30.s[1], v29.s[0]
sampler_c1_bi_L1:
cmp x3, #0
beq sampler_c1_bi_End
mov v21.s[0], w4
mov v21.s[1], w5
ucvtf v22.2s, v21.2s
sampler_c1_bi_LoopL1:
    mov v0.8b, v30.8b
    fmin v0.2s, v0.2s, v22.2s
    fmax v0.2s, v0.2s, v23.2s
    fadd v30.2s, v30.2s, v31.2s
    fcvtms v1.2s, v0.2s
    frintm v3.2s, v0.2s
    fcvtps v2.2s, v0.2s
    mov w8, v1.s[1]
    mov w7, v1.s[0]
    mov w9, v2.s[0]
    fabd v5.2s, v0.2s, v3.2s
    smull x8, w8, w6
    add x8, x8, x0
    sxtw x7, w7
    mov v6.s[0], v5.s[1]
    sxtw x9, w9
    add x10, x8, x7
    ld1 {v4.b}[0], [x10]
    add x10, x8, x9
    ld1 {v4.b}[1], [x10]
    mov w8, v2.s[1]
    umull x8, w8, w6
    add x8, x8, x0
    add x10, x8, x7
    ld1 {v4.b}[2], [x10]
    add x10, x8, x9
    ld1 {v4.b}[3], [x10]
    uxtl v4.8h, v4.8b
    uxtl v0.4s, v4.4h
    ucvtf v0.4s, v0.4s
    mov s1, v0.s[1]
    mov s2, v0.s[2]
    mov s3, v0.s[3]
    fsub s1, s1, s0
    fsub s3, s3, s2
    fmul s1, s1, s5
    fmul s3, s3, s5
    fadd s0, s0, s1
    fadd s2, s2, s3
    fsub s2, s2, s0
    fmul s2, s2, s6
    fadd s0, s0, s2
    fcvtzs s0, s0
    uqxtn v4.4h, v0.4s
    uqxtn v4.8b, v4.8h
    st1 {v4.b}[0], [x1], #1
    subs x3, x3, #1
    bne sampler_c1_bi_LoopL1
sampler_c1_bi_End:
ret
.text
.align 5
// _TaskSamplerC1NearestOpt
// void _TaskSamplerC1NearestOpt(const unsigned char* source, unsigned char* dest,
//                               float* points, size_t count, size_t iw, size_t ih, size_t yStride)
// x0: source, x1: dest, x2: points, x3: count, w4: xMax, w5: yMax, w6: yStride
asm_function _TaskSamplerC1NearestOpt
movi v19.4s, #0
ld1 {v0.2s, v1.2s}, [x2]
sampler_c1_near_L4:
cmp x3, #4
blt sampler_c1_near_L1
dup v16.4s, w4
dup v17.4s, w5
movi v3.2s, #4
scvtf v3.2s, v3.2s
fmul v3.2s, v3.2s, v1.2s
dup v25.4s, v3.s[0]
dup v26.4s, v3.s[1]
fadd v2.2s, v0.2s, v1.2s
mov v4.s[0], v0.s[0]
fadd v3.2s, v2.2s, v1.2s
mov v5.s[0], v0.s[1]
mov v4.s[1], v2.s[0]
mov v5.s[1], v2.s[1]
mov v4.s[2], v3.s[0]
fadd v2.2s, v3.2s, v1.2s
mov v5.s[2], v3.s[1]
mov v4.s[3], v2.s[0]
mov v5.s[3], v2.s[1]
dup v23.4s, w6
dup v22.2d, x0
sampler_c1_near_L4Loop:
fcvtns v6.4s, v4.4s
fcvtns v7.4s, v5.4s
smin v6.4s, v6.4s, v16.4s
smin v7.4s, v7.4s, v17.4s
smax v6.4s, v6.4s, v19.4s
smax v7.4s, v7.4s, v19.4s
mla v6.4s, v7.4s, v23.4s
uxtl2 v7.2d, v6.4s
uxtl v6.2d, v6.2s
add v6.2d, v6.2d, v22.2d
add v7.2d, v7.2d, v22.2d
mov x12, v6.d[0]
mov x13, v6.d[1]
ld1 {v3.b}[0], [x12]
mov x12, v7.d[0]
ld1 {v3.b}[1], [x13]
fadd v5.4s, v26.4s, v5.4s
mov x13, v7.d[1]
ld1 {v3.b}[2], [x12]
fadd v4.4s, v25.4s, v4.4s
ld1 {v3.b}[3], [x13]
st1 {v3.s}[0], [x1], #4
sub x3, x3, #4
cmp x3, #4
bge sampler_c1_near_L4Loop
mov v0.s[0], v4.s[0]
mov v0.s[1], v5.s[0]
sampler_c1_near_L1:
cmp x3, #0
beq sampler_c1_near_End
mov v6.s[0], w4
mov v6.s[1], w5
mov w12, #1
sampler_c1_near_L1Loop:
fcvtns v2.2s, v0.2s
smax v2.2s, v2.2s, v19.2s
smin v2.2s, v2.2s, v6.2s
mov w4, v2.s[0]
mov w5, v2.s[1]
umull x5, w5, w6
umull x4, w4, w12
add x7, x0, x5
add x4, x4, x7
ld1 {v2.b}[0], [x4]
fadd v0.2s, v0.2s, v1.2s
subs x3, x3, #1
st1 {v2.b}[0], [x1], #1
bne sampler_c1_near_L1Loop
sampler_c1_near_End:
ret
.text
.align 5
// _TaskNV21ToRGBUnit
// void _TaskNV21ToRGBUnit(const unsigned char* y, unsigned char* rgb, size_t count, const unsigned char* uv)
// x0: y, x1: rgb, x2: count, x3: uv (UV interleaved)
asm_function _TaskNV21ToRGBUnit
mov w4, #73
movi v31.8b, #128
mov w12, #25
mov v0.h[0], w4
mov v0.h[1], w12
mov w4, #37
mov w12, #130
mov v0.h[2], w4
mov v0.h[3], w12
movi v4.8h, #0
nv21_rgb_LoopL1:
ld2 {v19.8b, v20.8b}, [x3], #16
usubl v18.8h, v20.8b, v31.8b
ld2 {v16.8b, v17.8b}, [x0], #16
usubl v19.8h, v19.8b, v31.8b
mul v1.8h, v19.8h, v0.h[0]
mul v2.8h, v18.8h, v0.h[1]
mul v3.8h, v18.8h, v0.h[3]
mla v2.8h, v19.8h, v0.h[2]
ushll v16.8h, v16.8b, #6
ushll v17.8h, v17.8b, #6
add v20.8h, v16.8h, v1.8h
sub v21.8h, v16.8h, v2.8h
add v22.8h, v16.8h, v3.8h
smax v20.8h, v20.8h, v4.8h
smax v21.8h, v21.8h, v4.8h
smax v22.8h, v22.8h, v4.8h
uqshrn v20.8b, v20.8h, #6
uqshrn v21.8b, v21.8h, #6
uqshrn v22.8b, v22.8h, #6
add v23.8h, v17.8h, v1.8h
sub v24.8h, v17.8h, v2.8h
add v25.8h, v17.8h, v3.8h
smax v23.8h, v23.8h, v4.8h
smax v24.8h, v24.8h, v4.8h
smax v25.8h, v25.8h, v4.8h
uqshrn v23.8b, v23.8h, #6
uqshrn v24.8b, v24.8h, #6
uqshrn v25.8b, v25.8h, #6
zip1 v5.8b, v20.8b, v23.8b
zip2 v20.8b, v20.8b, v23.8b
zip1 v6.8b, v21.8b, v24.8b
zip2 v21.8b, v21.8b, v24.8b
zip1 v7.8b, v22.8b, v25.8b
zip2 v22.8b, v22.8b, v25.8b
st3 {v5.8b, v6.8b, v7.8b}, [x1], #24
st3 {v20.8b, v21.8b, v22.8b}, [x1], #24
subs x2, x2, #1
bne nv21_rgb_LoopL1
ret
.text
.align 5
// _TaskNV21ToBGRUnit
// void _TaskNV21ToBGRUnit(const unsigned char* y, unsigned char* bgr, size_t count, const unsigned char* uv)
// x0: y, x1: bgr, x2: count, x3: uv (UV interleaved)
asm_function _TaskNV21ToBGRUnit
mov w4, #73
movi v31.8b, #128
mov w12, #25
mov v0.h[0], w4
mov v0.h[1], w12
mov w4, #37
mov w12, #130
mov v0.h[2], w4
mov v0.h[3], w12
movi v4.8h, #0
nv21_bgr_LoopL1:
ld2 {v19.8b, v20.8b}, [x3], #16
usubl v18.8h, v20.8b, v31.8b
ld2 {v16.8b, v17.8b}, [x0], #16
usubl v19.8h, v19.8b, v31.8b
mul v1.8h, v18.8h, v0.h[3]
mul v2.8h, v18.8h, v0.h[1]
mla v2.8h, v19.8h, v0.h[2]
mul v3.8h, v19.8h, v0.h[0]
ushll v16.8h, v16.8b, #6
ushll v17.8h, v17.8b, #6
add v20.8h, v16.8h, v1.8h
sub v21.8h, v16.8h, v2.8h
add v22.8h, v16.8h, v3.8h
smax v20.8h, v20.8h, v4.8h
smax v21.8h, v21.8h, v4.8h
smax v22.8h, v22.8h, v4.8h
uqshrn v20.8b, v20.8h, #6
uqshrn v21.8b, v21.8h, #6
uqshrn v22.8b, v22.8h, #6
add v23.8h, v17.8h, v1.8h
sub v24.8h, v17.8h, v2.8h
add v25.8h, v17.8h, v3.8h
smax v23.8h, v23.8h, v4.8h
smax v24.8h, v24.8h, v4.8h
smax v25.8h, v25.8h, v4.8h
uqshrn v23.8b, v23.8h, #6
uqshrn v24.8b, v24.8h, #6
uqshrn v25.8b, v25.8h, #6
zip1 v5.8b, v20.8b, v23.8b
zip2 v20.8b, v20.8b, v23.8b
zip1 v6.8b, v21.8b, v24.8b
zip2 v21.8b, v21.8b, v24.8b
zip1 v7.8b, v22.8b, v25.8b
zip2 v22.8b, v22.8b, v25.8b
st3 {v5.8b, v6.8b, v7.8b}, [x1], #24
st3 {v20.8b, v21.8b, v22.8b}, [x1], #24
subs x2, x2, #1
bne nv21_bgr_LoopL1
ret
.text
.align 5
// _TaskNV21ToRGBAUnit
// void _TaskNV21ToRGBAUnit(const unsigned char* y, unsigned char* rgba, size_t count, const unsigned char* uv)
// x0: y, x1: rgba, x2: count, x3: uv (UV interleaved)
asm_function _TaskNV21ToRGBAUnit
mov w4, #73
movi v31.8b, #128
mov w12, #25
mov v0.h[0], w4
mov v0.h[1], w12
mov w4, #37
mov w12, #130
mov v0.h[2], w4
mov v0.h[3], w12
movi v30.8h, #0
movi v7.8b, #255
movi v26.8b, #255
nv21_rgba_LoopL1:
ld2 {v19.8b, v20.8b}, [x3], #16
ld2 {v16.8b, v17.8b}, [x0], #16
usubl v18.8h, v20.8b, v31.8b
usubl v19.8h, v19.8b, v31.8b
mul v1.8h, v19.8h, v0.h[0]
mul v2.8h, v18.8h, v0.h[1]
mul v3.8h, v18.8h, v0.h[3]
mla v2.8h, v19.8h, v0.h[2]
ushll v16.8h, v16.8b, #6
ushll v17.8h, v17.8b, #6
add v20.8h, v16.8h, v1.8h
sub v21.8h, v16.8h, v2.8h
add v22.8h, v16.8h, v3.8h
smax v20.8h, v20.8h, v30.8h
smax v21.8h, v21.8h, v30.8h
smax v22.8h, v22.8h, v30.8h
uqshrn v20.8b, v20.8h, #6
uqshrn v21.8b, v21.8h, #6
uqshrn v22.8b, v22.8h, #6
add v23.8h, v17.8h, v1.8h
sub v24.8h, v17.8h, v2.8h
add v25.8h, v17.8h, v3.8h
smax v23.8h, v23.8h, v30.8h
smax v24.8h, v24.8h, v30.8h
smax v25.8h, v25.8h, v30.8h
uqshrn v23.8b, v23.8h, #6
uqshrn v24.8b, v24.8h, #6
uqshrn v25.8b, v25.8h, #6
zip1 v4.8b, v20.8b, v23.8b
zip2 v23.8b, v20.8b, v23.8b
zip1 v5.8b, v21.8b, v24.8b
zip2 v24.8b, v21.8b, v24.8b
zip1 v6.8b, v22.8b, v25.8b
zip2 v25.8b, v22.8b, v25.8b
st4 {v4.8b, v5.8b, v6.8b, v7.8b}, [x1], #32
st4 {v23.8b, v24.8b, v25.8b, v26.8b}, [x1], #32
subs x2, x2, #1
bne nv21_rgba_LoopL1
ret
.text
.align 5
// _TaskNV21ToBGRAUnit
// void _TaskNV21ToBGRAUnit(const unsigned char* y, unsigned char* bgra, size_t count, const unsigned char* uv)
// x0: y, x1: bgra, x2: count, x3: uv (UV interleaved)
asm_function _TaskNV21ToBGRAUnit
mov w4, #73
movi v31.8b, #128
mov w12, #25
mov v0.h[0], w4
mov v0.h[1], w12
mov w4, #37
mov w12, #130
mov v0.h[2], w4
mov v0.h[3], w12
movi v30.8h, #0
movi v7.8b, #255
movi v26.8b, #255
nv21_bgra_LoopL1:
ld2 {v19.8b, v20.8b}, [x3], #16
ld2 {v16.8b, v17.8b}, [x0], #16
usubl v18.8h, v20.8b, v31.8b
usubl v19.8h, v19.8b, v31.8b
mul v1.8h, v18.8h, v0.h[3]
mul v2.8h, v18.8h, v0.h[1]
mla v2.8h, v19.8h, v0.h[2]
mul v3.8h, v19.8h, v0.h[0]
ushll v16.8h, v16.8b, #6
ushll v17.8h, v17.8b, #6
add v20.8h, v16.8h, v1.8h
sub v21.8h, v16.8h, v2.8h
add v22.8h, v16.8h, v3.8h
smax v20.8h, v20.8h, v30.8h
smax v21.8h, v21.8h, v30.8h
smax v22.8h, v22.8h, v30.8h
uqshrn v20.8b, v20.8h, #6
uqshrn v21.8b, v21.8h, #6
uqshrn v22.8b, v22.8h, #6
add v23.8h, v17.8h, v1.8h
sub v24.8h, v17.8h, v2.8h
add v25.8h, v17.8h, v3.8h
smax v23.8h, v23.8h, v30.8h
smax v24.8h, v24.8h, v30.8h
smax v25.8h, v25.8h, v30.8h
uqshrn v23.8b, v23.8h, #6
uqshrn v24.8b, v24.8h, #6
uqshrn v25.8b, v25.8h, #6
zip1 v4.8b, v20.8b, v23.8b
zip2 v23.8b, v20.8b, v23.8b
zip1 v5.8b, v21.8b, v24.8b
zip2 v24.8b, v21.8b, v24.8b
zip1 v6.8b, v22.8b, v25.8b
zip2 v25.8b, v22.8b, v25.8b
st4 {v4.8b, v5.8b, v6.8b, v7.8b}, [x1], #32
st4 {v23.8b, v24.8b, v25.8b, v26.8b}, [x1], #32
subs x2, x2, #1
bne nv21_bgra_LoopL1
ret
.text
.align 5
// _TaskBlitC3ToFloatRGBA
// void _TaskBlitC3ToFloatRGBA(const unsigned char* source, float* dest,
//                             const float* mean, const float* normal, size_t count)
// x0: source, x1: dest, x2: mean, x3: normal, [sp]: count
asm_function _TaskBlitC3ToFloatRGBA
ld1 {v22.4s}, [x2]
ld1 {v23.4s}, [x3]
blit_c3_rgba_L8:
cmp x4, #8
blt blit_c3_rgba_L1
dup v24.4s, v22.s[0]
dup v25.4s, v22.s[1]
dup v26.4s, v22.s[2]
dup v27.4s, v23.s[0]
dup v28.4s, v23.s[1]
dup v29.4s, v23.s[2]
blit_c3_rgba_LoopL8:
ld3 {v0.8b, v1.8b, v2.8b}, [x0], #24
uxtl v18.8h, v0.8b
uxtl v19.8h, v1.8b
uxtl v20.8h, v2.8b
uxtl v0.4s, v18.4h
uxtl2 v16.4s, v18.8h
uxtl v1.4s, v19.4h
uxtl2 v17.4s, v19.8h
uxtl v2.4s, v20.4h
uxtl2 v18.4s, v20.8h
ucvtf v0.4s, v0.4s
ucvtf v1.4s, v1.4s
ucvtf v2.4s, v2.4s
movi v3.4s, #0
.macro COMPUTE
fsub v0.4s, v0.4s, v24.4s
fsub v1.4s, v1.4s, v25.4s
fsub v2.4s, v2.4s, v26.4s
fmul v0.4s, v0.4s, v27.4s
fmul v1.4s, v1.4s, v28.4s
fmul v2.4s, v2.4s, v29.4s
st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
.endm
COMPUTE
ucvtf v0.4s, v16.4s
ucvtf v1.4s, v17.4s
ucvtf v2.4s, v18.4s
COMPUTE
sub x4, x4, #8
cmp x4, #8
bge blit_c3_rgba_LoopL8
blit_c3_rgba_L1:
cmp x4, #0
beq blit_c3_rgba_End
blit_c3_rgba_LoopL1:
movi v0.8b, #0
ld1 {v0.b}[0], [x0], #1
ld1 {v0.b}[1], [x0], #1
ld1 {v0.b}[2], [x0], #1
uxtl v0.8h, v0.8b
uxtl v0.4s, v0.4h
ucvtf v0.4s, v0.4s
fsub v0.4s, v0.4s, v22.4s
fmul v0.4s, v0.4s, v23.4s
st1 {v0.4s}, [x1], #16
subs x4, x4, #1
bne blit_c3_rgba_LoopL1
blit_c3_rgba_End:
ret
.text
.align 5
// _TaskBlitC1ToFloatRGBA
// void _TaskBlitC1ToFloatRGBA(const unsigned char* source, float* dest,
//                             const float* mean, const float* normal, size_t count)
// x0: source, x1: dest, x2: mean, x3: normal, [sp]: count
asm_function _TaskBlitC1ToFloatRGBA
ldr w5, [x2]
dup v22.4s, w5
ldr w6, [x3]
dup v23.4s, w6
blit_c1_rgba_L8:
cmp x4, #8
blt blit_c1_rgba_L1
blit_c1_rgba_LoopL8:
ld1 {v0.8b}, [x0], #8
uxtl v0.8h, v0.8b
uxtl2 v1.4s, v0.8h
uxtl v0.4s, v0.4h
ucvtf v1.4s, v1.4s
ucvtf v0.4s, v0.4s
fsub v1.4s, v1.4s, v22.4s
fsub v0.4s, v0.4s, v22.4s
fmul v1.4s, v1.4s, v23.4s
fmul v0.4s, v0.4s, v23.4s
movi v16.4s, #0
movi v17.4s, #0
movi v18.4s, #0
movi v19.4s, #0
mov v16.s[0], v0.s[0]
mov v17.s[0], v0.s[1]
mov v18.s[0], v0.s[2]
st1 {v16.4s, v17.4s}, [x1],#32
mov v19.s[0], v0.s[3]
st1 {v18.4s, v19.4s}, [x1],#32
mov v16.s[0], v1.s[0]
mov v17.s[0], v1.s[1]
mov v18.s[0], v1.s[2]
st1 {v16.4s, v17.4s}, [x1],#32
mov v19.s[0], v1.s[3]
st1 {v18.4s, v19.4s}, [x1],#32
sub x4, x4, #8
cmp x4, #8
bge blit_c1_rgba_LoopL8
blit_c1_rgba_L1:
cmp x4, #0
beq blit_c1_rgba_End
blit_c1_rgba_LoopL1:
ld1 {v0.b}[0], [x0], #1
uxtl v0.8h, v0.8b
movi v1.4s, #0
uxtl v0.4s, v0.4h
mov v1.s[0], v0.s[0]
ucvtf v1.4s, v1.4s
fsub v1.4s, v1.4s, v22.4s
fmul v0.4s, v1.4s, v23.4s
st1 {v0.4s}, [x1], #16
subs x4, x4, #1
bne blit_c1_rgba_LoopL1
blit_c1_rgba_End:
ret

