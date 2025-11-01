.macro asm_function fname
.globl _\fname
_\fname:
.endm

.text
.align 5
// _TaskSamplerC4BilinearOpt
// void _TaskSamplerC4BilinearOpt(const unsigned char* source, unsigned char* dest,
//                                 float* points, size_t count, size_t iw, size_t ih, size_t yStride)
// r0: source, r1: dest, r2: points, r3: count, [sp]: xMax,yMax,yStride
asm_function _TaskSamplerC4BilinearOpt
push {r4-r8, r10, r11, lr}
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
mov r12, #4
vpush {q4-q7}
vmov.i32 q15, #0
vdup.i32 q14, r5
vdup.i32 q13, r4
vld1.32 {q12}, [r2]
c4_bi_L4:
cmp r3, #4
blt c4_bi_L1
c4_bi_LoopL4:
    vmov.32 d4, d24
    vadd.f32 d6, d24, d25
    vadd.f32 d5, d6, d25
    vadd.f32 d7, d5, d25
    vtrn.32 d4, d6
    vadd.f32 d24, d7, d25
    vtrn.32 d5, d7
    sub r3, r3, #4
    vcvt.s32.f32 q8, q2
    vcvt.s32.f32 q9, q3
    vcvt.f32.s32 q10, q8
    vcvt.f32.s32 q11, q9
    vsub.f32 q0, q2, q10
    vsub.f32 q1, q3, q11
    vmax.s32 q8, q8, q15
    vmax.s32 q9, q9, q15
    vmov.i32 q2, #1
    vadd.s32 q10, q8, q2
    vadd.s32 q11, q9, q2
    vmin.s32 q8, q8, q13
    vmin.s32 q9, q9, q14
    vmin.s32 q10, q10, q13
    vmin.s32 q11, q11, q14
    vmov.i32 d4[0], r6
    vmov.i32 d4[1], r12
    vmul.u32 q11, q11, d4[0]
    vmul.u32 q9, q9, d4[0]
    vmul.u32 q8, q8, d4[1]
    vmul.u32 q10, q10, d4[1]
    vadd.u32 q2, q8, q9
    vadd.u32 q3, q8, q11
    vadd.u32 q8, q10, q9
    vadd.u32 q9, q10, q11
    vdup.i32 q10, r0
    vadd.u32 q2, q2, q10
    vadd.u32 q3, q3, q10
    vadd.u32 q8, q8, q10
    vadd.u32 q9, q9, q10
    vmov.i32 r7, d4[0]
    vld1.32 {d20[0]}, [r7]
    vmov.i32 r7, d6[0]
    vld1.32 {d22[0]}, [r7]
    vmov.i32 r7, d16[0]
    vld1.32 {d20[1]}, [r7]
    vmov.i32 r7, d18[0]
    vld1.32 {d22[1]}, [r7]
    vmovl.u8 q10, d20
    vmovl.u8 q11, d22
    vmovl.s16 q4, d20
    vmovl.s16 q5, d21
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5
    vmovl.s16 q6, d22
    vmovl.s16 q7, d23
    vcvt.f32.s32 q6, q6
    vcvt.f32.s32 q7, q7
    vmov.i32 r7, d4[1]
    vsub.f32 q5, q5, q4
    vld1.32 {d20[0]}, [r7]
    vsub.f32 q7, q7, q6
    vmla.f32 q4, q5, d0[0]
    vmov.i32 r7, d6[1]
    vmla.f32 q6, q7, d0[0]
    vld1.32 {d22[0]}, [r7]
    vsub.f32 q5, q6, q4
    vmov.i32 r7, d16[1]
    vmla.f32 q4, q5, d2[0]
    vld1.32 {d20[1]}, [r7]
    vcvt.s32.f32 q4, q4
    vmov.i32 r7, d18[1]
    vmovn.u32 d8, q4
    vld1.32 {d22[1]}, [r7]
    vmovn.u16 d8, q4
    vst1.32 {d8[0]}, [r1]!
    vmovl.u8 q10, d20
    vmovl.u8 q11, d22
    vmovl.s16 q4, d20
    vmovl.s16 q5, d21
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5
    vmovl.s16 q6, d22
    vmovl.s16 q7, d23
    vcvt.f32.s32 q6, q6
    vcvt.f32.s32 q7, q7
    vmov.i32 r7, d5[0]
    vsub.f32 q5, q5, q4
    vld1.32 {d20[0]}, [r7]
    vsub.f32 q7, q7, q6
    vmov.i32 r7, d7[0]
    vmla.f32 q4, q5, d0[1]
    vld1.32 {d22[0]}, [r7]
    vmla.f32 q6, q7, d0[1]
    vmov.i32 r7, d17[0]
    vsub.f32 q5, q6, q4
    vld1.32 {d20[1]}, [r7]
    vmla.f32 q4, q5, d2[1]
    vmov.i32 r7, d19[0]
    vcvt.s32.f32 q4, q4
    vld1.32 {d22[1]}, [r7]
    vmovn.u32 d8, q4
    vmovn.u16 d8, q4
    vst1.32 {d8[0]}, [r1]!
    vmovl.u8 q10, d20
    vmovl.u8 q11, d22
    vmovl.s16 q4, d20
    vmovl.s16 q5, d21
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5
    vmovl.s16 q6, d22
    vmovl.s16 q7, d23
    vcvt.f32.s32 q6, q6
    vcvt.f32.s32 q7, q7
    vmov.i32 r7, d5[1]
    vsub.f32 q5, q5, q4
    vld1.32 {d20[0]}, [r7]
    vsub.f32 q7, q7, q6
    vmov.i32 r7, d7[1]
    vmla.f32 q4, q5, d1[0]
    vld1.32 {d22[0]}, [r7]
    vmla.f32 q6, q7, d1[0]
    vmov.i32 r7, d17[1]
    vsub.f32 q5, q6, q4
    vld1.32 {d20[1]}, [r7]
    vmla.f32 q4, q5, d3[0]
    vmov.i32 r7, d19[1]
    vcvt.s32.f32 q4, q4
    vld1.32 {d22[1]}, [r7]
    vmovn.u32 d8, q4
    vmovl.u8 q10, d20
    vmovn.u16 d8, q4
    vst1.32 {d8[0]}, [r1]!
    vmovl.u8 q11, d22
    vmovl.s16 q4, d20
    vmovl.s16 q5, d21
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5
    vmovl.s16 q6, d22
    vmovl.s16 q7, d23
    vcvt.f32.s32 q6, q6
    vcvt.f32.s32 q7, q7
    vsub.f32 q5, q5, q4
    vsub.f32 q7, q7, q6
    vmla.f32 q4, q5, d1[1]
    vmla.f32 q6, q7, d1[1]
    vsub.f32 q5, q6, q4
    vmla.f32 q4, q5, d3[1]
    vcvt.s32.f32 q4, q4
    vmovn.u32 d8, q4
    vmovn.u16 d8, q4
    vst1.32 {d8[0]}, [r1]!
    cmp r3, #4
    bge c4_bi_LoopL4
c4_bi_L1:
cmp r3, #0
beq End
vmov.i32 d26[0], r4
vmov.i32 d26[1], r5
vmov.i32 d28, #1
vcvt.f32.s32 d27, d26
c4_bi_LoopL1:
    vmov.32 d0, d24
    vadd.f32 d24, d24, d25
    vmax.f32 d0, d0, d30
    vcvt.s32.f32 d1, d0
    vadd.s32 d2, d1, d28
    vcvt.f32.s32 d3, d1
    vmin.s32 d2, d2, d26
    vmin.s32 d1, d1, d26
    vsub.f32 d3, d0, d3
    vmov.i32 r7, d1[0]
    mul r7, r7, r12
    vmov.i32 r8, d1[1]
    mul r8, r8, r6
    add r8, r8, r0
    add r7, r8, r7
    vld1.32 {d16[0]}, [r7]
    vmov.i32 r7, d2[0]
    mul r7, r7, r12
    add r7, r8, r7
    vld1.32 {d16[1]}, [r7]
    vmov.i32 r7, d1[0]
    mul r7, r7, r12
    vmov.i32 r8, d2[1]
    mul r8, r8, r6
    add r8, r8, r0
    add r7, r8, r7
    vld1.32 {d18[0]}, [r7]
    vmov.i32 r7, d2[0]
    mul r7, r7, r12
    add r7, r8, r7
    vld1.32 {d18[1]}, [r7]
    vmovl.u8 q8, d16
    vmovl.u8 q9, d18
    vmovl.s16 q0, d16
    vmovl.s16 q2, d17
    vcvt.f32.s32 q0, q0
    vcvt.f32.s32 q2, q2
    vsub.f32 q8, q2, q0
    vmla.f32 q0, q8, d3[0]
    vmovl.s16 q10, d18
    vmovl.s16 q11, d19
    vcvt.f32.s32 q10, q10
    vcvt.f32.s32 q11, q11
    vsub.f32 q8, q11, q10
    vmla.f32 q10, q8, d3[0]
    vsub.f32 q2, q10, q0
    vmla.f32 q0, q2, d3[1]
    vcvt.s32.f32 q0, q0
    vmovn.u32 d0, q0
    vmovn.u16 d0, q0
    vst1.32 {d0[0]}, [r1]!
    subs r3, r3, #1
    bne c4_bi_LoopL1
End:
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}
.text
.align 5
// _TaskSamplerC1BilinearOpt
// void _TaskSamplerC1BilinearOpt(const unsigned char* source, unsigned char* dest,
//                                float* points, size_t count, size_t iw, size_t ih, size_t yStride)
// r0: source, r1: dest, r2: points, r3: count, [sp]: xMax,yMax,yStride
asm_function _TaskSamplerC1BilinearOpt
push {r4-r8, r10, r11, lr}
ldr r4, [sp, #32]
ldr r5, [sp, #36]
ldr r6, [sp, #40]
vpush {q4-q7}
vmov.i32 q15, #0
vmov.i32 q12, #1
vdup.i32 q14, r5
vdup.i32 q13, r4
vcvt.f32.s32 q14, q14
vcvt.f32.s32 q13, q13
vld1.32 {q6}, [r2]
c1_bi_L4:
cmp r3, #4
blt c1_bi_L1
c1_bi_LoopL4:
    vmov.32 d4, d12
    vadd.f32 d6, d12, d13
    vadd.f32 d5, d6, d13
    vadd.f32 d7, d5, d13
    vtrn.32 d4, d6
    vadd.f32 d12, d7, d13
    vtrn.32 d5, d7
    sub r3, r3, #4
    vmin.f32 q2, q2, q13
    vmin.f32 q3, q3, q14
    vmax.f32 q2, q2, q15
    vmax.f32 q3, q3, q15
    vcvt.s32.f32 q8, q2
    vcvt.s32.f32 q9, q3
    vcvt.f32.s32 q10, q8
    vcvt.f32.s32 q11, q9
    vsub.f32 q0, q2, q10
    vsub.f32 q1, q3, q11
    vdup.i32 q2, r4
    vdup.i32 q3, r5
    vadd.s32 q10, q8, q12
    vadd.s32 q11, q9, q12
    vmin.s32 q10, q10, q2
    vmin.s32 q11, q11, q3
    vdup.i32 q2, r6
    vmul.s32 q11, q2, q11
    vmul.s32 q9, q2, q9
    vadd.s32 q2, q8, q9
    vadd.s32 q3, q8, q11
    vadd.s32 q8, q10, q9
    vdup.i32 q4, r0
    vadd.s32 q9, q10, q11
    vadd.u32 q2, q2, q4
    vadd.u32 q3, q3, q4
    vadd.u32 q8, q8, q4
    vadd.u32 q9, q9, q4
    vmov.i32 r7, d4[0]
    vld1.8 {d8[0]}, [r7]
    vmov.i32 r7, d4[1]
    vld1.8 {d8[1]}, [r7]
    vmov.i32 r7, d5[0]
    vld1.8 {d8[2]}, [r7]
    vmov.i32 r7, d5[1]
    vld1.8 {d8[3]}, [r7]
    vmov.i32 r7, d6[0]
    vld1.8 {d8[4]}, [r7]
    vmov.i32 r7, d6[1]
    vld1.8 {d8[5]}, [r7]
    vmov.i32 r7, d7[0]
    vld1.8 {d8[6]}, [r7]
    vmov.i32 r7, d7[1]
    vld1.8 {d8[7]}, [r7]
    vmov.i32 r7, d16[0]
    vld1.8 {d10[0]}, [r7]
    vmov.i32 r7, d16[1]
    vld1.8 {d10[1]}, [r7]
    vmov.i32 r7, d17[0]
    vld1.8 {d10[2]}, [r7]
    vmov.i32 r7, d17[1]
    vld1.8 {d10[3]}, [r7]
    vmov.i32 r7, d18[0]
    vld1.8 {d10[4]}, [r7]
    vmov.i32 r7, d18[1]
    vld1.8 {d10[5]}, [r7]
    vmov.i32 r7, d19[0]
    vld1.8 {d10[6]}, [r7]
    vmov.i32 r7, d19[1]
    vld1.8 {d10[7]}, [r7]
    vmovl.u8 q4, d8
    vmovl.u8 q5, d10
    vmovl.s16 q2, d8
    vmovl.s16 q3, d9
    vcvt.f32.s32 q2, q2
    vcvt.f32.s32 q3, q3
    vmovl.s16 q4, d10
    vmovl.s16 q5, d11
    vcvt.f32.s32 q4, q4
    vcvt.f32.s32 q5, q5
    vsub.f32 q4, q4, q2
    vsub.f32 q5, q5, q3
    vmla.f32 q2, q4, q0
    vmla.f32 q3, q5, q0
    vsub.f32 q3, q3, q2
    vmla.f32 q2, q3, q1
    vcvt.s32.f32 q0, q2
    vmovn.u32 d0, q0
    vmovn.u16 d0, q0
    vst1.32 {d0[0]}, [r1]!
    cmp r3, #4
    bge c1_bi_LoopL4
c1_bi_L1:
cmp r3, #0
beq End
vmov.i32 d26[0], r4
vmov.i32 d26[1], r5
vcvt.f32.s32 d27, d26
c1_bi_LoopL1:
    vmov.32 d0, d12
    vadd.f32 d12, d12, d13
    vmin.f32 d0, d0, d27
    vmax.f32 d0, d0, d30
    vcvt.s32.f32 d1, d0
    vadd.s32 d2, d1, d24
    vcvt.f32.s32 d3, d1
    vmin.s32 d2, d2, d26
    vsub.f32 d3, d0, d3
    vmov.i32 r7, d1[0]
    vmov.i32 r8, d1[1]
    mul r8, r8, r6
    add r8, r8, r0
    add r7, r8, r7
    vld1.8 {d16[0]}, [r7]
    vmov.i32 r7, d2[0]
    add r7, r8, r7
    vld1.8 {d16[1]}, [r7]
    vmov.i32 r7, d1[0]
    vmov.i32 r8, d2[1]
    mul r8, r8, r6
    add r8, r8, r0
    add r7, r8, r7
    vld1.8 {d16[2]}, [r7]
    vmov.i32 r7, d2[0]
    add r7, r8, r7
    vld1.8 {d16[3]}, [r7]
    vmovl.u8 q8, d16
    vmovl.s16 q0, d16
    vcvt.f32.s32 q0, q0
    vsub.f32 s8, s1, s0
    vsub.f32 s9, s3, s2
    vmla.f32 s0, s8, s6
    vmla.f32 s2, s9, s6
    vsub.f32 s2, s2, s0
    vmla.f32 s0, s2, s7
    vcvt.s32.f32 s0, s0
    vmovn.u32 d0, q0
    vmovn.u16 d0, q0
    vst1.8 {d0[0]}, [r1]!
    subs r3, r3, #1
    bne c1_bi_LoopL1
End:
vpop {q4-q7}
pop {r4-r8, r10, r11, pc}
.text
.align 5
// _TaskSamplerC1NearestOpt
// void _TaskSamplerC1NearestOpt(const unsigned char* source, unsigned char* dest,
//                               float* points, size_t count, size_t iw, size_t ih, size_t yStride)
// r0: source, r1: dest, r2: points, r3: count, [sp]: xMax,yMax,yStride
asm_function _TaskSamplerC1NearestOpt
push {r4-r8, r10, r11, lr}
ldr r4, [sp, #32]
ldr r5, [sp, #36]
vdup.32 q12, r4
vdup.32 q13, r5
vmov.i32 q11, #0
ldr r6, [sp, #40]
vld1.32 {q0}, [r2]
c1_near_L8:
cmp r3, #8
blt c1_near_L1
vdup.32 q15, r6
vdup.32 q14, r0
c1_near_L8Loop:
vcvtr.s32.f32 s8, s0
vcvtr.s32.f32 s12, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s9, s6
vcvtr.s32.f32 s13, s7
vadd.f32 d0, d3, d1
vcvtr.s32.f32 s10, s0
vcvtr.s32.f32 s14, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s11, s6
vcvtr.s32.f32 s15, s7
vmax.s32 q2, q2, q11
vmax.s32 q3, q3, q11
vmin.s32 q2, q2, q12
vmin.s32 q3, q3, q13
vadd.f32 d0, d3, d1
vmla.s32 q2, q15, q3
vadd.s32 q2, q2, q14
vmov.i32 r4, d4[0]
vmov.i32 r5, d4[1]
vld1.8 {d16[0]}, [r4]
vcvtr.s32.f32 s8, s0
vld1.8 {d16[1]}, [r5]
vmov.i32 r4, d5[0]
vmov.i32 r5, d5[1]
vld1.8 {d16[2]}, [r4]
vcvtr.s32.f32 s12, s1
vld1.8 {d16[3]}, [r5]
vadd.f32 d3, d0, d1
vst1.32 {d16[0]}, [r1]!
vcvtr.s32.f32 s9, s6
vcvtr.s32.f32 s13, s7
vadd.f32 d0, d3, d1
vcvtr.s32.f32 s10, s0
vcvtr.s32.f32 s14, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s11, s6
vcvtr.s32.f32 s15, s7
vadd.f32 d0, d3, d1
vmax.s32 q2, q2, q11
vmax.s32 q3, q3, q11
vmin.s32 q2, q2, q12
vmin.s32 q3, q3, q13
vmla.s32 q2, q15, q3
vadd.s32 q2, q2, q14
vmov.i32 r4, d4[0]
vmov.i32 r5, d4[1]
vld1.8 {d16[0]}, [r4]
vld1.8 {d16[1]}, [r5]
vmov.i32 r4, d5[0]
vmov.i32 r5, d5[1]
vld1.8 {d16[2]}, [r4]
vld1.8 {d16[3]}, [r5]
vst1.32 {d16[0]}, [r1]!
sub r3, r3, #8
cmp r3, #8
bge c1_near_L8Loop
c1_near_L1:
cmp r3, #0
beq End
c1_near_L1Loop:
vcvtr.s32.f32 s4, s0
vcvtr.s32.f32 s6, s1
vmax.s32 d2, d2, d22
vmax.s32 d3, d3, d22
vmin.s32 d2, d2, d24
vmin.s32 d3, d3, d26
vmov.i32 r4, d2[0]
vmov.i32 r5, d3[0]
mul r5, r5, r6
add r4, r4, r0
add r4, r4, r5
vld1.8 {d3[0]}, [r4]
vadd.f32 d0, d0, d1
vst1.8 {d3[0]}, [r1]!
subs r3, r3, #1
bne c1_near_L1Loop
End:
pop {r4-r8, r10, r11, pc}
.text
.align 5
// _TaskSamplerC4NearestOpt
// void _TaskSamplerC4NearestOpt(const unsigned char* source, unsigned char* dest,
//                               float* points, size_t count, size_t iw, size_t ih, size_t yStride)
// r0: source, r1: dest, r2: points, r3: count, [sp]: xMax,yMax,yStride
asm_function _TaskSamplerC4NearestOpt
push {r4-r8, r10, r11, lr}
ldr r4, [sp, #32]
l dr r5, [sp, #36]
vdup.32 q12, r4
vdup.32 q13, r5
vmov.i32 q11, #0
vmov.i32 q10, #4
ldr r6, [sp, #40]
vld1.32 {q0}, [r2]
c4_near_L8:
cmp r3, #8
blt c4_near_L1
vdup.32 q15, r6
vdup.32 q14, r0
c4_near_L8Loop:
vcvtr.s32.f32 s8, s0
vcvtr.s32.f32 s12, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s9, s6
vcvtr.s32.f32 s13, s7
vadd.f32 d0, d3, d1
vcvtr.s32.f32 s10, s0
vcvtr.s32.f32 s14, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s11, s6
vcvtr.s32.f32 s15, s7
vmax.s32 q2, q2, q11
vmax.s32 q3, q3, q11
vmin.s32 q2, q2, q12
vmin.s32 q3, q3, q13
vmul.s32 q2, q2, q10
vadd.f32 d0, d3, d1
vmla.s32 q2, q15, q3
vadd.s32 q2, q2, q14
vmov.i32 r4, d4[0]
vmov.i32 r5, d4[1]
vld1.32 {d16[0]}, [r4]
vcvtr.s32.f32 s8, s0
vld1.32 {d16[1]}, [r5]
vmov.i32 r4, d5[0]
vmov.i32 r5, d5[1]
vld1.32 {d17[0]}, [r4]
vcvtr.s32.f32 s12, s1
vld1.32 {d17[1]}, [r5]
vadd.f32 d3, d0, d1
vst1.32 {q8}, [r1]!
vcvtr.s32.f32 s9, s6
vcvtr.s32.f32 s13, s7
vadd.f32 d0, d3, d1
vcvtr.s32.f32 s10, s0
vcvtr.s32.f32 s14, s1
vadd.f32 d3, d0, d1
vcvtr.s32.f32 s11, s6
vcvtr.s32.f32 s15, s7
vadd.f32 d0, d3, d1
vmax.s32 q2, q2, q11
vmax.s32 q3, q3, q11
vmin.s32 q2, q2, q12
vmin.s32 q3, q3, q13
vmul.s32 q2, q2, q10
vmla.s32 q2, q15, q3
vadd.s32 q2, q2, q14
vmov.i32 r4, d4[0]
vmov.i32 r5, d4[1]
vld1.32 {d16[0]}, [r4]
vld1.32 {d16[1]}, [r5]
vmov.i32 r4, d5[0]
vmov.i32 r5, d5[1]
vld1.32 {d17[0]}, [r4]
vld1.32 {d17[1]}, [r5]
vst1.32 {q8}, [r1]!
sub r3, r3, #8
cmp r3, #8
bge c4_near_L8Loop
c4_near_L1:
cmp r3, #0
mov r12, #4
beq End
c4_near_L1Loop:
vcvtr.s32.f32 s4, s0
vcvtr.s32.f32 s6, s1
vmax.s32 d2, d2, d22
vmax.s32 d3, d3, d22
vmin.s32 d2, d2, d24
vmin.s32 d3, d3, d26
vmov.i32 r4, d2[0]
vmov.i32 r5, d3[0]
mul r4, r4, r12
add r4, r4, r0
mul r5, r5, r6
add r4, r4, r5
vld1.32 {d3[0]}, [r4]
vadd.f32 d0, d0, d1
vst1.32 {d3[0]}, [r1]!
subs r3, r3, #1
bne c4_near_L1Loop
End:
pop {r4-r8, r10, r11, pc}
.text
.align 5
// _TaskNV21ToRGBUnit
// void _TaskNV21ToRGBUnit(const unsigned char* y, unsigned char* rgb, size_t count, const unsigned char* uv)
// r0: y, r1: rgb, r2: count, r3: uv
asm_function _TaskNV21ToRGBUnit
push {r4, lr}
vpush {q4}
mov r4, #73
vmov.i8 d31, #128
mov r12, #25
vmov.i16 d0[0], r4
vmov.i16 d0[1], r12
mov r4, #37
mov r12, #130
vmov.i16 d0[2], r4
vmov.i16 d0[3], r12
vmov.i16 q4, #0
nv21_rgb_LoopL1:
vld2.8 {d18, d19}, [r3]!
vsubl.u8 q11, d18, d31
vld2.8 {d16, d17}, [r0]!
vsubl.u8 q10, d19, d31
vmul.s16 q1, q11, d0[0]
vmul.s16 q2, q10, d0[1]
vmul.s16 q3, q10, d0[3]
vmla.s16 q2, q11, d0[2]
vshll.u8 q10, d16, #6
vshll.u8 q11, d17, #6
vadd.s16 q12, q10, q1
vsub.s16 q13, q10, q2
vadd.s16 q14, q10, q3
vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4
vqshrn.u16 d16, q12, #6
vqshrn.u16 d17, q13, #6
vqshrn.u16 d18, q14, #6
vadd.s16 q12, q11, q1
vsub.s16 q13, q11, q2
vadd.s16 q14, q11, q3
vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4
vqshrn.u16 d19, q12, #6
vqshrn.u16 d20, q13, #6
vqshrn.u16 d21, q14, #6
vzip.8 d16, d19
vzip.8 d17, d20
vzip.8 d18, d21
vst3.u8 {d16, d17, d18}, [r1]!
vst3.u8 {d19, d20, d21}, [r1]!
subs r2, r2, #1
bne nv21_rgb_LoopL1
vpop {q4}
pop {r4, pc}
.text
.align 5
// _TaskNV21ToBGRUnit
// void _TaskNV21ToBGRUnit(const unsigned char* y, unsigned char* bgr, size_t count, const unsigned char* uv)
// r0: y, r1: bgr, r2: count, r3: uv
asm_function _TaskNV21ToBGRUnit
push {r4, lr}
vpush {q4}
mov r4, #73
vmov.i8 d31, #128
mov r12, #25
vmov.i16 d0[0], r4
vmov.i16 d0[1], r12
mov r4, #37
mov r12, #130
vmov.i16 d0[2], r4
vmov.i16 d0[3], r12
vmov.i16 q4, #0
nv21_bgr_LoopL1:
vld2.8 {d18, d19}, [r3]!
vsubl.u8 q11, d18, d31
vld2.8 {d16, d17}, [r0]!
vsubl.u8 q10, d19, d31
vmul.s16 q1, q10, d0[3]
vmul.s16 q2, q10, d0[1]
vmla.s16 q2, q11, d0[2]
vmul.s16 q3, q11, d0[0]
vshll.u8 q10, d16, #6
vshll.u8 q11, d17, #6
vadd.s16 q12, q10, q1
vsub.s16 q13, q10, q2
vadd.s16 q14, q10, q3
vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4
vqshrn.u16 d16, q12, #6
vqshrn.u16 d17, q13, #6
vqshrn.u16 d18, q14, #6
vadd.s16 q12, q11, q1
vsub.s16 q13, q11, q2
vadd.s16 q14, q11, q3
vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4
vqshrn.u16 d19, q12, #6
vqshrn.u16 d20, q13, #6
vqshrn.u16 d21, q14, #6
vzip.8 d16, d19
vzip.8 d17, d20
vzip.8 d18, d21
vst3.u8 {d16, d17, d18}, [r1]!
vst3.u8 {d19, d20, d21}, [r1]!
subs r2, r2, #1
bne nv21_bgr_LoopL1
vpop {q4}
pop {r4, pc}
.text
.align 5
// _TaskNV21ToRGBAUnit
// void _TaskNV21ToRGBAUnit(const unsigned char* y, unsigned char* rgba, size_t count, const unsigned char* uv)
// r0: y, r1: rgba, r2: count, r3: uv
asm_function _TaskNV21ToRGBAUnit
push {r4, lr}
vpush {q4}
mov r4, #73
vmov.i8 d31, #128
mov r12, #25
vmov.i16 d0[0], r4
vmov.i16 d0[1], r12
mov r4, #37
mov r12, #130
vmov.i16 d0[2], r4
vmov.i16 d0[3], r12
vmov.i16 q4, #0
nv21_rgba_LoopL1:
vld2.8 {d18, d19}, [r3]!
vsubl.u8 q11, d18, d31
vld2.8 {d16, d17}, [r0]!
vsubl.u8 q10, d19, d31
vmul.s16 q1, q11, d0[0]
vmul.s16 q2, q10, d0[1]
vmul.s16 q3, q10, d0[3]
vmla.s16 q2, q11, d0[2]
vshll.u8 q10, d16, #6
vshll.u8 q11, d17, #6
vadd.s16 q12, q10, q1
vsub.s16 q13, q10, q2
vadd.s16 q14, q10, q3
vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4
vqshrn.u16 d16, q12, #6
vqshrn.u16 d17, q13, #6
vqshrn.u16 d18, q14, #6
vadd.s16 q12, q11, q1
vsub.s16 q13, q11, q2
vadd.s16 q14, q11, q3
vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4
vqshrn.u16 d20, q12, #6
vqshrn.u16 d21, q13, #6
vqshrn.u16 d22, q14, #6
vzip.8 d16, d20
vmov.i8 d19, #255
vzip.8 d17, d21
vzip.8 d18, d22
vmov.i8 d23, #255
vst4.u8 {d16, d17, d18, d19}, [r1]!
vst4.u8 {d20, d21, d22, d23}, [r1]!
subs r2, r2, #1
bne nv21_rgba_LoopL1
vpop {q4}
pop {r4, pc}
.text
.align 5
// _TaskNV21ToBGRAUnit
// void _TaskNV21ToBGRAUnit(const unsigned char* y, unsigned char* bgra, size_t count, const unsigned char* uv)
// r0: y, r1: bgra, r2: count, r3: uv
asm_function _TaskNV21ToBGRAUnit
push {r4, lr}
vpush {q4}
mov r4, #73
vmov.i8 d31, #128
mov r12, #25
vmov.i16 d0[0], r4
vmov.i16 d0[1], r12
mov r4, #37
mov r12, #130
vmov.i16 d0[2], r4
vmov.i16 d0[3], r12
vmov.i16 q4, #0
nv21_bgra_LoopL1:
vld2.8 {d18, d19}, [r3]!
vsubl.u8 q11, d18, d31
vld2.8 {d16, d17}, [r0]!
vsubl.u8 q10, d19, d31
vmul.s16 q1, q10, d0[3]
vmul.s16 q2, q10, d0[1]
vmla.s16 q2, q11, d0[2]
vmul.s16 q3, q11, d0[0]
vshll.u8 q10, d16, #6
vshll.u8 q11, d17, #6
vadd.s16 q12, q10, q1
vsub.s16 q13, q10, q2
vadd.s16 q14, q10, q3
vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4
vqshrn.u16 d16, q12, #6
vqshrn.u16 d17, q13, #6
vqshrn.u16 d18, q14, #6
vadd.s16 q12, q11, q1
vsub.s16 q13, q11, q2
vadd.s16 q14, q11, q3
vmax.s16 q12, q12, q4
vmax.s16 q13, q13, q4
vmax.s16 q14, q14, q4
vqshrn.u16 d20, q12, #6
vqshrn.u16 d21, q13, #6
vqshrn.u16 d22, q14, #6
vzip.8 d16, d20
vmov.i8 d19, #255
vzip.8 d17, d21
vzip.8 d18, d22
vmov.i8 d23, #255
vst4.u8 {d16, d17, d18, d19}, [r1]!
vst4.u8 {d20, d21, d22, d23}, [r1]!
subs r2, r2, #1
bne nv21_bgra_LoopL1
vpop {q4}
pop {r4, pc}
.text
.align 5
// _TaskBlitC3ToFloatRGBA
// void _TaskBlitC3ToFloatRGBA(const unsigned char* source, float* dest,
//                             const float* mean, const float* normal, size_t count)
// r0: source, r1: dest, r2: mean, r3: normal, [sp]: count
asm_function _TaskBlitC3ToFloatRGBA
push {r4, lr}
ldr r4, [sp, #8]
vld1.32 {q14}, [r2]
vld1.32 {q15}, [r3]
blit_c3_rgba_L8:
cmp r4, #8
blt blit_c3_rgba_L1
blit_c3_rgba_LoopL8:
vld3.8 {d0, d1, d2}, [r0]!
vmovl.u8 q10, d0
vmovl.u8 q11, d1
vmovl.u8 q12, d2
vmovl.u16 q0, d20
vmovl.u16 q8, d21
vmovl.u16 q1, d22
vmovl.u16 q9, d23
vmovl.u16 q2, d24
vmovl.u16 q10, d25
vcvt.f32.u32 q0, q0
vcvt.f32.u32 q1, q1
vcvt.f32.u32 q2, q2
.macro COMPUTE
vmov.i32 q3, #0
vtrn.32 d0, d2
vtrn.32 d1, d3
vtrn.32 d4, d6
vtrn.32 d5, d7
vswp d1, d4
vswp d3, d6
vsub.f32 q1, q1, q14
vsub.f32 q0, q0, q14
vmul.f32 q1, q1, q15
vmul.f32 q0, q0, q15
vsub.f32 q2, q2, q14
vsub.f32 q3, q3, q14
vst1.32 {q0, q1}, [r1]!
vmul.f32 q3, q3, q15
vmul.f32 q2, q2, q15
vst1.32 {q2, q3}, [r1]!
.endm
COMPUTE
vmov.i32 q3, #0
vcvt.f32.u32 q0, q8
vcvt.f32.u32 q1, q9
vcvt.f32.u32 q2, q10
COMPUTE
sub r4, r4, #8
cmp r4, #8
bge blit_c3_rgba_LoopL8
blit_c3_rgba_L1:
cmp r4, #0
beq End
blit_c3_rgba_LoopL1:
vmov.i32 d0, #0
vld1.8 {d0[0]}, [r0]!
vld1.8 {d0[1]}, [r0]!
vld1.8 {d0[2]}, [r0]!
vmovl.u8 q0, d0
vmovl.u16 q0, d0
vcvt.f32.u32 q0, q0
vsub.f32 q0, q0, q14
vmul.f32 q0, q0, q15
vst1.32 {q0}, [r1]!
subs r4, r4, #1
bne blit_c3_rgba_LoopL1
End:
pop {r4, pc}
.text
.align 5
// _TaskBlitC1ToFloatRGBA
// void _TaskBlitC1ToFloatRGBA(const unsigned char* source, float* dest,
//                             const float* mean, const float* normal, size_t count)
// r0: source, r1: dest, r2: mean, r3: normal, [sp]: count
asm_function _TaskBlitC1ToFloatRGBA
push {r4, lr}
ldr r4, [sp, #8]
vld1.32 {d0[0]}, [r2]
vdup.i32 q14, d0[0]
vld1.32 {d1[0]}, [r3]
vdup.i32 q15, d1[0]
blit_c1_rgba_L8:
cmp r4, #8
blt blit_c1_rgba_L1
blit_c1_rgba_LoopL8:
vld1.8 {d0}, [r0]!
vmovl.u8 q0, d0
vmovl.u16 q1, d1
vmovl.u16 q0, d0
vcvt.f32.u32 q1, q1
vcvt.f32.u32 q0, q0
vsub.f32 q1, q1, q14
vsub.f32 q0, q0, q14
vmul.f32 q1, q1, q15
vmul.f32 q0, q0, q15
vdup.32 q8, d0[0]
vdup.32 q9, d0[1]
vdup.32 q10, d1[0]
vst1.32 {q8, q9}, [r1]!
vdup.32 q11, d1[1]
vdup.32 q8, d2[0]
vst1.32 {q10, q11}, [r1]!
vdup.32 q9, d2[1]
vdup.32 q10, d3[0]
vst1.32 {q8, q9}, [r1]!
vdup.32 q11, d3[1]
vst1.32 {q10, q11}, [r1]!
sub r4, r4, #8
cmp r4, #8
bge blit_c1_rgba_LoopL8
blit_c1_rgba_L1:
cmp r4, #0
beq End
blit_c1_rgba_LoopL1:
vld1.8 {d0[0]}, [r0]!
vmovl.u8 q0, d0
vmovl.u16 q0, d0
vdup.32 q0, d0[0]
vcvt.f32.u32 q0, q0
vsub.f32 q0, q0, q14
vmul.f32 q0, q0, q15
vst1.32 {q0}, [r1]!
subs r4, r4, #1
bne blit_c1_rgba_LoopL1
End:
pop {r4, pc}

