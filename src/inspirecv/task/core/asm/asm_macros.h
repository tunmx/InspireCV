#ifndef ASM_MACROS_H
#define ASM_MACROS_H
.macro asm_function fname
#ifdef __APPLE__
.globl _\fname
_\fname:
#else
.global \fname
#ifdef __ELF__
.hidden \fname
.type \fname, %function
#endif
\fname:
#endif
.endm


#endif /* ASM_MACROS_H */


