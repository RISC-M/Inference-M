.section .text.init
.global _start
_start:
    /* Set up stack */
    li sp, 0x80080000
    call main

    /* Exit via HTIF */
    la t0, tohost
    li t1, 1
    sw t1, 0(t0)
1:  j 1b

.section .data
.align 6
.global tohost
tohost: .dword 0
.global fromhost
fromhost: .dword 0
