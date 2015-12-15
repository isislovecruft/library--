/*
// AEZ v3 AES-NI version. AEZ info: http://www.cs.ucdavis.edu/~rogaway/aez
//
// REQUIREMENTS: - CPU supporting the AES-NI instruction set
//               - 16-byte aligned aez_ctx_t structures
//               - Max 16 byte nonce, 16 byte authenticator, 4095 byte key
//               - Single AD (AEZ spec allows vector AD but this code doesn't)
//               - Max 2^32-1 byte buffers allowed (due to use of unsigned int)
//
// Written by Ted Krovetz (ted@krovetz.net). Last modified 3 October 2014.
//
// This is free and unencumbered software released into the public domain.
//
// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.
//
// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// For more information, please refer to <http://unlicense.org/>
*/
#include <emmintrin.h>
#include <smmintrin.h>
#include <wmmintrin.h>

/* ------------------------------------------------------------------------- */

typedef struct {
    __m128i I;
    __m128i L;
    __m128i J[5];            /* 1J,2J,4J,8J,16J */
    __m128i delta3_cache;
} aez_ctx_t;

/* ------------------------------------------------------------------------- */

#define zero           _mm_setzero_si128()
#define vand(x,y)      _mm_and_si128(x,y)
#define vor(x,y)       _mm_or_si128(x,y)
#define vxor(x,y)      _mm_xor_si128(x,y)
#define vxor3(x,y,z)   _mm_xor_si128(_mm_xor_si128(x,y),z)
#define vxor4(w,x,y,z) _mm_xor_si128(_mm_xor_si128(w,x),_mm_xor_si128(y,z))

static const unsigned char pad[] = {0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
                                    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
                                    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                    0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};

static __m128i zero_pad(__m128i x, unsigned zero_bytes) {
    return vand(x, _mm_loadu_si128((__m128i*)(pad + zero_bytes)));
}

static __m128i one_zero_pad(__m128i x, unsigned one_zero_bytes) {
    __m128i *p = (__m128i*)(pad + one_zero_bytes);
    return vor(vand(x, _mm_loadu_si128(p)), _mm_loadu_si128(p+1));
}

static __m128i bswap16(__m128i b) {
    return _mm_shuffle_epi8(
        b,_mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
    );
}

static __m128i double_block(__m128i bl) {
    const __m128i mask = _mm_set_epi32(135,1,1,1);
    __m128i tmp = _mm_srai_epi32(bl, 31);
    tmp = _mm_and_si128(tmp, mask);
    tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2,1,0,3));
    bl = _mm_slli_epi32(bl, 1);
    return _mm_xor_si128(bl,tmp);
}

static __m128i aes(__m128i *key, __m128i in) {
    in = _mm_aesenc_si128 (in,key[0]);
    in = _mm_aesenc_si128 (in,key[1]);
    in = _mm_aesenc_si128 (in,key[2]);
    in = _mm_aesenc_si128 (in,key[0]);
    in = _mm_aesenc_si128 (in,key[1]);
    in = _mm_aesenc_si128 (in,key[2]);
    in = _mm_aesenc_si128 (in,key[0]);
    in = _mm_aesenc_si128 (in,key[1]);
    in = _mm_aesenc_si128 (in,key[2]);
    return _mm_aesenc_si128 (in,key[0]);
}

static __m128i aes4(__m128i in, __m128i a, __m128i b, __m128i c, __m128i d) {
    in = _mm_aesenc_si128(in,a);
    in = _mm_aesenc_si128(in,b);
    in = _mm_aesenc_si128(in,c);
    return _mm_aesenc_si128 (in,d);
}

/* loads/stores:
   loadu(p): always do unaligned load.
   load(p): may do aligned load if base pointers are guaranteed to be aligned
   load_partial(p): if aligned do block load, else do byte for-loop
*/
static __m128i loadu(const void *p) { return _mm_loadu_si128((__m128i*)p); }
static __m128i load(const void *p) { return _mm_loadu_si128((__m128i*)p); }
static void storeu(const void *p, __m128i x) {_mm_storeu_si128((__m128i*)p,x);}
static void store(const void *p, __m128i x) {_mm_storeu_si128((__m128i*)p,x);}
static __m128i load_partial(const void *p, unsigned n) {
    if ((unsigned long)p % 16 == 0) return _mm_loadu_si128((__m128i*)p);
    else {
        __m128i tmp; unsigned i;
        for (i=0; i<n; i++) ((char*)&tmp)[i] = ((char*)p)[i];
        return tmp;
    }
}

/* ------------------------------------------------------------------------- */

/* kbytes MUST be fewer than 4096 bytes */
void aez_setup(unsigned char *key, unsigned kbytes, aez_ctx_t *ctx) {
    __m128i tmp;
    if (kbytes==16) {
        __m128i K   = load(key);
        __m128i C11 = _mm_setr_epi8(0xCB,0xEC,0x5B,0xC6,0xB0,0x2F,0xFA,0xA8,
                                    0xA5,0x0D,0x52,0x99,0xA9,0x94,0xA2,0x0A);
        __m128i C12 = _mm_setr_epi8(0x0B,0x97,0x9B,0xB6,0x0A,0x61,0x7C,0x2C,
                                    0xBB,0x65,0x2B,0x68,0x7D,0x12,0xED,0x8D);
        __m128i C13 = _mm_setr_epi8(0x1D,0x8B,0x1E,0x93,0xA6,0x94,0x06,0x4D,
                                    0x4A,0xC9,0x92,0xAF,0xDE,0x78,0x67,0x0F);
        ctx->I      = aes4(vxor(K,C11),C11,C11,C11,C11);
        ctx->J[0]   = aes4(vxor(K,C12),C12,C12,C12,C12);
        ctx->L      = aes4(vxor(K,C13),C13,C13,C13,C13);
    } else {
        __m128i Z  = _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        __m128i i1 = _mm_insert_epi8(zero, 1, 7);
        __m128i i2 = _mm_insert_epi8(zero, 2, 7);
        __m128i i3 = _mm_insert_epi8(zero, 3, 7);
        __m128i j, one = _mm_insert_epi8(zero, 1, 15);
        ctx->I = ctx->J[0] = ctx->L = zero;
        if (kbytes==0) key = (unsigned char *)ctx->J; /* used as a flag later */
        for (j=one; kbytes>=16; key+=16, kbytes-=16, j = _mm_add_epi8(j,one)) {
            __m128i K   = load(key);
            __m128i C1j = aes4(vxor3(i1,j,Z),Z,Z,Z,Z);
            __m128i C2j = aes4(vxor3(i2,j,Z),Z,Z,Z,Z);
            __m128i C3j = aes4(vxor3(i3,j,Z),Z,Z,Z,Z);
            ctx->I      = vxor(ctx->I,    aes4(vxor(K,C1j),C1j,C1j,C1j,C1j));
            ctx->J[0]   = vxor(ctx->J[0], aes4(vxor(K,C2j),C2j,C2j,C2j,C2j));
            ctx->L      = vxor(ctx->L,    aes4(vxor(K,C3j),C3j,C3j,C3j,C3j));
        }
        if (kbytes || (key == (unsigned char *)ctx->J)) {
            __m128i K = one_zero_pad(load_partial(key,kbytes),16-kbytes);
            __m128i C10 = aes4(vxor(i1,Z),Z,Z,Z,Z);
            __m128i C20 = aes4(vxor(i2,Z),Z,Z,Z,Z);
            __m128i C30 = aes4(vxor(i3,Z),Z,Z,Z,Z);
            ctx->I      = vxor(ctx->I,    aes4(vxor(K,C10),C10,C10,C10,C10));
            ctx->J[0]   = vxor(ctx->J[0], aes4(vxor(K,C20),C20,C20,C20,C20));
            ctx->L      = vxor(ctx->L,    aes4(vxor(K,C30),C30,C30,C30,C30));
        }
    }

    /* Fill ctx */
    ctx->J[1] = bswap16(tmp = double_block(bswap16(ctx->J[0])));
    ctx->J[2] = bswap16(tmp = double_block(tmp));
    ctx->J[3] = bswap16(tmp = double_block(tmp));
    ctx->J[4] = bswap16(tmp = double_block(tmp));
    ctx->delta3_cache = zero;
}

/* ------------------------------------------------------------------------- */

/* !! Warning !! Only handles nbytes <= 16 and abytes <= 16 */
__m128i aez_hash(aez_ctx_t *ctx, char *n, unsigned nbytes, char *ad,
                     unsigned adbytes, unsigned abytes) {
    __m128i o0, o1, o2, o3, o4, o5, o6, o7, sum, offset, tmp1, tmp2;
    __m128i I=ctx->I, L=ctx->L, J=ctx->J[0], J8=ctx->J[3], J16=ctx->J[4];
    __m128i J24 = vxor(J16,J8);

    /* Process abytes and nonce */
    tmp1 = vxor4(_mm_insert_epi8(zero,(int)(8*abytes),15), J, L, J8);
    tmp2 = vxor(one_zero_pad(load_partial(n,nbytes),16-nbytes), J16);
    if (nbytes==16) tmp2 = vxor3(tmp2, J, L);
    sum = aes4(tmp1,I,J,L,zero);
    sum = vxor(sum, aes4(tmp2,I,J,L,zero));

    if (adbytes==0) ctx->delta3_cache = aes4(vxor(loadu(pad+32),J24),I,J,L,zero);
    else if (ad) {
        __m128i delta3 = zero, Lfordoubling = bswap16(L);
        offset = vxor(L, J24);
        while (adbytes >= 8*16) {
            o0 = offset;
            o1 = vxor(o0,ctx->J[0]);
            o2 = vxor(o0,ctx->J[1]);
            o3 = vxor(o1,ctx->J[1]);
            o4 = vxor(o0,ctx->J[2]);
            o5 = vxor(o1,ctx->J[2]);
            o6 = vxor(o2,ctx->J[2]);
            o7 = vxor(o3,ctx->J[2]);
            offset = vxor(bswap16(Lfordoubling = double_block(Lfordoubling)),J24);
            delta3 = vxor(delta3, aes4(vxor(load(ad+  0),o1), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 16),o2), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 32),o3), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 48),o4), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 64),o5), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 80),o6), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 96),o7), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+112),o0), I, J, L, zero));
            adbytes-=8*16; ad+=8*16;
        }
        if (adbytes >= 4*16) {
            o1 = vxor(offset,ctx->J[0]);
            o2 = vxor(offset,ctx->J[1]);
            o3 = vxor(o1,ctx->J[1]);
            o4 = offset = vxor(offset,ctx->J[2]);
            delta3 = vxor(delta3, aes4(vxor(load(ad+  0),o1), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 16),o2), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 32),o3), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 48),o4), I, J, L, zero));
            adbytes-=4*16; ad+=4*16;
        }
        if (adbytes >= 2*16) {
            o1 = vxor(offset,ctx->J[0]);
            o2 = offset = vxor(offset,ctx->J[1]);
            delta3 = vxor(delta3, aes4(vxor(load(ad+  0),o1), I, J, L, zero));
            delta3 = vxor(delta3, aes4(vxor(load(ad+ 16),o2), I, J, L, zero));
            adbytes-=2*16; ad+=2*16;
        }
        if (adbytes >= 1*16) {
            o1 = offset = vxor(offset,ctx->J[0]);
            delta3 = vxor(delta3, aes4(vxor(load(ad+  0),o1), I, J, L, zero));
            adbytes-=1*16; ad+=1*16;
        }
        if (adbytes) {
            __m128i tmp = one_zero_pad(load(ad),16-adbytes);
            delta3 = aes4(vxor(tmp,J24), I, J, L, delta3);
        }
        ctx->delta3_cache = delta3;
    }
    return vxor(sum,ctx->delta3_cache);
}

/* ------------------------------------------------------------------------- */

__m128i pass_one(aez_ctx_t *ctx, __m128i *src, unsigned bytes, __m128i *dst) {
    __m128i o0, o1, o2, o3, o4, o5, o6, o7, sum=zero, offset;
    __m128i I=ctx->I, L=ctx->L, J=ctx->J[0];
    __m128i Lfordoubling = bswap16(L), tmp;
    offset = L;
    while (bytes >= 16*16) {
        o0 = offset;
        o1 = vxor(o0,ctx->J[0]);
        o2 = vxor(o0,ctx->J[1]);
        o3 = vxor(o1,ctx->J[1]);
        o4 = vxor(o0,ctx->J[2]);
        o5 = vxor(o1,ctx->J[2]);
        o6 = vxor(o2,ctx->J[2]);
        o7 = vxor(o3,ctx->J[2]);
        offset = bswap16(Lfordoubling = double_block(Lfordoubling));
        store(dst+ 0, aes4(vxor(load(src + 1),o1), J, L, I, load(src+ 0)));
        store(dst+ 2, aes4(vxor(load(src + 3),o2), J, L, I, load(src+ 2)));
        store(dst+ 4, aes4(vxor(load(src + 5),o3), J, L, I, load(src+ 4)));
        store(dst+ 6, aes4(vxor(load(src + 7),o4), J, L, I, load(src+ 6)));
        store(dst+ 8, aes4(vxor(load(src + 9),o5), J, L, I, load(src+ 8)));
        store(dst+10, aes4(vxor(load(src +11),o6), J, L, I, load(src+10)));
        store(dst+12, aes4(vxor(load(src +13),o7), J, L, I, load(src+12)));
        store(dst+14, aes4(vxor(load(src +15),o0), J, L, I, load(src+14)));
        tmp=aes4(load(dst+ 0),I,J,L,load(src+ 1));store(dst+ 1,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+ 2),I,J,L,load(src+ 3));store(dst+ 3,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+ 4),I,J,L,load(src+ 5));store(dst+ 5,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+ 6),I,J,L,load(src+ 7));store(dst+ 7,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+ 8),I,J,L,load(src+ 9));store(dst+ 9,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+10),I,J,L,load(src+11));store(dst+11,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+12),I,J,L,load(src+13));store(dst+13,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+14),I,J,L,load(src+15));store(dst+15,tmp);sum=vxor(sum,tmp);
        bytes -= 16*16; dst += 16; src += 16;
    }
    if (bytes >= 8*16) {
        o1 = vxor(offset,ctx->J[0]);
        o2 = vxor(offset,ctx->J[1]);
        o3 = vxor(o1,ctx->J[1]);
        o4 = offset = vxor(offset,ctx->J[2]);
        store(dst+ 0, aes4(vxor(load(src + 1),o1), J, L, I, load(src+ 0)));
        store(dst+ 2, aes4(vxor(load(src + 3),o2), J, L, I, load(src+ 2)));
        store(dst+ 4, aes4(vxor(load(src + 5),o3), J, L, I, load(src+ 4)));
        store(dst+ 6, aes4(vxor(load(src + 7),o4), J, L, I, load(src+ 6)));
        tmp=aes4(load(dst+ 0),I,J,L,load(src+ 1));store(dst+ 1,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+ 2),I,J,L,load(src+ 3));store(dst+ 3,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+ 4),I,J,L,load(src+ 5));store(dst+ 5,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+ 6),I,J,L,load(src+ 7));store(dst+ 7,tmp);sum=vxor(sum,tmp);
        bytes -= 8*16; dst += 8; src += 8;
    }
    if (bytes >= 4*16) {
        o1 = vxor(offset,ctx->J[0]);
        o2 = offset = vxor(offset,ctx->J[1]);
        store(dst+ 0, aes4(vxor(load(src + 1),o1), J, L, I, load(src+ 0)));
        store(dst+ 2, aes4(vxor(load(src + 3),o2), J, L, I, load(src+ 2)));
        tmp=aes4(load(dst+ 0),I,J,L,load(src+ 1));store(dst+ 1,tmp);sum=vxor(sum,tmp);
        tmp=aes4(load(dst+ 2),I,J,L,load(src+ 3));store(dst+ 3,tmp);sum=vxor(sum,tmp);
        bytes -= 4*16; dst += 4; src += 4;
    }
    if (bytes) {
        o1 = vxor(offset,ctx->J[0]);
        store(dst+ 0, aes4(vxor(load(src + 1),o1), J, L, I, load(src+ 0)));
        tmp=aes4(load(dst+ 0),I,J,L,load(src+ 1));store(dst+ 1,tmp);sum=vxor(sum,tmp);
    }
    return sum;
}

/* ------------------------------------------------------------------------- */

__m128i pass_two(aez_ctx_t *ctx, __m128i s,
                                 __m128i *src, unsigned bytes, __m128i *dst) {
    __m128i o0, o1, o2, o3, o4, o5, o6, o7, sum=zero, offset;
    __m128i fs[8], tmp[8];
    __m128i I=ctx->I, L=ctx->L, J=ctx->J[0];
    __m128i Lfordoubling = bswap16(L);
    offset = L;
    while (bytes >= 16*16) {
        o0 = offset;
        o1 = vxor(o0,ctx->J[0]);
        o2 = vxor(o0,ctx->J[1]);
        o3 = vxor(o1,ctx->J[1]);
        o4 = vxor(o0,ctx->J[2]);
        o5 = vxor(o1,ctx->J[2]);
        o6 = vxor(o2,ctx->J[2]);
        o7 = vxor(o3,ctx->J[2]);
        offset = bswap16(Lfordoubling = double_block(Lfordoubling));
        fs[0] = aes4(vxor(s,o1),L,I,J,I); fs[1] = aes4(vxor(s,o2),L,I,J,I);
        fs[2] = aes4(vxor(s,o3),L,I,J,I); fs[3] = aes4(vxor(s,o4),L,I,J,I);
        fs[4] = aes4(vxor(s,o5),L,I,J,I); fs[5] = aes4(vxor(s,o6),L,I,J,I);
        fs[6] = aes4(vxor(s,o7),L,I,J,I); fs[7] = aes4(vxor(s,o0),L,I,J,I);
        tmp[0] = vxor(load(dst+ 0),fs[0]); sum = vxor(sum,tmp[0]); 
        store(dst+ 0,vxor(load(dst+ 1),fs[0]));
        tmp[1] = vxor(load(dst+ 2),fs[1]); sum = vxor(sum,tmp[1]); 
        store(dst+ 2,vxor(load(dst+ 3),fs[1]));
        tmp[2] = vxor(load(dst+ 4),fs[2]); sum = vxor(sum,tmp[2]); 
        store(dst+ 4,vxor(load(dst+ 5),fs[2]));
        tmp[3] = vxor(load(dst+ 6),fs[3]); sum = vxor(sum,tmp[3]); 
        store(dst+ 6,vxor(load(dst+ 7),fs[3]));
        tmp[4] = vxor(load(dst+ 8),fs[4]); sum = vxor(sum,tmp[4]); 
        store(dst+ 8,vxor(load(dst+ 9),fs[4]));
        tmp[5] = vxor(load(dst+10),fs[5]); sum = vxor(sum,tmp[5]); 
        store(dst+10,vxor(load(dst+11),fs[5]));
        tmp[6] = vxor(load(dst+12),fs[6]); sum = vxor(sum,tmp[6]); 
        store(dst+12,vxor(load(dst+13),fs[6]));
        tmp[7] = vxor(load(dst+14),fs[7]); sum = vxor(sum,tmp[7]); 
        store(dst+14,vxor(load(dst+15),fs[7]));
        store(dst+ 1, aes4(load(dst+ 0), I, J, L, tmp[0]));
        store(dst+ 3, aes4(load(dst+ 2), I, J, L, tmp[1]));
        store(dst+ 5, aes4(load(dst+ 4), I, J, L, tmp[2]));
        store(dst+ 7, aes4(load(dst+ 6), I, J, L, tmp[3]));
        store(dst+ 9, aes4(load(dst+ 8), I, J, L, tmp[4]));
        store(dst+11, aes4(load(dst+10), I, J, L, tmp[5]));
        store(dst+13, aes4(load(dst+12), I, J, L, tmp[6]));
        store(dst+15, aes4(load(dst+14), I, J, L, tmp[7]));
        store(dst+ 0, aes4(vxor(load(dst+ 1),o1), J, L, I, load(dst+ 0)));
        store(dst+ 2, aes4(vxor(load(dst+ 3),o2), J, L, I, load(dst+ 2)));
        store(dst+ 4, aes4(vxor(load(dst+ 5),o3), J, L, I, load(dst+ 4)));
        store(dst+ 6, aes4(vxor(load(dst+ 7),o4), J, L, I, load(dst+ 6)));
        store(dst+ 8, aes4(vxor(load(dst+ 9),o5), J, L, I, load(dst+ 8)));
        store(dst+10, aes4(vxor(load(dst+11),o6), J, L, I, load(dst+10)));
        store(dst+12, aes4(vxor(load(dst+13),o7), J, L, I, load(dst+12)));
        store(dst+14, aes4(vxor(load(dst+15),o0), J, L, I, load(dst+14)));
        bytes -= 16*16; dst += 16; src += 16;
    }
    if (bytes >= 8*16) {
        o1 = vxor(offset,ctx->J[0]);
        o2 = vxor(offset,ctx->J[1]);
        o3 = vxor(o1,ctx->J[1]);
        o4 = offset = vxor(offset,ctx->J[2]);
        fs[0] = aes4(vxor(s,o1),L,I,J,I); fs[1] = aes4(vxor(s,o2),L,I,J,I);
        fs[2] = aes4(vxor(s,o3),L,I,J,I); fs[3] = aes4(vxor(s,o4),L,I,J,I);
        tmp[0] = vxor(load(dst+ 0),fs[0]); sum = vxor(sum,tmp[0]); 
        store(dst+ 0,vxor(load(dst+ 1),fs[0]));
        tmp[1] = vxor(load(dst+ 2),fs[1]); sum = vxor(sum,tmp[1]); 
        store(dst+ 2,vxor(load(dst+ 3),fs[1]));
        tmp[2] = vxor(load(dst+ 4),fs[2]); sum = vxor(sum,tmp[2]); 
        store(dst+ 4,vxor(load(dst+ 5),fs[2]));
        tmp[3] = vxor(load(dst+ 6),fs[3]); sum = vxor(sum,tmp[3]); 
        store(dst+ 6,vxor(load(dst+ 7),fs[3]));
        store(dst+ 1, aes4(load(dst+ 0), I, J, L, tmp[0]));
        store(dst+ 3, aes4(load(dst+ 2), I, J, L, tmp[1]));
        store(dst+ 5, aes4(load(dst+ 4), I, J, L, tmp[2]));
        store(dst+ 7, aes4(load(dst+ 6), I, J, L, tmp[3]));
        store(dst+ 0, aes4(vxor(load(dst+ 1),o1), J, L, I, load(dst+ 0)));
        store(dst+ 2, aes4(vxor(load(dst+ 3),o2), J, L, I, load(dst+ 2)));
        store(dst+ 4, aes4(vxor(load(dst+ 5),o3), J, L, I, load(dst+ 4)));
        store(dst+ 6, aes4(vxor(load(dst+ 7),o4), J, L, I, load(dst+ 6)));
        bytes -= 8*16; dst += 8; src += 8;
    }
    if (bytes >= 4*16) {
        o1 = vxor(offset,ctx->J[0]);
        o2 = offset = vxor(offset,ctx->J[1]);
        fs[0] = aes4(vxor(s,o1),L,I,J,I); fs[1] = aes4(vxor(s,o2),L,I,J,I);
        tmp[0] = vxor(load(dst+ 0),fs[0]); sum = vxor(sum,tmp[0]); 
        store(dst+ 0,vxor(load(dst+ 1),fs[0]));
        tmp[1] = vxor(load(dst+ 2),fs[1]); sum = vxor(sum,tmp[1]); 
        store(dst+ 2,vxor(load(dst+ 3),fs[1]));
        store(dst+ 1, aes4(load(dst+ 0), I, J, L, tmp[0]));
        store(dst+ 3, aes4(load(dst+ 2), I, J, L, tmp[1]));
        store(dst+ 0, aes4(vxor(load(dst+ 1),o1), J, L, I, load(dst+ 0)));
        store(dst+ 2, aes4(vxor(load(dst+ 3),o2), J, L, I, load(dst+ 2)));
        bytes -= 4*16; dst += 4; src += 4;
    }
    if (bytes) {
        o1 = vxor(offset,ctx->J[0]);
        fs[0] = aes4(vxor(s,o1),L,I,J,I);
        tmp[0] = vxor(load(dst+ 0),fs[0]); sum = vxor(sum,tmp[0]); 
        store(dst+ 0,vxor(load(dst+ 1),fs[0]));
        store(dst+ 1, aes4(load(dst+ 0), I, J, L, tmp[0]));
        store(dst+ 0, aes4(vxor(load(dst+ 1),o1), J, L, I, load(dst+ 0)));
    }
    return sum;
}

/* ------------------------------------------------------------------------- */

int cipher_aez_core(aez_ctx_t *ctx, __m128i t, int d, char *src, unsigned bytes,
                    unsigned abytes, char *dst) {
    __m128i s, x, y, frag0, frag1, final0, final1;
    __m128i I=ctx->I, L=ctx->L, J=ctx->J[0], J4=ctx->J[2];
    unsigned i, frag_bytes, initial_bytes;

    if (!d) bytes += abytes;
    frag_bytes = bytes % 32;
    initial_bytes = bytes - frag_bytes - 32;

    /* Compute x and store intermediate results */
    x = pass_one(ctx, (__m128i*)src, initial_bytes, (__m128i*)dst);
    if (frag_bytes >= 16) {
        frag0 = load(src + initial_bytes);
        frag1 = one_zero_pad(load(src + initial_bytes + 16), 32-frag_bytes);
        x  = aes4(vxor(frag0, J4),     I, J, L, x);
        x  = vxor(x, aes4(vxor3(frag1, J4, J), I, J, L, zero));
    } else if (frag_bytes) {
        frag0 = one_zero_pad(load(src + initial_bytes), 16-frag_bytes);
        x = aes4(vxor(frag0, J4),     I, J, L, x);
    }

    /* Calculate s and final block values (y xor'd to final1 later) */
    final0 = vxor3(loadu(src + (bytes - 32)), x, t);
    if (d || !abytes) final1 = loadu(src+(bytes-32)+16);
    else              final1 = zero_pad(loadu(src+(bytes-32)+16), abytes);
    final0 = aes4(vxor(final1, ctx->J[d]), I, J, L, final0);
    final1 = vxor(final1, aes((__m128i*)ctx, vxor(final0, ctx->J[d])));
    s = vxor(final0, final1);
    final0 = vxor(final0, aes((__m128i*)ctx, vxor(final1, ctx->J[d^1])));
    /* Decryption: final0 should hold abytes zero bytes. If not, failure */
    if (d && !_mm_testc_si128(loadu(pad+abytes),final0)) return -1;
    final1 = aes4(vxor(final0, ctx->J[d^1]), I, J, L, final1);

    /* Compute y and store final results */
    y = pass_two(ctx, s, (__m128i*)dst, initial_bytes, (__m128i*)dst);
    if (frag_bytes >= 16) {
        frag0 = vxor(frag0, aes((__m128i*)ctx, vxor(s, J4)));
        frag1 = vxor(frag1, aes((__m128i*)ctx, vxor3(s, J4, J)));
        frag1 = one_zero_pad(frag1, 32-frag_bytes);
        y  = aes4(vxor(frag0, J4),     I, J, L, y);
        y  = vxor(y, aes4(vxor3(frag1, J4, J), I, J, L, zero));
       store(dst + initial_bytes, frag0);
       store(dst + initial_bytes + 16, frag1);
    } else if (frag_bytes) {
        frag0 = vxor(frag0, aes((__m128i*)ctx, vxor(s, J4)));
        frag0 = one_zero_pad(frag0, 16-frag_bytes);
        y = aes4(vxor(frag0, J4), I, J, L, y);
        store(dst + initial_bytes, frag0);
    }

    storeu(dst + (bytes - 32), vxor3(final1, y, t));
    if (!d || !abytes)
        storeu(dst + (bytes - 32) + 16, final0);
    else {
        for (i=0; i<16-abytes; i++)
            ((char*)dst + (bytes - 16))[i] = ((char*)&final0)[i];
    }
    return 0;
}

/* ------------------------------------------------------------------------- */

int cipher_aez_tiny(aez_ctx_t *ctx, __m128i t, int d, char *src, unsigned bytes,
                    unsigned abytes, char *dst) {
    __m128i l, r, tmp, one, rcon, buf[2], mask_10, mask_ff;
    __m128i I=ctx->I, L=ctx->L, J=ctx->J[0], t_orig = t;
    unsigned rnds, i;

    /* load src into buf, zero pad, update bytes for abytes */
    if (bytes >= 16) {
        buf[0] = load(src);
        buf[1] = zero_pad(load_partial(src+16,bytes-16),32-bytes);
    } else {
        buf[0] = zero_pad(load_partial(src,bytes),16-bytes);
        buf[1] = zero;
    }
    if (!d) bytes += abytes;

    /* load l/r, create 10* padding masks, shift r 4 bits if odd length */
    l = buf[0];
    r = loadu((char*)buf+bytes/2);
    mask_ff = loadu(pad+16-bytes/2);
    mask_10 = loadu(pad+32-bytes/2);
    if (bytes&1) {  /* Odd length. Deal with nibbles. */
        mask_10 = _mm_srli_epi32(mask_10,4);
        ((char*)&mask_ff)[bytes/2] = 0xf0;
        r = bswap16(r);
        r = vor(_mm_slli_epi64(r, 4), _mm_srli_epi64(_mm_slli_si128(r, 8), 60));
        r = bswap16(r);
    }
    r = vor(vand(r, mask_ff), mask_10);

    /* Add tweak offset into t, and determine the number of rounds */
    if (bytes >= 16) {
        t = vxor3(t, ctx->J[1], ctx->J[2]);             /* (0,6) offset */
        rnds = 8;
    } else {
        t = vxor4(t, ctx->J[0], ctx->J[1], ctx->J[2]);  /* (0,7) offset */
        if (bytes>=3) rnds = 10; else if (bytes==2) rnds = 16; else rnds = 24;
    }

    if (!d) {
        one = _mm_insert_epi8(zero,1,15);
        rcon = zero;
    } else {
        one = _mm_insert_epi8(zero,-1,15);
        rcon = _mm_insert_epi8(zero,rnds-1,15);
    }

    if ((d) && (bytes < 16)) {
        tmp = vor(l, loadu(pad+32));
        tmp = aes4(vxor4(tmp,t_orig,ctx->J[0],ctx->J[1]), I, J, L, zero);
        tmp = vand(tmp, loadu(pad+32));
        l = vxor(l, tmp);
    }

    /* Feistel */
    for (i=0; i<rnds; i+=2) {
        l = vor(vand(aes4(vxor3(t,r,rcon), I, J, L, l), mask_ff), mask_10);
        rcon = _mm_add_epi8(rcon,one);
        r = vor(vand(aes4(vxor3(t,l,rcon), I, J, L, r), mask_ff), mask_10);
        rcon = _mm_add_epi8(rcon,one);
    }
    buf[0] = r;
    if (bytes&1) {
        l = bswap16(l);
        l = vor(_mm_srli_epi64(l, 4), _mm_slli_epi64(_mm_srli_si128(l, 8), 60));
        l = bswap16(l);
        r = vand(loadu((char*)buf+bytes/2), _mm_insert_epi8(zero,0xf0,0));
        l = vor(l, r);
    }
    storeu((char*)buf+bytes/2, l);
    if (d) {
        bytes -= abytes;
        if (abytes==16) tmp = loadu((char*)buf+bytes);
        else {
            tmp = zero;
            for (i=0; i<abytes; i++) ((char*)&tmp)[i] = ((char*)buf+bytes)[i];
        }
        if (!_mm_testz_si128(tmp,tmp)) return -1;
    } else if (bytes < 16) {
        tmp = vor(zero_pad(buf[0], 16-bytes), loadu(pad+32));
        tmp = aes4(vxor4(tmp,t_orig,ctx->J[0],ctx->J[1]), I, J, L, zero);
        buf[0] = vxor(buf[0], vand(tmp, loadu(pad+32)));
    }
    for (i=0; i<bytes; i++) dst[i] = ((char*)buf)[i];
    return 0;
}

/* ------------------------------------------------------------------------- */

void aez_encrypt(aez_ctx_t *ctx, char *n, unsigned nbytes,
                 char *ad, unsigned adbytes, unsigned abytes,
                 char *src, unsigned bytes, char *dst) {

    __m128i t = aez_hash(ctx, n, nbytes, ad, adbytes, abytes);
    if (bytes==0) {
        unsigned i;
        t = aes((__m128i*)ctx, vxor3(t, ctx->J[0], ctx->J[1]));
        for (i=0; i<abytes; i++) dst[i] = ((char*)&t)[i];
    } else if (bytes+abytes < 32)
        cipher_aez_tiny(ctx, t, 0, src, bytes, abytes, dst);
    else
        cipher_aez_core(ctx, t, 0, src, bytes, abytes, dst);
}

/* ------------------------------------------------------------------------- */

int aez_decrypt(aez_ctx_t *ctx, char *n, unsigned nbytes,
                 char *ad, unsigned adbytes, unsigned abytes,
                 char *src, unsigned bytes, char *dst) {

    __m128i t;
    if (bytes < abytes) return -1;
    t = aez_hash(ctx, n, nbytes, ad, adbytes, abytes);
    if (bytes==abytes) {
        __m128i claimed = zero_pad(load_partial(src,abytes), 16-abytes);
        t = zero_pad(aes((__m128i*)ctx, vxor3(t, ctx->J[0], ctx->J[1])), 16-abytes);
        return _mm_testc_si128(t, claimed) - 1;
    } else if (bytes < 32) {
        return cipher_aez_tiny(ctx, t, 1, src, bytes, abytes, dst);
    } else {
        return cipher_aez_core(ctx, t, 1, src, bytes, abytes, dst);
    }
}

/* ------------------------------------------------------------------------- */
/* aez mapping for CAESAR competition                                        */

int crypto_aead_encrypt(
    unsigned char *c,unsigned long long *clen,
    const unsigned char *m,unsigned long long mlen,
    const unsigned char *ad,unsigned long long adlen,
    const unsigned char *nsec,
    const unsigned char *npub,
    const unsigned char *k
)
{
    aez_ctx_t ctx;
    (void)nsec;
    if (clen) *clen = mlen+16;
    aez_setup((unsigned char *)k, 16, &ctx);
    aez_encrypt(&ctx, (char *)npub, 12, (char *)ad, (unsigned)adlen, 16,
                (char *)m, (unsigned)mlen, (char *)c);
    return 0;
}

int crypto_aead_decrypt(
    unsigned char *m,unsigned long long *mlen,
    unsigned char *nsec,
    const unsigned char *c,unsigned long long clen,
    const unsigned char *ad,unsigned long long adlen,
    const unsigned char *npub,
    const unsigned char *k
)
{
    aez_ctx_t ctx;
    (void)nsec;
    if (mlen) *mlen = clen-16;
    aez_setup((unsigned char *)k, 16, &ctx);
    return aez_decrypt(&ctx, (char *)npub, 12, (char *)ad, (unsigned)adlen, 16,
                        (char *)c, (unsigned)clen, (char *)m);
}
