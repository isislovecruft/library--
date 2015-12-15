/*
// AEZ v3 reference code. AEZ info: http://www.cs.ucdavis.edu/~rogaway/aez
//
// ** This version is slow and susceptible to side-channel attacks. **
// ** Do not use for any purpose other than to understand AEZ.      **
//
// Written by Ted Krovetz (ted@krovetz.net). Last modified 25 September 2014.
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
#include <stdlib.h>
#include <string.h>

typedef unsigned char byte;

/* Use Rijmen, Bosselaers and Barreto's reference code.
   INTERMEDIATE_VALUE_KAT must be defined in its header for AES4/AES10
   functionality. Also, it defines the "u32" type used for AES internal keys.
*/
#include "rijndael-alg-fst.h"

/* ------------------------------------------------------------------------- */

static void write32_big_endian(unsigned x, void *ptr) {
    byte *p = (byte *)ptr;
    p[0] = (byte)(x>>24); p[1] = (byte)(x>>16);
    p[2] = (byte)(x>> 8); p[3] = (byte)(x>> 0);
}

/* ------------------------------------------------------------------------- */

/* Adjust our constructed round keys to be compatible with rijndael-alg-fst  */
static void correct_key(u32 *p, unsigned nbytes) {
    unsigned i;
    for (i=0; i<nbytes/4; i++) write32_big_endian(p[i], p+i);
}

/* ------------------------------------------------------------------------- */

static void xor_bytes(byte *src1, byte *src2, unsigned n, byte *dst) {
    while (n) { n--; dst[n] = src1[n] ^ src2[n]; }
}

/* ------------------------------------------------------------------------- */

static void double_block(byte *p) {
    byte i, tmp = p[0];
    for (i=0; i<15; i++)
        p[i] = (p[i] << 1) | (p[i+1] >> 7);
    p[15] = (p[15] << 1) ^ ((tmp >> 7) * 135);
}

/* ------------------------------------------------------------------------- */

static void mult_block(unsigned x, byte *src, byte *dst) {
    byte t[16], r[16];
    memcpy(t,src,16); memset(r,0,16);
    for ( ; x; x>>=1) {
        if (x&1) xor_bytes(r,t,16,r);
        double_block(t);
    }
    memcpy(dst,r,16);
}

/* ------------------------------------------------------------------------- */

static void Extract(byte *K, unsigned kbytes, byte extracted_key[3*16]) {
    unsigned i,j,k,empty;
    byte buf[16];
    u32 aes4_key_z[4*5],aes4_key_c[4*5];

    empty = (kbytes==0);
    memset(extracted_key,0,48);
    for (i=0;i<16;i++) ((byte *)aes4_key_z)[i]=(byte)i;
    correct_key(aes4_key_z, 16);
    for (i=1;i<5;i++) memcpy((byte*)aes4_key_z+16*i,aes4_key_z,16);

    for (j=1; kbytes >= 16; K+=16, kbytes-=16, j++) {
        for (i=1; i<=3; i++) {
            /* aes4 ([i]_64 [j]_64) */
            memset(aes4_key_c,0,12); ((byte *)aes4_key_c)[7]=(byte)i;
            write32_big_endian(j, (byte *)aes4_key_c+12);
            rijndaelEncryptRound(aes4_key_z, 10, (byte *)aes4_key_c, 4);
            /* Set aes4 key to derived key, aes4 K_j, accumulate result */
            correct_key(aes4_key_c, 16);
            for (k=1; k<5; k++) memcpy((byte*)aes4_key_c+16*k, aes4_key_c, 16);
            memcpy(buf, K, 16);
            rijndaelEncryptRound(aes4_key_c, 10, buf, 4);
            xor_bytes(extracted_key+(i-1)*16, buf, 16, extracted_key+(i-1)*16);
        }
    }
    if (kbytes || empty) {  /* If key remains, or kbytes was initially zero */
        for (i=1; i<=3; i++) {
            /* aes4 ([i]_64 [0]_64) */
            memset(aes4_key_c,0,16); ((byte *)aes4_key_c)[7]=(byte)i;
            rijndaelEncryptRound(aes4_key_z, 10, (byte *)aes4_key_c, 4);
            /* Set aes4 key to derived key, aes4 K_m10*, accumulate result */
            correct_key(aes4_key_c, 16);
            for (k=1; k<5; k++) memcpy((byte*)aes4_key_c+16*k, aes4_key_c, 16);
            memset(buf,0,16); memcpy(buf,K,kbytes); buf[kbytes]=0x80;
            rijndaelEncryptRound(aes4_key_c, 10, buf, 4);
            xor_bytes(extracted_key+(i-1)*16, buf, 16, extracted_key+(i-1)*16);
        }
    }
}

/* ------------------------------------------------------------------------- */

static void E(byte *K, unsigned kbytes, int i, unsigned j,
                                                byte src[16], byte dst[16]) {
    byte k, extracted_key[3*16], iJ[16], jJ[16], buf[16], I[16], J[16], L[16];

    Extract(K, kbytes, extracted_key);
    memcpy(I,extracted_key,16);
    memcpy(J,extracted_key+16,16);
    memcpy(L,extracted_key+32,16);

    /* Encipher */
    if (i < 0) {
        u32 aes_key[4*11];
        memset(aes_key,0,16);                                  /* 0        */
        memcpy((byte*)aes_key+ 16, I, 16);                     /* I        */
        memcpy((byte*)aes_key+ 32, L, 16);                     /* L        */
        memcpy((byte*)aes_key+ 48, J, 16);                     /* J        */
        correct_key(aes_key,4*16);
        memcpy((byte*)aes_key+ 64, (byte*)aes_key+16, 48);     /* I L J    */
        memcpy((byte*)aes_key+112, (byte*)aes_key+16, 48);     /* I L J    */
        memcpy((byte*)aes_key+160, (byte*)aes_key+16, 16);     /* I        */
        mult_block(j,J,jJ); xor_bytes(jJ,src,16,buf);
        rijndaelEncryptRound(aes_key, 11, buf, 10); /* incl final MixColumns */
    } else {
        u32 aes4_key[4*5];
        unsigned first = (i>=3?0:i);  /* Which key comes first in aes4_key */
        /* Build AES4 key in use */
        memset((byte *)aes4_key, 0, 16);
        for (k=0; k<3; k++)
            memcpy((byte*)aes4_key+16+k*16, extracted_key+(k+first)%3*16, 16);
        if (i==2) memcpy((byte*)aes4_key+64, I, 16);
        else      memset((byte*)aes4_key+64, 0, 16);
        correct_key(aes4_key,5*16);
        /* Build encryption buffer from src and various offsets used */
        memcpy(buf, src, 16);
        if (i == 0) {
            mult_block(j,J,jJ); xor_bytes(buf,jJ,16,buf);
        } else if (i==1 || i==2) {
            mult_block(j%8,J,jJ); xor_bytes(buf,jJ,16,buf);
            for ( ; j>8; j-=8) mult_block(2,L,L);  /* L = 2^((j-1)/8) L */
            xor_bytes(buf, L, 16, buf);
        } else /* i >= 3 */ {
            mult_block((i-2)*8,J,iJ); xor_bytes(buf,iJ,16,buf);
            if (j>0) {
                mult_block(j%8,J,jJ); xor_bytes(buf,jJ,16,buf);
                for ( ; j>8; j-=8) mult_block(2,L,L);  /* L = 2^((j-1)/8) L */
                xor_bytes(buf, L, 16, buf);
            }
        }
        rijndaelEncryptRound(aes4_key, 10, buf, 4);
    }
    memcpy(dst, buf, 16);
}

/* ------------------------------------------------------------------------- */

static void AEZhash(byte *K, unsigned kbytes, byte *N, unsigned nbytes,
    byte *A[], unsigned abytes[], unsigned veclen, unsigned tau, byte *result) {

    byte buf[16], sum[16], *p;
    unsigned i, j, bytes, empty;

    /* Initialize sum with hash of tau */
    memset(buf,0,12); write32_big_endian(tau, buf+12);
    E(K,kbytes,3,1,buf,sum);

    /* Hash nonce, accumulate into sum */
    empty = (nbytes==0);
    for (j=1; nbytes>=16; j++, nbytes-=16, N+=16) {
        E(K,kbytes,4,j,N,buf); xor_bytes(sum, buf, 16, sum);
    }
    if (nbytes || empty) {
        memset(buf,0,16); memcpy(buf,N,nbytes); buf[nbytes]=0x80;
        E(K,kbytes,4,0,buf,buf);
        xor_bytes(sum, buf, 16, sum);
    }

    /* Hash each vector element, accumulate into sum */
    for (i=0; i<veclen; i++) {
        p = A[i]; bytes = abytes[i]; empty = (bytes==0);
        for (j=1; bytes>=16; j++, bytes-=16, p+=16) {
            E(K,kbytes,5+i,j,p,buf); xor_bytes(sum, buf, 16, sum);
        }
        if (bytes || empty) {
            memset(buf,0,16); memcpy(buf,p,bytes); buf[bytes]=0x80;
            E(K,kbytes,5+i,0,buf,buf);
            xor_bytes(sum, buf, 16, sum);
        }
    }
    memcpy(result,sum,16);
}

/* ------------------------------------------------------------------------- */

static void AEZprf(byte *K, unsigned kbytes, byte delta[16], unsigned bytes, byte *result) {

    byte buf[16], ctr[16];
    memset(ctr,0,16);
    for ( ; bytes >= 16; bytes-=16, result+=16) {
        unsigned i=15;
        xor_bytes(delta, ctr, 16, buf);
        E(K,kbytes,-1,3,buf,result);
        do { ctr[i]++; i--; } while (ctr[i+1]==0);   /* ctr+=1 */
    }
    if (bytes) {
        xor_bytes(delta, ctr, 16, buf);
        E(K,kbytes,-1,3,buf,buf);
        memcpy(result, buf, bytes);
    }
}

/* ------------------------------------------------------------------------- */

/* Set d=0 for EncipherAEZcore and d=1 for DecipherAEZcore */
static void AEZcore(byte *K, unsigned kbytes, byte delta[16],
                        byte *in, unsigned inbytes, unsigned d, byte *out) {
    byte tmp[16], X[16], Y[16], S[16];
    byte *in_orig = in, *out_orig = out;
    unsigned j, inbytes_orig = inbytes;

    memset(X,0,16); memset(Y,0,16);

    /* Pass 1 over in[0:-32], store intermediate values in out[0:-32] */
    for (j=1; inbytes >= 64; j++, inbytes-=32, in+=32, out+=32) {
        E(K, kbytes, 1, j, in+16, tmp); xor_bytes(in, tmp, 16, out);
        E(K, kbytes, 0, 0, out, tmp); xor_bytes(in+16, tmp, 16, out+16);
        xor_bytes(out+16, X, 16, X);
    }

    /* Finish X calculation */
    inbytes -= 32;                /* inbytes now has fragment length 0..31 */
    if (inbytes >= 16) {
        E(K, kbytes, 0, 4, in, tmp); xor_bytes(X, tmp, 16, X);
        inbytes -= 16; in += 16; out += 16;
        memset(tmp,0,16); memcpy(tmp,in,inbytes); tmp[inbytes] = 0x80;
        E(K, kbytes, 0, 5, tmp, tmp); xor_bytes(X, tmp, 16, X);
    } else if (inbytes > 0) {
        memset(tmp,0,16); memcpy(tmp,in,inbytes); tmp[inbytes] = 0x80;
        E(K, kbytes, 0, 4, tmp, tmp); xor_bytes(X, tmp, 16, X);
    }
    in += inbytes; out += inbytes;

    /* Calculate S */
    E(K, kbytes, 0, 1+d, in+16, tmp);
    xor_bytes(X, in, 16, out);
    xor_bytes(delta, out, 16, out);
    xor_bytes(tmp, out, 16, out);
    E(K, kbytes, -1, 1+d, out, tmp);
    xor_bytes(in+16, tmp, 16, out+16);
    xor_bytes(out, out+16, 16, S);

    /* Pass 2 over intermediate values in out[32..]. Final values written */
    inbytes = inbytes_orig; out = out_orig; in = in_orig;
    for (j=1; inbytes >= 64; j++, inbytes-=32, in+=32, out+=32) {
        E(K, kbytes, 2, j, S, tmp);
        xor_bytes(out, tmp, 16, out); xor_bytes(out+16, tmp, 16, out+16);
        xor_bytes(out, Y, 16, Y);
        E(K, kbytes, 0, 0, out+16, tmp); xor_bytes(out, tmp, 16, out);
        E(K, kbytes, 1, j, out, tmp); xor_bytes(out+16, tmp, 16, out+16);
        memcpy(tmp, out, 16); memcpy(out, out+16, 16); memcpy(out+16, tmp, 16);
    }

    /* Finish Y calculation and finish encryption of fragment bytes */
    inbytes -= 32;                /* inbytes now has fragment length 0..31 */
    if (inbytes >= 16) {
        E(K, kbytes, -1, 4, S, tmp); xor_bytes(in, tmp, 16, out);
        E(K, kbytes, 0, 4, out, tmp); xor_bytes(Y, tmp, 16, Y);
        inbytes -= 16; in += 16; out += 16;
        E(K, kbytes, -1, 5, S, tmp); xor_bytes(in, tmp, inbytes, tmp);
        memcpy(out,tmp,inbytes);
        memset(tmp+inbytes,0,16-inbytes); tmp[inbytes] = 0x80;
        E(K, kbytes, 0, 5, tmp, tmp); xor_bytes(Y, tmp, 16, Y);
    } else if (inbytes > 0) {
        E(K, kbytes, -1, 4, S, tmp); xor_bytes(in, tmp, inbytes, tmp);
        memcpy(out,tmp,inbytes);
        memset(tmp+inbytes,0,16-inbytes); tmp[inbytes] = 0x80;
        E(K, kbytes, 0, 4, tmp, tmp); xor_bytes(Y, tmp, 16, Y);
    }
    in += inbytes; out += inbytes;

    /* Finish encryption of last two blocks */
    E(K, kbytes, -1, 2-d, out+16, tmp);
    xor_bytes(out, tmp, 16, out);
    E(K, kbytes, 0, 2-d, out, tmp);
    xor_bytes(tmp, out+16, 16, out+16);
    xor_bytes(delta, out+16, 16, out+16);
    xor_bytes(Y, out+16, 16, out+16);
    memcpy(tmp, out, 16); memcpy(out, out+16, 16); memcpy(out+16, tmp, 16);
}

/* ------------------------------------------------------------------------- */

/* Set d=0 for EncipherAEZtiny and d=1 for DecipherAEZtiny */
static void AEZtiny(byte *K, unsigned kbytes, byte delta[16],
                        byte *in, unsigned inbytes, unsigned d, byte *out) {
    unsigned rounds,i,j=7,k;
    int step;
    byte mask=0x00, pad=0x80, L[16], R[16], buf[32];
    if      (inbytes==1) rounds=24;
    else if (inbytes==2) rounds=16;
    else if (inbytes<16) rounds=10;
    else {          j=6; rounds=8; }
    /* Split (inbytes*8)/2 bits into L and R. Beware: May end in nibble. */
    memcpy(L, in,           (inbytes+1)/2);
    memcpy(R, in+inbytes/2, (inbytes+1)/2);
    if (inbytes&1) {                     /* Must shift R left by half a byte */
        for (i=0; i<inbytes/2; i++)
            R[i] = (byte)((R[i] << 4) | (R[i+1] >> 4));
        R[inbytes/2] = (byte)(R[inbytes/2] << 4);
        pad = 0x08; mask = 0xf0;
    }
    if (d) {
        if (inbytes < 16) {
            memset(buf,0,16); memcpy(buf,in,inbytes); buf[0] |= 0x80;
            xor_bytes(delta, buf, 16, buf);
            E(K, kbytes,0,3,buf,buf);
            L[0] ^= (buf[0] & 0x80);
        }
        i = rounds-1; step = -1;
    } else {
        i = 0; step = 1;
    }
    for (k=0; k<rounds/2; k++,i=(unsigned)((int)i+2*step)) {
        memset(buf, 0, 16);
        memcpy(buf,R,(inbytes+1)/2);
        buf[inbytes/2] = (buf[inbytes/2] & mask) | pad;
        xor_bytes(buf, delta, 16, buf);
        buf[15] ^= (byte)i;
        E(K, kbytes,0,j,buf,buf);
        xor_bytes(L, buf, 16, L);

        memset(buf, 0, 16);
        memcpy(buf,L,(inbytes+1)/2);
        buf[inbytes/2] = (buf[inbytes/2] & mask) | pad;
        xor_bytes(buf, delta, 16, buf);
        buf[15] ^= (byte)((int)i+step);
        E(K, kbytes,0,j,buf,buf);
        xor_bytes(R, buf, 16, R);
    }
    memcpy(buf,           R, inbytes/2);
    memcpy(buf+inbytes/2, L, (inbytes+1)/2);
    if (inbytes&1) {
        for (i=inbytes-1; i>inbytes/2; i--)
            buf[i] = (byte)((buf[i] >> 4) | (buf[i-1] << 4));
        buf[inbytes/2] = (byte)((L[0] >> 4) | (R[inbytes/2] & 0xf0));
    }
    memcpy(out,buf,inbytes);
    if ((inbytes < 16) && !d) {
        memset(buf+inbytes,0,16-inbytes); buf[0] |= 0x80;
        xor_bytes(delta, buf, 16, buf);
        E(K, kbytes,0,3,buf,buf);
        out[0] ^= (buf[0] & 0x80);
    }
}

/* ------------------------------------------------------------------------- */

static void Encipher(byte *K, unsigned kbytes, byte delta[16],
                                    byte *in, unsigned inbytes, byte *out) {
    if (inbytes == 0) return;
    if (inbytes < 32) AEZtiny(K, kbytes, delta, in, inbytes, 0, out);
    else              AEZcore(K, kbytes, delta, in, inbytes, 0, out);
}

/* ------------------------------------------------------------------------- */

static void Decipher(byte *K, unsigned kbytes, byte delta[16],
                                    byte *in, unsigned inbytes, byte *out) {
    if (inbytes == 0) return;
    if (inbytes < 32) AEZtiny(K, kbytes, delta, in, inbytes, 1, out);
    else              AEZcore(K, kbytes, delta, in, inbytes, 1, out);
}

/* ------------------------------------------------------------------------- */

int Decrypt(byte *K, unsigned kbytes,
            byte *N, unsigned nbytes,
            byte *AD[], unsigned adbytes[],
            unsigned veclen, unsigned abytes,
            byte *C, unsigned cbytes, byte *M) {
    byte delta[16], *X, sum=0;
    unsigned i;
    if (cbytes < abytes) return -1;
    AEZhash(K, kbytes, N, nbytes, AD, adbytes, veclen, abytes*8, delta);
    X = (byte *)malloc(cbytes);
    if (cbytes==abytes) {
        AEZprf(K, kbytes, delta, abytes, X);
        for (i=0; i<abytes; i++) sum |= (X[i] ^ C[i]);
    } else {
        Decipher(K, kbytes, delta, C, cbytes, X);
        for (i=0; i<abytes; i++) sum |= X[cbytes-abytes+i];
        if (sum==0) memcpy(M,X,cbytes-abytes);
    }
    free(X);
    return (sum == 0 ? 0 : -1);  /* return 0 if valid, -1 if invalid */
}

/* ------------------------------------------------------------------------- */

void Encrypt(byte *K, unsigned kbytes,
             byte *N, unsigned nbytes,
             byte *AD[], unsigned adbytes[],
             unsigned veclen, unsigned abytes,
             byte *M, unsigned mbytes, byte *C) {
    byte delta[16], *X;
    AEZhash(K, kbytes, N, nbytes, AD, adbytes, veclen, abytes*8, delta);
    if (mbytes==0) {
        AEZprf(K, kbytes, delta, abytes, C);
    } else {
        X = (byte *)malloc(mbytes+abytes);
        memcpy(X, M, mbytes); memset(X+mbytes,0,abytes);
        Encipher(K, kbytes, delta, X, mbytes+abytes, X);
        memcpy(C, X, mbytes+abytes);
        free(X);
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
    byte *AD[] = {(byte*)ad};
    unsigned adbytes[] = {(unsigned)adlen};
    (void)nsec;
    if (clen) *clen = mlen+16;
    Encrypt((byte*)k, 16, (byte*)npub, 12, AD, adbytes, 1, 16, (byte*)m, mlen, (byte*)c);
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
    byte *AD[] = {(byte*)ad};
    unsigned adbytes[] = {(unsigned)adlen};
    (void)nsec;
    if (mlen) *mlen = clen-16;
    return Decrypt((byte*)k, 16, (byte*)npub, 12, AD, adbytes, 1, 16, (byte*)c, clen, (byte*)m);
}

/* ------------------------------------------------------------------------- */
