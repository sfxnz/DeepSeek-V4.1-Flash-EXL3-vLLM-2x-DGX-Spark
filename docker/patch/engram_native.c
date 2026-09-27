/* Engram disk-row gather + fp8 e4m3 x ue8m0 -> bf16 dequant (EngramDiskStager).
 *
 * One call replaces, for one table and one decode step, the per-table Python
 * work of the stock stage (engram_stage_fast._fast_stage_one + the gather v2
 * DiskEngramTable path): hash ids -> owned file rows -> pread the 256 B fp8
 * row and the 8 B ue8m0 scale row -> f32(fp8) * 2^(e - 127) -> bf16 into the
 * pinned [n, local_heads, dim] staging buffer; rows this rank does not own are
 * zero. It is called through ctypes, so the GIL is released for the whole call
 * (the stock path hands the GIL back and forth ~200 times per step).
 *
 * I/O: a first pass reads every owned row with preadv2(RWF_NOWAIT), which
 * only succeeds from the page cache. Rows that miss get one
 * posix_fadvise(WILLNEED) each (their reads are then in flight together) and
 * a blocking pread afterwards. Without RWF_NOWAIT support every row is a
 * plain blocking pread. The bytes read are the same either way.
 *
 * Numerics: lut[256] is torch's float8_e4m3fn -> float32 table, built by the
 * caller in the serving process; the product is one IEEE f32 multiply; the
 * bf16 rounding matches torch's CPU float -> bfloat16 conversion (see
 * f32_to_bf16_rne). The caller verifies the output against the stock path
 * bit for bit before trusting it.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>

#define ENG_ABI_VERSION 1

enum {
    ENG_ST_ROWS = 0,     /* rows in the call: n * local_heads */
    ENG_ST_OWNED = 1,    /* rows read from the file */
    ENG_ST_MISS = 2,     /* owned rows not fully in the page cache */
    ENG_ST_SYSCALLS = 3, /* preadv2 + pread + fadvise calls */
    ENG_ST_NOWAIT = 4,   /* 1 when RWF_NOWAIT was used, 0 when it fell back */
    ENG_NSTATS = 5,
};

int eng_abi_version(void) { return ENG_ABI_VERSION; }

static inline uint32_t f32_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

static inline float bits_f32(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* Round to nearest even, as torch's CPU float -> bfloat16 conversion does on
 * this aarch64 build. NaN keeps its sign and top payload bits and is made
 * quiet (BFCVT semantics); torch turns fp8 0x7F/0xFF (f32 0x7FF00000 /
 * 0xFFF00000) into bf16 0x7FF0 / 0xFFF0, not c10's scalar 0x7FC0. */
static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u = f32_bits(f);
    if (f != f) return (uint16_t)((u >> 16) | 0x0040u);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* Exported for the exhaustive numerics check. */
uint16_t eng_dequant_one(const float *lut, uint8_t w, uint8_t s) {
    return f32_to_bf16_rne(lut[w] * bits_f32((uint32_t)s << 23));
}

static int read_full(int fd, uint8_t *buf, size_t len, int64_t off, int64_t *calls) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = pread(fd, buf + got, len - got, (off_t)(off + (int64_t)got));
        *calls += 1;
        if (n < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (n == 0) return -EIO; /* past EOF: the table geometry is wrong */
        got += (size_t)n;
    }
    return 0;
}

/* 1: served whole from the page cache; 0: not (retry blocking); <0: -errno. */
static int read_nowait(int fd, uint8_t *buf, size_t len, int64_t off, int64_t *calls) {
    struct iovec iov = {buf, len};
    ssize_t n = preadv2(fd, &iov, 1, (off_t)off, RWF_NOWAIT);
    *calls += 1;
    if (n == (ssize_t)len) return 1;
    if (n >= 0) return 0;
    if (errno == EAGAIN || errno == EINTR) return 0;
    return -errno;
}

static void willneed(int fd, int64_t off, int64_t len, int64_t *calls) {
    int64_t a = off & ~(int64_t)4095;
    int64_t b = (off + len + 4095) & ~(int64_t)4095;
    posix_fadvise(fd, (off_t)a, (off_t)(b - a), POSIX_FADV_WILLNEED);
    *calls += 1;
}

/*
 * hash:  this table's first id, [n_tok rows] x [heads_valid ids], row stride
 *        tok_stride int32 elements (the pinned hash_host view).
 * out:   [n_tok, local_heads, dim] bf16 bits. Heads >= heads_valid are
 *        padding (the stock path pads the ids with -1) and come out zero.
 * lut:   256 f32, torch float8_e4m3fn -> float32.
 * sw/ss: scratch, n_tok * local_heads * dim and * sb bytes.
 * flags: bit 0 = try RWF_NOWAIT first.
 * stats: ENG_NSTATS int64 counters (overwritten).
 * Returns 0, or -errno (the caller then disarms and uses the stock path).
 */
int eng_gather_bf16(int w_fd, int64_t w_off, int s_fd, int64_t s_off, int dim, int sb,
                    int64_t vocab_start, int64_t vocab_end, const int32_t *hash,
                    int64_t tok_stride, int n_tok, int heads_valid, int local_heads,
                    const float *lut, uint16_t *out, uint8_t *sw, uint8_t *ss, int flags,
                    int64_t *stats) {
    int64_t st[ENG_NSTATS] = {0};
    if (dim <= 0 || sb <= 0 || dim % sb != 0 || n_tok < 0 || local_heads <= 0 ||
        heads_valid < 0 || heads_valid > local_heads)
        return -EINVAL;
    const int64_t rows = (int64_t)n_tok * local_heads;
    const int qb = dim / sb;
    int nowait = flags & 1;
    uint8_t *state = NULL; /* 0 unowned, 1 read done, 2 read pending */
    int rc = 0;
    st[ENG_ST_ROWS] = rows;
    if (rows == 0) goto done;
    state = (uint8_t *)malloc((size_t)rows);
    if (!state) return -ENOMEM;

    for (int64_t r = 0; r < rows; r++) {
        const int t = (int)(r / local_heads), h = (int)(r % local_heads);
        const int64_t id = h < heads_valid ? (int64_t)hash[(int64_t)t * tok_stride + h] : -1;
        if (id < vocab_start || id >= vocab_end) {
            state[r] = 0;
            continue;
        }
        st[ENG_ST_OWNED] += 1;
        state[r] = 2;
        if (!nowait) continue;
        int a = read_nowait(w_fd, sw + r * dim, (size_t)dim, w_off + id * dim, &st[ENG_ST_SYSCALLS]);
        int b = a < 0 ? a : read_nowait(s_fd, ss + r * sb, (size_t)sb, s_off + id * sb, &st[ENG_ST_SYSCALLS]);
        if (a == 1 && b == 1) {
            state[r] = 1;
        } else if (a < 0 || b < 0) {
            int e = a < 0 ? a : b;
            if (e == -EOPNOTSUPP || e == -ENOSYS || e == -EINVAL) {
                nowait = 0; /* no RWF_NOWAIT here: every remaining row blocks */
            } else {
                rc = e;
                goto done;
            }
        }
    }
    if (nowait) {
        for (int64_t r = 0; r < rows; r++) {
            if (state[r] != 2) continue;
            const int t = (int)(r / local_heads), h = (int)(r % local_heads);
            const int64_t id = (int64_t)hash[(int64_t)t * tok_stride + h];
            st[ENG_ST_MISS] += 1;
            willneed(w_fd, w_off + id * dim, dim, &st[ENG_ST_SYSCALLS]);
            willneed(s_fd, s_off + id * sb, sb, &st[ENG_ST_SYSCALLS]);
        }
    }
    for (int64_t r = 0; r < rows; r++) {
        if (state[r] != 2) continue;
        const int t = (int)(r / local_heads), h = (int)(r % local_heads);
        const int64_t id = (int64_t)hash[(int64_t)t * tok_stride + h];
        rc = read_full(w_fd, sw + r * dim, (size_t)dim, w_off + id * dim, &st[ENG_ST_SYSCALLS]);
        if (rc == 0)
            rc = read_full(s_fd, ss + r * sb, (size_t)sb, s_off + id * sb, &st[ENG_ST_SYSCALLS]);
        if (rc) goto done;
        state[r] = 1;
    }
    for (int64_t r = 0; r < rows; r++) {
        uint16_t *o = out + r * dim;
        if (!state[r]) {
            memset(o, 0, (size_t)dim * sizeof *o);
            continue;
        }
        const uint8_t *w = sw + r * dim, *s = ss + r * sb;
        for (int b = 0; b < sb; b++) {
            const float scale = bits_f32((uint32_t)s[b] << 23);
            for (int j = 0; j < qb; j++) o[b * qb + j] = f32_to_bf16_rne(lut[w[b * qb + j]] * scale);
        }
    }
done:
    st[ENG_ST_NOWAIT] = nowait;
    free(state);
    if (stats) memcpy(stats, st, sizeof st);
    return rc;
}
