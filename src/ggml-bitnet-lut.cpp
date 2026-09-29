#include <vector>
#include <type_traits>
#include <algorithm>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "ggml-cpu-impl.h"

#if defined(GGML_BITNET_ARM_TL1) || defined(GGML_BITNET_X86_TL2)
#include "bitnet-lut-kernels.h"
#endif

#if defined(GGML_BITNET_ARM_TL1)

void ggml_bitnet_init(void) {
    if (initialized) {
        return;
    }
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_TL1) {
        return false;
    } else {
        return true;
    }
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        if (src1->ne[1] <= 1) {
            return true;
        }
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne00 = src0->ne[0];

    // QLUT needs 16 bytes per K element (see lut_ctor<K> in bitnet-lut-kernels.h),
    // plus one bitnet_float_type for the per-activation LUT scale.
    size_t wsize = ne00 * 16 * sizeof(int8_t) + sizeof(bitnet_float_type);
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

void ggml_bitnet_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int m = (int)src0->ne[1];
    const int k = (int)src0->ne[0];

    GGML_ASSERT(src1->ne[1] <= 1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    if (ith == 0) {
        ggml_bitnet_transform_tensor(const_cast<struct ggml_tensor *>(src0));
    }
    if (nth > 1) {
        ggml_barrier(params->threadpool);
    }

    struct bitnet_tensor_extra * extra = (struct bitnet_tensor_extra *)src0->extra;
    GGML_ASSERT(extra != nullptr);

    char * wdata = (char *)params->wdata;
    int8_t * qlut = (int8_t *)wdata;
    bitnet_float_type * lut_scales = (bitnet_float_type *)(qlut + (size_t)k * 16);

    if (ith == 0) {
        ggml_preprocessor(m, k, src1->data, lut_scales, qlut);
    }

    // barrier
    if (nth > 1) {
        ggml_barrier(params->threadpool);
    }

    // qgemm_lut_{M}_{K} only computes one BM-sized tile of the M output rows per
    // call (BM is baked into the generated kernel, e.g. 256 for M=2560) - loop
    // over all n_tile_num = M/BM tiles, offsetting the weight/output pointers,
    // and split the tiles across threads.
    const int n_tile_num = extra->n_tile_num;
    GGML_ASSERT(m % n_tile_num == 0);
    const int w_size = k * m / 4;
    const int w_tile_size = w_size / n_tile_num;
    const int c_tile_size = m / n_tile_num;

    const int th_tile_num = (n_tile_num + nth - 1) / nth;
    const int th_tile_beg = ith * th_tile_num;
    const int th_tile_end = std::min((ith + 1) * th_tile_num, n_tile_num);

    for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
        const int w_offset = i_tile * w_tile_size;
        const int dst_offset = i_tile * c_tile_size;
        ggml_qgemm_lut(m, k, extra->qweights + w_offset, qlut, extra->scales, lut_scales,
                        (bitnet_float_type *)dst->data + dst_offset);
    }
}

#endif
#if defined(GGML_BITNET_X86_TL2)
void ggml_bitnet_init(void) {
    if (initialized) {
        return;
    }
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        return true;
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];

    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

// bitnet_float_type is `float` on x86 (see ggml-bitnet.h), so unlike the ARM_TL1
// path there is no fp16 scratch buffer / conversion step needed here: activations
// and outputs are used in place.
void ggml_bitnet_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int ne00 = (int)src0->ne[0];
    const int ne01 = (int)src0->ne[1];
    const int ne10 = (int)src1->ne[0];
    const int ne11 = (int)src1->ne[1];
    const int ne0  = (int)dst->ne[0];

    const int ith = params->ith;
    const int nth = params->nth;

    struct bitnet_tensor_extra * wt = (struct bitnet_tensor_extra *)src0->extra;
    GGML_ASSERT(wt != nullptr);

    char * wdata = (char *)params->wdata;

    const int total_k = ne10;
    const int three_k = (total_k / wt->BK) * wt->BK;
    const int two_k   = total_k - three_k;

    int8_t * three_qlut = (int8_t *)wdata;
    bitnet_float_type * lut_scales = (bitnet_float_type *)(three_qlut + three_k / 3 * 16 * 2 * ne11);
    int8_t * two_qlut = (int8_t *)(lut_scales + ne11);

    if (ith == 0) {
        // Although we have done this in file `llama.cpp`,
        // we still need to do it here for non-model inference, e.g., test-backend-ops.cpp.
        ggml_bitnet_transform_tensor(const_cast<struct ggml_tensor *>(src0));
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        ggml_preprocessor(ne11, ne01, three_k, two_k, src1->data, lut_scales, three_qlut, two_qlut);
    }

    ggml_barrier(params->threadpool);

    float * act_output = (float *)dst->data;

    const int n_tile_num = wt->n_tile_num;
    GGML_ASSERT(ne0 % n_tile_num == 0);
    const int w_size        = three_k * ne01 / (2 * 3);
    const int w_tile_size   = w_size / n_tile_num;
    const int c_size        = ne01;
    const int c_tile_size   = c_size / n_tile_num;
    const int sign_size     = three_k * ne01 / 24;
    const int sign_tile_size = sign_size / n_tile_num;

    const int th_tile_num = (n_tile_num + nth - 1) / nth;
    const int th_tile_beg = ith * th_tile_num;
    const int th_tile_end = std::min((ith + 1) * th_tile_num, n_tile_num);

    uint8_t * sign = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2;
    const int two_w_size      = ne01 * two_k / (2 * 2);
    const int two_w_tile_size = two_w_size / n_tile_num;
    // The "two" region starts right after the sign region (weight_bytes + sign_bytes),
    // i.e. 5*sign_size, NOT three_k*ne01/4 -- that matched an older packer layout that
    // reserved extra alignment padding this one (preprocess_weights_tl2) doesn't add.
    uint8_t * two_A = sign + sign_size;

    // The generated qgemm_lut kernels write batch row b's tile output at
    // C[b*c_tile_size + i] (tile-local, BM-strided), but dst->data needs row
    // (done+b) at act_output[(done+b)*ne01 + tile_col_offset + i] (ne01-strided,
    // since c_tile_size == ne01 only when n_tile_num == 1). Route through a
    // per-tile scratch buffer and scatter it into place with the right stride.
    std::vector<float> tile_scratch((size_t)512 * c_tile_size);

    auto do_bs_batch = [&](int bs, int qlut_off_elems_three, int qlut_off_elems_two,
                           int lut_scales_off, int done) {
        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int w_offset       = i_tile * w_tile_size;
            const int sign_offset    = i_tile * sign_tile_size;
            const int two_w_offset   = i_tile * two_w_tile_size;
            const int tile_col_offset = i_tile * c_tile_size;

            ggml_qgemm_lut(bs, ne01, ne00, three_k, ((uint8_t *)(wt->qweights) + w_offset),
                           sign + sign_offset,
                           three_qlut + qlut_off_elems_three,
                           wt->scales,
                           lut_scales + lut_scales_off,
                           tile_scratch.data());
            ggml_qgemm_lut(bs, ne01, ne00, two_k, two_A + two_w_offset,
                           NULL,
                           two_qlut + qlut_off_elems_two,
                           wt->scales,
                           lut_scales + lut_scales_off,
                           tile_scratch.data());

            for (int b = 0; b < bs; b++) {
                memcpy(act_output + (size_t)(done + b) * ne01 + tile_col_offset,
                       tile_scratch.data() + (size_t)b * c_tile_size,
                       c_tile_size * sizeof(float));
            }
        }
    };

    if (ne11 > 1) {
        int iter = ne11;
        const int bs512_num = iter / 512; iter -= 512 * bs512_num;
        const int bs256_num = iter / 256; iter -= 256 * bs256_num;
        const int bs128_num = iter / 128; iter -= 128 * bs128_num;
        const int bs32_num  = iter / 32;  iter -= 32  * bs32_num;
        const int bs8_num   = iter / 8;   iter -= 8   * bs8_num;
        const int bs1_num   = iter / 1;

        int done = 0; // rows of ne11 processed so far
        for (int i = 0; i < bs512_num; i++, done += 512) {
            do_bs_batch(512, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs256_num; i++, done += 256) {
            do_bs_batch(256, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs128_num; i++, done += 128) {
            do_bs_batch(128, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs32_num; i++, done += 32) {
            do_bs_batch(32, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs8_num; i++, done += 8) {
            do_bs_batch(8, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
        for (int i = 0; i < bs1_num; i++, done += 1) {
            do_bs_batch(1, done * three_k / 3 * 32, done * two_k / 2 * 32, done, done);
        }
    } else {
        do_bs_batch(1, 0, 0, 0, 0);
    }
}

#endif
