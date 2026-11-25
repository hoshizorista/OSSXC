#pragma once
#include <cstdint>
#include <cstddef>

// ============================================================================
// Lossless Warp-Synchronous Ballot-Interleaved rANS Format for MXFP4 Experts
// 100.000% mathematically lossless (canonicalizes -0.0 nibble 8 -> +0.0 nibble 0)
// ============================================================================
static constexpr uint32_t RANS_MAGIC        = 0x52414E36u; // "RAN6"
static constexpr uint32_t RANS_PROB_BITS    = 12u;
static constexpr uint32_t RANS_PROB_SCALE   = 1u << RANS_PROB_BITS; // 4096
static constexpr uint32_t RANS_PROB_MASK    = RANS_PROB_SCALE - 1u; // 4095
static constexpr uint32_t RANS_L            = 1u << 16;             // 65536

// Blocks: 8,294,400 (GU) + 4,147,200 (DN) = 12,441,600 bytes = 270 warps x 32 lanes x 360 uint32 words
static constexpr int RANS_BLK_WARPS         = 270;
static constexpr int RANS_BLK_GU_WARPS      = 180;
static constexpr int RANS_BLK_WORDS_PER_LANE = 360; // 360 uint32 = 1440 bytes = 2880 nibbles per lane
static constexpr int RANS_BLK_SYMS          = 15;   // 15 canonicalized FP4 symbols (no -0.0)

// Scales: 518,400 (GU: 5760x90) + 259,200 (DN: 2880x90) = 777,600 bytes = 30 warps x 32 lanes x 9 rows x 90 scales
static constexpr int RANS_SCL_WARPS         = 30;
static constexpr int RANS_SCL_GU_WARPS      = 20;
static constexpr int RANS_SCL_ROWS_PER_LANE = 9;
static constexpr int RANS_SCL_ROW_LEN       = 90;
static constexpr int RANS_SCL_SYMS          = 32;   // 0..30 = delta in [-15..+15], 31 = escape raw uint8

// Mapping 15 canonical symbols <-> original 4-bit FP4 nibbles (skipping 8 = -0.0)
static constexpr uint8_t kSym15ToNib[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 0
};
static constexpr uint8_t kNibToSym15[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14
};

#pragma pack(push, 1)
struct RansExpertHeader {
    uint32_t magic;
    uint32_t total_bytes;       // total compressed expert bytes (aligned to 16B)
    uint32_t blk_stream_off;    // byte offset to uint16_t blk stream data
    uint32_t blk_stream_u16s;   // total uint16_t count across all 270 block warps
    uint32_t scl_stream_off;    // byte offset to uint16_t scl stream data
    uint32_t scl_stream_u16s;   // total uint16_t count across all 30 scale warps
    uint32_t scl_esc_off;       // byte offset to uint8_t escape raw scales
    uint32_t scl_esc_bytes;     // total escape byte count across all 30 scale warps
    uint8_t  scl_base;          // mode scale for k==0 delta prediction
    uint8_t  pad[3];

    uint16_t blk_freq[16];      // [0..14] sum to 4096, [15] = 0
    uint16_t scl_freq[32];      // [0..31] sum to 4096

    uint32_t blk_warp_u16_off[RANS_BLK_WARPS + 1];  // 271 uint32
    uint32_t blk_init_state[RANS_BLK_WARPS][32];    // 270 x 32 uint32

    uint32_t scl_warp_u16_off[RANS_SCL_WARPS + 1];  // 31 uint32
    uint32_t scl_warp_esc_off[RANS_SCL_WARPS + 1];  // 31 uint32
    uint32_t scl_init_state[RANS_SCL_WARPS][32];    // 30 x 32 uint32
};
#pragma pack(pop)
