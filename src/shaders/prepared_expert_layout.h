#ifndef HF2Q_PREPARED_EXPERT_LAYOUT
#define HF2Q_PREPARED_EXPERT_LAYOUT
#include <metal_stdlib>
using namespace metal;

constant constexpr uint M8_TILE_ROWS = 64;
constant constexpr uint M8_Q4 = 0;
constant constexpr uint M8_Q6 = 1;

inline ulong grouped_m8_tiled_index(uint row, uint unit, uint units) {
    return (ulong(row / M8_TILE_ROWS) * units + unit) * M8_TILE_ROWS + row % M8_TILE_ROWS;
}

#endif
