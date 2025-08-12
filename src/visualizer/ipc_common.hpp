#pragma once
#include <cstdint>
#include <atomic>

namespace ipc {
constexpr uint32_t kMagic   = 0xBA7C4F1u;
constexpr uint32_t kVersion = 1;

struct alignas(64) BatchHeader {
  uint32_t magic   = kMagic;
  uint32_t version = kVersion;
  uint32_t N = 0, C = 0, H = 0, W = 0;
  uint64_t ts_ns = 0;
  std::atomic<uint64_t> seq{0};  // even=stable, odd=writing
  uint64_t valid_mask = 0;       // bit i set => camera i present this batch
};
} // namespace ipc