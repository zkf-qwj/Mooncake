#include <gtest/gtest.h>

#include <array>
#include "rdma_buffer.h"

namespace mooncake {
namespace {

TEST(RdmaBuffer, AllocateCopyAndFree) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        GTEST_SKIP() << "No GPU is available";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    void* pointer = nullptr;
    constexpr size_t kBytes = 4096;
    ASSERT_EQ(allocateRdmaBuffer(&pointer, kBytes), cudaSuccess);
    ASSERT_NE(pointer, nullptr);
    cudaPointerAttributes attributes{};
    EXPECT_EQ(cudaPointerGetAttributes(&attributes, pointer), cudaSuccess);
#ifdef USE_MACA
    EXPECT_EQ(attributes.type, mcMemoryTypeUnified);
#else
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
#endif
    std::array<unsigned char, kBytes> input{};
    std::array<unsigned char, kBytes> output{};
    input.fill(0x5a);
    EXPECT_EQ(cudaMemcpy(pointer, input.data(), kBytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(output.data(), pointer, kBytes, cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(input, output);
    EXPECT_EQ(cudaFree(pointer), cudaSuccess);
}

}  // namespace
}  // namespace mooncake
