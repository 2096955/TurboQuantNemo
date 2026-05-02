#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstdio>

int main() {
    auto* pool = NS::AutoreleasePool::alloc()->init();

    auto* default_device = MTL::CreateSystemDefaultDevice();
    auto* devices = MTL::CopyAllDevices();
    const unsigned long count = devices ? static_cast<unsigned long>(devices->count()) : 0UL;

    std::printf("default_device=%s\n",
                default_device ? default_device->name()->utf8String() : "null");
    std::printf("device_count=%lu\n", count);

    for (unsigned long i = 0; i < count; ++i) {
        auto* device = static_cast<MTL::Device*>(devices->object(i));
        std::printf(
            "device[%lu]=%s unified=%d low_power=%d removable=%d\n",
            i,
            device->name()->utf8String(),
            static_cast<int>(device->hasUnifiedMemory()),
            static_cast<int>(device->isLowPower()),
            static_cast<int>(device->isRemovable())
        );

        auto* queue = device->newCommandQueue();
        std::printf("  command_queue=%s\n", queue ? "ok" : "null");
        if (queue) {
            queue->release();
        }
    }

    if (devices) {
        devices->release();
    }
    if (default_device) {
        default_device->release();
    }
    pool->release();
    return 0;
}
