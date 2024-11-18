#ifndef WIGRAPHICSDEVICE_METAL_H
#define WIGRAPHICSDEVICE_METAL_H

#include "CommonInclude.h"
#include "wiPlatform.h"
#include <memory>

#ifdef PLATFORM_MACOS
#define WICKEDENGINE_BUILD_METAL
#endif

#ifdef WICKEDENGINE_BUILD_METAL
#define __OBJC_BOOL_IS_BOOL 1
#include <Metal/Metal.hpp>
#include <Metal/MTLFence.hpp>
#include <Metal/MTLCommandQueue.hpp>
#include <Metal/MTLCommandBuffer.hpp>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <mutex>
#include "wiGraphicsDevice.h"

// namespace MTL {
// class Device;
// class Buffer;
// class Fence;
// class RenderPassDescriptor;
// class Texture;
// class RenderCommandEncoder;
// class ComputeCommandEncoder;
// class RenderPipelineState;
// class CommandBufferDescriptor;
// }

namespace wi::graphics
{
class GraphicsDevice_Metal : public GraphicsDevice
{
    friend struct CommandQueue;
protected:
    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    std::unordered_map<size_t, MTL::Buffer*> resourceCache;
    std::mutex resourceMutex;

    struct Semaphore
    {
        std::shared_ptr<MTL::Fence> fence;
        uint64_t fenceValue = 0;
    };

    struct CommandQueue
    {
        MTL::CommandBufferDescriptor *desc = nullptr;
        MTL::CommandQueue *queue;
        wi::vector<std::shared_ptr<MTL::CommandBuffer>> submit_cmds;

        void signal(const Semaphore& semaphore);
        void wait(const Semaphore& semaphore);
        void submit();
    } queues[QUEUE_COUNT];

    struct CopyAllocator
    {
        GraphicsDevice_Metal* device = nullptr;
        std::shared_ptr<MTL::CommandQueue> queue; // create separate copy queue to reduce interference with main QUEUE_COPY
        std::mutex locker;

        struct CopyCMD
        {
            std::shared_ptr<MTL::CommandBuffer> commandAllocator;
            std::shared_ptr<MTL::Buffer> commandList;
            std::shared_ptr<MTL::Fence> fence;
            uint64_t fenceValueSignaled = 0;
            GPUBuffer uploadbuffer;
            bool IsValid() const;
            bool IsCompleted() const;
        };
        wi::vector<CopyCMD> freelist;

        void init(GraphicsDevice_Metal* device);
        CopyCMD allocate(uint64_t staging_size);
        void submit(CopyCMD cmd);
    };
    mutable CopyAllocator copyAllocator;

    MTL::Fence *frame_fence[BUFFERCOUNT][QUEUE_COUNT];

    struct DescriptorBinder
    {
        DescriptorBindingTable table;
        GraphicsDevice_Metal* device = nullptr;

        const void* optimizer_graphics = nullptr;
        uint64_t dirty_graphics = 0ull; // 1 dirty bit flag per root parameter
        const void* optimizer_compute = nullptr;
        uint64_t dirty_compute = 0ull; // 1 dirty bit flag per root parameter

        void init(GraphicsDevice_Metal* device);
        void reset();
        void flush(bool graphics, CommandList cmd);
    };

    wi::vector<Semaphore> semaphore_pool;
    std::mutex semaphore_pool_locker;
    Semaphore new_semaphore()
    {
        std::scoped_lock lck(semaphore_pool_locker);
        if (semaphore_pool.empty())
        {
            Semaphore& dependency = semaphore_pool.emplace_back();
            //HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, PPV_ARGS(dependency.fence));
            //assert(SUCCEEDED(hr));
        }
        Semaphore semaphore = std::move(semaphore_pool.back());
        semaphore_pool.pop_back();
        semaphore.fenceValue++;
        return semaphore;
    }
    void free_semaphore(const Semaphore& semaphore)
    {
        std::scoped_lock lck(semaphore_pool_locker);
        semaphore_pool.push_back(semaphore);
    }

    struct CommandList_Metal
    {
        MTL::CommandQueue* commandQueue;
        MTL::CommandBuffer* commandBuffer;
        MTL::RenderCommandEncoder* renderEncoder;
        MTL::ComputeCommandEncoder* computeEncoder;

        uint32_t bufferIndex = 0;

        using PipelineStateCache = std::unordered_map<size_t, std::shared_ptr<MTL::RenderPipelineState>>;
        PipelineStateCache pipelineCache;

        // Synchronization objects
        std::vector<Semaphore> waitEvents;
        std::vector<Semaphore> signalEvents;

		GPULinearAllocator frame_allocators[BUFFERCOUNT];

        // Resource management
        struct DiscardResource
        {
            std::shared_ptr<MTL::Texture> resource = nullptr;
        };
        std::vector<DiscardResource> discards;

        // Render pass state
        struct RenderPassInfo
        {
            std::shared_ptr<MTL::RenderPassDescriptor> descriptor;
        };
        RenderPassInfo renderPassInfo;

        std::vector<std::shared_ptr<MTL::Fence>> frameFences;

        void reset(uint32_t bufferIndex);

        // Command Buffer management
        MTL::CommandBuffer *GetCommandBuffer();

        // Render Encoder management
        MTL::RenderCommandEncoder *GetRenderEncoder();

        // Compute Encoder management
        MTL::ComputeCommandEncoder *GetComputeEncoder();

        // Pipeline state retrieval or creation
        //std::shared_ptr<MTL::RenderPipelineState> getPipelineState(size_t pipelineHash, std::shared_ptr<MTLDevice> device, MTLRenderPipelineDescriptor* descriptor);
    };

    constexpr CommandList_Metal& GetCommandList(CommandList cmd) const
    {
        assert(cmd.IsValid());
        return *(CommandList_Metal*)cmd.internal_state;
    }

    void pso_validate(CommandList cmd);

    void predraw(CommandList cmd);
    void predispatch(CommandList cmd);

    static constexpr uint32_t immutable_sampler_slot_begin = 100;

public:
    GraphicsDevice_Metal(wi::platform::window_type window, ValidationMode validationMode_, GPUPreference preference);
    virtual ~GraphicsDevice_Metal();

    // Create a SwapChain. If the SwapChain is to be recreated, the window handle can be nullptr.
    virtual bool CreateSwapChain(const SwapChainDesc* desc, wi::platform::window_type window, SwapChain* swapchain) const override;
    // Create a buffer with a callback to initialize its data. Note: don't read from callback's dest pointer, reads will be very slow! Use memcpy to write to it to make sure only writes happen!
    virtual bool CreateBuffer2(const GPUBufferDesc* desc, const std::function<void(void* dest)>& init_callback, GPUBuffer* buffer, const GPUResource* alias = nullptr, uint64_t alias_offset = 0ull) const override;
    virtual bool CreateTexture(const TextureDesc* desc, const SubresourceData* initial_data, Texture* texture, const GPUResource* alias = nullptr, uint64_t alias_offset = 0ull) const override;
    virtual bool CreateShader(ShaderStage stage, const void* shadercode, size_t shadercode_size, Shader* shader) const override;
    virtual bool CreateSampler(const SamplerDesc* desc, Sampler* sampler) const override;
    virtual bool CreateQueryHeap(const GPUQueryHeapDesc* desc, GPUQueryHeap* queryheap) const override;
    // Creates a graphics pipeline state. If renderpass_info is specified, then it will be only compatible with that renderpass info, but it will be created immediately (it can also take longer to be created)
    virtual bool CreatePipelineState(const PipelineStateDesc* desc, PipelineState* pso, const RenderPassInfo* renderpass_info = nullptr) const override;
    virtual bool CreateRaytracingAccelerationStructure(const RaytracingAccelerationStructureDesc* desc, RaytracingAccelerationStructure* bvh) const { return false; }
    virtual bool CreateRaytracingPipelineState(const RaytracingPipelineStateDesc* desc, RaytracingPipelineState* rtpso) const { return false; }
    virtual bool CreateVideoDecoder(const VideoDesc* desc, VideoDecoder* video_decoder) const { return false; };

    virtual int CreateSubresource(Texture* texture, SubresourceType type, uint32_t firstSlice, uint32_t sliceCount, uint32_t firstMip, uint32_t mipCount, const Format* format_change = nullptr, const ImageAspect* aspect = nullptr, const Swizzle* swizzle = nullptr, float min_lod_clamp = 0) const override;
    virtual int CreateSubresource(GPUBuffer* buffer, SubresourceType type, uint64_t offset, uint64_t size = ~0, const Format* format_change = nullptr, const uint32_t* structuredbuffer_stride_change = nullptr) const override;

    virtual void DeleteSubresources(GPUResource* resource) override;

    virtual int GetDescriptorIndex(const GPUResource* resource, SubresourceType type, int subresource = -1) const override;
    virtual int GetDescriptorIndex(const Sampler* sampler) const override;

    virtual CommandList BeginCommandList(QUEUE_TYPE queue = QUEUE_GRAPHICS) override;

    virtual void SubmitCommandLists() override;

    virtual void WaitForGPU() const override;

    virtual void ClearPipelineStateCache() override;

    virtual size_t GetActivePipelineCount() const override;

    // Get the shader binary format that the underlying graphics API consumes
    virtual ShaderFormat GetShaderFormat() const override;

    // Get a Texture resource that represents the current back buffer of the SwapChain
    virtual Texture GetBackBuffer(const SwapChain* swapchain) const override;
    // Returns the current color space of the swapchain output
    virtual ColorSpace GetSwapChainColorSpace(const SwapChain* swapchain) const override;
    // Returns true if the swapchain could support HDR output regardless of current format
    //	Returns false if the swapchain couldn't support HDR output
    virtual bool IsSwapChainSupportsHDR(const SwapChain* swapchain) const override;

    // Returns the minimum required alignment for buffer offsets when creating subresources
    virtual uint64_t GetMinOffsetAlignment(const GPUBufferDesc* desc) const override;

    // Returns video memory statistics for the current application
    virtual MemoryUsage GetMemoryUsage() const override;

    // Returns the maximum amount of viewports that can be bound at once
    virtual uint32_t GetMaxViewportCount() const override;

    virtual void WaitCommandList(CommandList cmd, CommandList wait_for) override;

    virtual void WaitQueue(CommandList cmd, QUEUE_TYPE wait_for) override;
    virtual void RenderPassBegin(const SwapChain* swapchain, CommandList cmd) override;
    virtual void RenderPassBegin(const RenderPassImage* images, uint32_t image_count, CommandList cmd, RenderPassFlags flags = RenderPassFlags::NONE) override;
    virtual void RenderPassEnd(CommandList cmd) override;
    virtual void BindScissorRects(uint32_t numRects, const Rect* rects, CommandList cmd) override;
    virtual void BindViewports(uint32_t NumViewports, const Viewport* pViewports, CommandList cmd) override;
    virtual void BindResource(const GPUResource* resource, uint32_t slot, CommandList cmd, int subresource = -1) override;
    virtual void BindResources(const GPUResource *const* resources, uint32_t slot, uint32_t count, CommandList cmd) override;
    virtual void BindUAV(const GPUResource* resource, uint32_t slot, CommandList cmd, int subresource = -1) override;
    virtual void BindUAVs(const GPUResource *const* resources, uint32_t slot, uint32_t count, CommandList cmd) override;
    virtual void BindSampler(const Sampler* sampler, uint32_t slot, CommandList cmd) override;
    virtual void BindConstantBuffer(const GPUBuffer* buffer, uint32_t slot, CommandList cmd, uint64_t offset = 0ull) override;
    virtual void BindVertexBuffers(const GPUBuffer *const* vertexBuffers, uint32_t slot, uint32_t count, const uint32_t* strides, const uint64_t* offsets, CommandList cmd) override;
    virtual void BindIndexBuffer(const GPUBuffer* indexBuffer, const IndexBufferFormat format, uint64_t offset, CommandList cmd) override;
    virtual void BindStencilRef(uint32_t value, CommandList cmd) override;
    virtual void BindBlendFactor(float r, float g, float b, float a, CommandList cmd) override;
    virtual void BindShadingRate(ShadingRate rate, CommandList cmd) {}
    virtual void BindPipelineState(const PipelineState* pso, CommandList cmd) override;
    virtual void BindComputeShader(const Shader* cs, CommandList cmd) override;
    virtual void BindDepthBounds(float min_bounds, float max_bounds, CommandList cmd) override;
    virtual void Draw(uint32_t vertexCount, uint32_t startVertexLocation, CommandList cmd) override;
    virtual void DrawIndexed(uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation, CommandList cmd) override;
    virtual void DrawInstanced(uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation, CommandList cmd) override;
    virtual void DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation, CommandList cmd) override;
    virtual void DrawInstancedIndirect(const GPUBuffer* args, uint64_t args_offset, CommandList cmd) override;
    virtual void DrawIndexedInstancedIndirect(const GPUBuffer* args, uint64_t args_offset, CommandList cmd) override;
    virtual void DrawInstancedIndirectCount(const GPUBuffer* args, uint64_t args_offset, const GPUBuffer* count, uint64_t count_offset, uint32_t max_count, CommandList cmd) override;
    virtual void DrawIndexedInstancedIndirectCount(const GPUBuffer* args, uint64_t args_offset, const GPUBuffer* count, uint64_t count_offset, uint32_t max_count, CommandList cmd) override;
    virtual void Dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ, CommandList cmd) override;
    virtual void DispatchIndirect(const GPUBuffer* args, uint64_t args_offset, CommandList cmd) override;
    virtual void DispatchMesh(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ, CommandList cmd) {}
    virtual void DispatchMeshIndirect(const GPUBuffer* args, uint64_t args_offset, CommandList cmd) {}
    virtual void DispatchMeshIndirectCount(const GPUBuffer* args, uint64_t args_offset, const GPUBuffer* count, uint64_t count_offset, uint32_t max_count, CommandList cmd) {}
    virtual void CopyResource(const GPUResource* pDst, const GPUResource* pSrc, CommandList cmd) override;
    virtual void CopyBuffer(const GPUBuffer* pDst, uint64_t dst_offset, const GPUBuffer* pSrc, uint64_t src_offset, uint64_t size, CommandList cmd) override;
    virtual void CopyTexture(const Texture* dst, uint32_t dstX, uint32_t dstY, uint32_t dstZ, uint32_t dstMip, uint32_t dstSlice, const Texture* src, uint32_t srcMip, uint32_t srcSlice, CommandList cmd, const Box* srcbox = nullptr, ImageAspect dst_aspect = ImageAspect::COLOR, ImageAspect src_aspect = ImageAspect::COLOR) override;
    virtual void QueryBegin(const GPUQueryHeap *heap, uint32_t index, CommandList cmd) override;
    virtual void QueryEnd(const GPUQueryHeap *heap, uint32_t index, CommandList cmd) override;
    virtual void QueryResolve(const GPUQueryHeap* heap, uint32_t index, uint32_t count, const GPUBuffer* dest, uint64_t dest_offset, CommandList cmd) override;
    virtual void QueryReset(const GPUQueryHeap* heap, uint32_t index, uint32_t count, CommandList cmd) {}
    virtual void Barrier(const GPUBarrier* barriers, uint32_t numBarriers, CommandList cmd) override;
    virtual void BuildRaytracingAccelerationStructure(const RaytracingAccelerationStructure* dst, CommandList cmd, const RaytracingAccelerationStructure* src = nullptr) {}
    virtual void BindRaytracingPipelineState(const RaytracingPipelineState* rtpso, CommandList cmd) {}
    virtual void DispatchRays(const DispatchRaysDesc* desc, CommandList cmd) {}
    virtual void PushConstants(const void* data, uint32_t size, CommandList cmd, uint32_t offset = 0) override;
    virtual void PredicationBegin(const GPUBuffer* buffer, uint64_t offset, PredicationOp op, CommandList cmd) {}
    virtual void PredicationEnd(CommandList cmd) {}
    virtual void ClearUAV(const GPUResource* resource, uint32_t value, CommandList cmd) override;
    virtual void VideoDecode(const VideoDecoder* video_decoder, const VideoDecodeOperation* op, CommandList cmd) {}

    virtual void EventBegin(const char* name, CommandList cmd) override;
    virtual void EventEnd(CommandList cmd) override;
    virtual void SetMarker(const char* name, CommandList cmd) override;

    virtual RenderPassInfo GetRenderPassInfo(CommandList cmd) override;

    virtual GPULinearAllocator& GetFrameAllocator(CommandList cmd) override;
};
}
#endif //WICKEDENGINE_BUILD_METAL

#endif // WIGRAPHICSDEVICE_METAL_H
