#include "wiGraphicsDevice_Metal.h"
#ifdef WICKEDENGINE_BUILD_METAL
#include <Foundation/Foundation.hpp>
#include <iostream>
#include "wiGraphicsDevice.h"

namespace wi::graphics
{
GraphicsDevice_Metal::GraphicsDevice_Metal(platform::window_type window, ValidationMode validationMode_, GPUPreference preference)
{
    device = MTL::CreateSystemDefaultDevice();
    capabilities |= GraphicsDeviceCapability::ALIASING_GENERIC;

    // This functionalty is missing from Vulkan but might be added in the future:
    //	Issue: https://github.com/KhronosGroup/Vulkan-Docs/issues/2079
    capabilities |= GraphicsDeviceCapability::COPY_BETWEEN_DIFFERENT_IMAGE_ASPECTS_NOT_SUPPORTED;

    capabilities |= GraphicsDeviceCapability::R9G9B9E5_SHAREDEXP_RENDERABLE;
    if (!device)
    {
        std::cerr << "Error: Metal is not supported on this device." << std::endl;
    }
}

GraphicsDevice_Metal::~GraphicsDevice_Metal()
{
    if (commandQueue) commandQueue->release();
    if (device) device->release();

    for (auto& resource : resourceCache)
    {
        resource.second->release();
    }
}

bool GraphicsDevice_Metal::CreateSwapChain(const SwapChainDesc *desc, platform::window_type window, SwapChain *swapchain) const
{
	//TODO: Implement
	return true;
}

bool GraphicsDevice_Metal::CreateBuffer2(const GPUBufferDesc *desc, const std::function<void (void *)> &init_callback, GPUBuffer *buffer, const GPUResource *alias, uint64_t alias_offset) const
{
    /*if (!desc || !buffer)
        return false;

    MTL::Buffer* metalBuffer = device->newBuffer(desc->size, MTL::ResourceStorageModePrivate);
    if (!metalBuffer)
        return false;

    if (init_callback)
    {
        void* contents = metalBuffer->contents();
        init_callback(contents);
    }

    buffer->internalResource = metalBuffer; // Assume GPUBuffer has a field for Metal resource
    return true;*/
}

bool GraphicsDevice_Metal::CreateTexture(const TextureDesc *desc, const SubresourceData *initial_data, Texture *texture, const GPUResource *alias, uint64_t alias_offset) const
{
	//TODO: Implement
	return true;
}

bool GraphicsDevice_Metal::CreateShader(ShaderStage stage, const void *shadercode, size_t shadercode_size, Shader *shader) const
{
    /*if (!shadercode || !shader)
        return false;

    MTL::Library* library = device->newLibraryWithData(MTL::Data::data(shadercode, shadercode_size), nullptr);
    if (!library)
        return false;

    std::string entryPoint = "main"; // Assuming entry point is "main"
    MTL::Function* function = library->newFunction(MTL::FunctionDescriptor::alloc()->init(entryPoint.c_str()));
    if (!function)
        return false;

    shader->internalResource = function; // Assume Shader has a field for Metal function
    return true;*/
}

bool GraphicsDevice_Metal::CreateSampler(const SamplerDesc *desc, Sampler *sampler) const
{
	//TODO: Implement
	return true;
}

bool GraphicsDevice_Metal::CreateQueryHeap(const GPUQueryHeapDesc *desc, GPUQueryHeap *queryheap) const
{
	//TODO: Implement
	return true;
}

bool GraphicsDevice_Metal::CreatePipelineState(const PipelineStateDesc *desc, PipelineState *pso, const RenderPassInfo *renderpass_info) const
{
	//TODO: Implement
	return true;
}

int GraphicsDevice_Metal::CreateSubresource(Texture* texture, SubresourceType type, uint32_t firstSlice, uint32_t sliceCount, uint32_t firstMip, uint32_t mipCount, const Format* format_change, const ImageAspect* aspect, const Swizzle* swizzle, float min_lod_clamp) const {
    /*if (!texture || !texture->resource)
    {
        std::cerr << "Error: Invalid texture or texture resource." << std::endl;
        return -1; // Failure
    }

    auto metalTexture = static_cast<MTL::Texture*>(texture->resource);

    // Calculate the subresource range based on the specified parameters
    MTL::TextureDescriptor* textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
        metalTexture->pixelFormat(),
        metalTexture->width(),
        metalTexture->height(),
        metalTexture->mipmapLevelCount() > 1
    );

    if (format_change)
    {
        textureDesc->setPixelFormat(ConvertToMetalPixelFormat(*format_change));
    }

    textureDesc->setMipmapLevelCount(mipCount);
    textureDesc->setArrayLength(sliceCount);

    // Validate and handle aspects and swizzles if required
    if (aspect)
    {
        // Metal does not expose explicit aspect settings like Vulkan; handled internally.
        // Add custom code for handling specific image aspects, if applicable.
    }

    if (swizzle)
    {
        // Metal API does not directly support swizzle; this must be done in shaders.
    }

    textureDesc->setUsage(metalTexture->usage());

    MTL::Texture* subresourceTexture = metalTexture->newTextureView(
        textureDesc->pixelFormat(),
        metalTexture->textureType(),
        metalTexture->pixelFormat(),
        NS::Range(firstMip, mipCount),
        NS::Range(firstSlice, sliceCount)
    );

    if (!subresourceTexture)
    {
        std::cerr << "Error: Failed to create subresource texture view." << std::endl;
        return -1; // Failure
    }

    // Cache or store the subresource texture for future reference
    texture->subresources.emplace_back(subresourceTexture);
    return static_cast<int>(texture->subresources.size() - 1); // Return the subresource index*/
}

int GraphicsDevice_Metal::CreateSubresource(GPUBuffer *buffer, SubresourceType type, uint64_t offset, uint64_t size, const Format *format_change, const uint32_t *structuredbuffer_stride_change) const
{
    /*if (!buffer || !buffer->resource)
    {
        std::cerr << "Error: Invalid buffer or buffer resource." << std::endl;
        return -1; // Failure
    }

    auto metalBuffer = static_cast<MTL::Buffer*>(buffer->resource);

    if (offset + size > metalBuffer->length())
    {
        std::cerr << "Error: Subresource range is out of bounds." << std::endl;
        return -1; // Failure
    }

    MTL::Buffer* subresourceBuffer = metalBuffer->newBufferWithBytesNoCopy(
        static_cast<void*>(static_cast<uint8_t*>(metalBuffer->contents()) + offset),
        size,
        metalBuffer->storageMode(),
        nullptr // No custom deallocator needed
    );

    if (!subresourceBuffer)
    {
        std::cerr << "Error: Failed to create subresource buffer." << std::endl;
        return -1; // Failure
    }

    // Cache or store the subresource buffer for future reference
    buffer->subresources.emplace_back(subresourceBuffer);
    return static_cast<int>(buffer->subresources.size() - 1); // Return the subresource index*/
}

void GraphicsDevice_Metal::DeleteSubresources(GPUResource *resource)
{

}

int GraphicsDevice_Metal::GetDescriptorIndex(const GPUResource *resource, SubresourceType type, int subresource) const
{
	//TODO: Implement
	return true;
}

int GraphicsDevice_Metal::GetDescriptorIndex(const Sampler *sampler) const
{
	//TODO: Implement
	return true;
}

CommandList GraphicsDevice_Metal::BeginCommandList(QUEUE_TYPE queue)
{
    /*if (!commandQueue)
    {
        commandQueue = device->newCommandQueue();
    }

    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    return reinterpret_cast<CommandList>(commandBuffer); // Assuming CommandList is typedef'd to a Metal-specific type*/
}

void GraphicsDevice_Metal::SubmitCommandLists()
{
	// Finalize and commit all command buffers in the current frame
    /*for (MTL::CommandBuffer* commandBuffer : commandBuffers)
    {
        commandBuffer->commit();
    }
    commandBuffers.clear();*/
}

void GraphicsDevice_Metal::WaitForGPU() const
{

}

void GraphicsDevice_Metal::ClearPipelineStateCache()
{

}

size_t GraphicsDevice_Metal::GetActivePipelineCount() const
{

}

ShaderFormat GraphicsDevice_Metal::GetShaderFormat() const
{

}

Texture GraphicsDevice_Metal::GetBackBuffer(const SwapChain *swapchain) const
{

}

ColorSpace GraphicsDevice_Metal::GetSwapChainColorSpace(const SwapChain *swapchain) const
{

}

bool GraphicsDevice_Metal::IsSwapChainSupportsHDR(const SwapChain *swapchain) const
{

}

uint64_t GraphicsDevice_Metal::GetMinOffsetAlignment(const GPUBufferDesc *desc) const
{

}

GraphicsDevice::MemoryUsage GraphicsDevice_Metal::GetMemoryUsage() const
{

}

uint32_t GraphicsDevice_Metal::GetMaxViewportCount() const
{

}

void GraphicsDevice_Metal::WaitCommandList(CommandList cmd, CommandList wait_for)
{

}

void GraphicsDevice_Metal::WaitQueue(CommandList cmd, QUEUE_TYPE wait_for)
{

}

void GraphicsDevice_Metal::RenderPassBegin(const SwapChain *swapchain, CommandList cmd)
{

}

void GraphicsDevice_Metal::RenderPassBegin(const RenderPassImage *images, uint32_t image_count, CommandList cmd, RenderPassFlags flags)
{

}

void GraphicsDevice_Metal::RenderPassEnd(CommandList cmd)
{

}

void GraphicsDevice_Metal::BindScissorRects(uint32_t numRects, const Rect *rects, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindViewports(uint32_t NumViewports, const Viewport *pViewports, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindResource(const GPUResource *resource, uint32_t slot, CommandList cmd, int subresource)
{

}

void GraphicsDevice_Metal::BindResources(const GPUResource * const *resources, uint32_t slot, uint32_t count, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindUAV(const GPUResource *resource, uint32_t slot, CommandList cmd, int subresource)
{

}

void GraphicsDevice_Metal::BindUAVs(const GPUResource * const *resources, uint32_t slot, uint32_t count, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindSampler(const Sampler *sampler, uint32_t slot, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindConstantBuffer(const GPUBuffer *buffer, uint32_t slot, CommandList cmd, uint64_t offset)
{

}

void GraphicsDevice_Metal::BindVertexBuffers(const GPUBuffer * const *vertexBuffers, uint32_t slot, uint32_t count, const uint32_t *strides, const uint64_t *offsets, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindIndexBuffer(const GPUBuffer *indexBuffer, const IndexBufferFormat format, uint64_t offset, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindStencilRef(uint32_t value, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindBlendFactor(float r, float g, float b, float a, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindPipelineState(const PipelineState *pso, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindComputeShader(const Shader *cs, CommandList cmd)
{

}

void GraphicsDevice_Metal::BindDepthBounds(float min_bounds, float max_bounds, CommandList cmd)
{

}

void GraphicsDevice_Metal::Draw(uint32_t vertexCount, uint32_t startVertexLocation, CommandList cmd)
{

}

void GraphicsDevice_Metal::DrawIndexed(uint32_t indexCount, uint32_t startIndexLocation, int32_t baseVertexLocation, CommandList cmd)
{

}

void GraphicsDevice_Metal::DrawInstanced(uint32_t vertexCount, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation, CommandList cmd)
{

}

void GraphicsDevice_Metal::DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation, CommandList cmd)
{

}

void GraphicsDevice_Metal::DrawInstancedIndirect(const GPUBuffer *args, uint64_t args_offset, CommandList cmd)
{

}

void GraphicsDevice_Metal::DrawIndexedInstancedIndirect(const GPUBuffer *args, uint64_t args_offset, CommandList cmd)
{

}

void GraphicsDevice_Metal::DrawInstancedIndirectCount(const GPUBuffer *args, uint64_t args_offset, const GPUBuffer *count, uint64_t count_offset, uint32_t max_count, CommandList cmd)
{

}

void GraphicsDevice_Metal::DrawIndexedInstancedIndirectCount(const GPUBuffer *args, uint64_t args_offset, const GPUBuffer *count, uint64_t count_offset, uint32_t max_count, CommandList cmd)
{

}

void GraphicsDevice_Metal::Dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ, CommandList cmd)
{

}

void GraphicsDevice_Metal::DispatchIndirect(const GPUBuffer *args, uint64_t args_offset, CommandList cmd)
{

}

void GraphicsDevice_Metal::CopyResource(const GPUResource *pDst, const GPUResource *pSrc, CommandList cmd)
{

}

void GraphicsDevice_Metal::CopyBuffer(const GPUBuffer *pDst, uint64_t dst_offset, const GPUBuffer *pSrc, uint64_t src_offset, uint64_t size, CommandList cmd)
{

}

void GraphicsDevice_Metal::CopyTexture(const Texture *dst, uint32_t dstX, uint32_t dstY, uint32_t dstZ, uint32_t dstMip, uint32_t dstSlice, const Texture *src, uint32_t srcMip, uint32_t srcSlice, CommandList cmd, const Box *srcbox, ImageAspect dst_aspect, ImageAspect src_aspect)
{

}

void GraphicsDevice_Metal::QueryBegin(const GPUQueryHeap *heap, uint32_t index, CommandList cmd)
{

}

void GraphicsDevice_Metal::QueryEnd(const GPUQueryHeap *heap, uint32_t index, CommandList cmd)
{

}

void GraphicsDevice_Metal::QueryResolve(const GPUQueryHeap *heap, uint32_t index, uint32_t count, const GPUBuffer *dest, uint64_t dest_offset, CommandList cmd)
{

}

void GraphicsDevice_Metal::Barrier(const GPUBarrier *barriers, uint32_t numBarriers, CommandList cmd)
{

}

void GraphicsDevice_Metal::PushConstants(const void *data, uint32_t size, CommandList cmd, uint32_t offset)
{

}

void GraphicsDevice_Metal::ClearUAV(const GPUResource *resource, uint32_t value, CommandList cmd)
{

}

void GraphicsDevice_Metal::EventBegin(const char *name, CommandList cmd)
{

}

void GraphicsDevice_Metal::EventEnd(CommandList cmd)
{

}

void GraphicsDevice_Metal::SetMarker(const char *name, CommandList cmd)
{

}

RenderPassInfo GraphicsDevice_Metal::GetRenderPassInfo(CommandList cmd)
{

}

GraphicsDevice::GPULinearAllocator &GraphicsDevice_Metal::GetFrameAllocator(CommandList cmd)
{
    //FIXME: Placeholder
    //return tmpAllocator;
    return GetCommandList(cmd).frame_allocators[GetBufferIndex()];
}

void GraphicsDevice_Metal::CommandList_Metal::reset(uint32_t bufferIndex)
{
    this->bufferIndex = bufferIndex;
    waitEvents.clear();
    signalEvents.clear();
    renderPassInfo.descriptor = nullptr;
    pipelineCache.clear();
    discards.clear();
    frameFences.clear();
	frame_allocators[bufferIndex].reset();

    if (commandBuffer)
    {
        commandBuffer = nullptr;
    }
}

MTL::CommandBuffer *GraphicsDevice_Metal::CommandList_Metal::GetCommandBuffer()
{
    if (!commandBuffer)
    {
        commandBuffer = commandQueue->commandBuffer();
    }
    return commandBuffer;
}

MTL::RenderCommandEncoder *GraphicsDevice_Metal::CommandList_Metal::GetRenderEncoder()
{
    if (!renderEncoder)
    {
        auto commandBuffer = GetCommandBuffer();
        renderEncoder = commandBuffer->renderCommandEncoder(renderPassInfo.descriptor.get());
    }
    return renderEncoder;
}

MTL::ComputeCommandEncoder *GraphicsDevice_Metal::CommandList_Metal::GetComputeEncoder()
{
    if (!computeEncoder)
    {
        auto commandBuffer = GetCommandBuffer();
        computeEncoder = commandBuffer->computeCommandEncoder();
    }
    return computeEncoder;
}
/*
std::shared_ptr<MTL::RenderPipelineState> GraphicsDevice_Metal::CommandList_Metal::getPipelineState(size_t pipelineHash, std::shared_ptr<MTLDevice> device, MTLRenderPipelineDescriptor *descriptor)
{
    auto it = pipelineCache.find(pipelineHash);
    if (it != pipelineCache.end())
    {
        return it->second;
    }

    NS::Error* error = nullptr;
    auto pipelineState = device->newRenderPipelineState(descriptor, &error);
    if (!pipelineState)
    {
        throw std::runtime_error("Failed to create pipeline state: " + std::string(error->localizedDescription()->utf8String()));
    }

    pipelineCache[pipelineHash] = pipelineState;
    return pipelineState;
}*/

bool GraphicsDevice_Metal::CopyAllocator::CopyCMD::IsValid() const {
    return commandList != nullptr;
}

bool GraphicsDevice_Metal::CopyAllocator::CopyCMD::IsCompleted() const {
    return std::stoi(fence->label()->utf8String()) >= fenceValueSignaled;
}

}
#endif
