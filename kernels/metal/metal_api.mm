#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#include <string.h>
#include <stdio.h>
#include "metal_api.h"

namespace {
id<MTLDevice> gDevice = nil;
id<MTLLibrary> gLibrary = nil;
id<MTLComputePipelineState> gPipeline = nil;

static void setErr(char* buf, int len, NSString* msg) {
  if (buf && len > 0 && msg) {
    snprintf(buf, (size_t)len, "%s", msg.UTF8String);
  }
}

struct Uniforms {
  uint32_t num_items;
  uint32_t num_candidates;
  uint32_t bytes_per_candidate;
  uint32_t num_groups;
  float    penalty_coeff;
  float    penalty_power;
  uint32_t num_obj_terms;
  uint32_t num_soft_constraints;
};
} // namespace

int knapsack_metal_init_from_data(const void* data, size_t len, char* errbuf, int errlen) {
  @autoreleasepool {
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) { setErr(errbuf, errlen, @"Metal device unavailable"); return -1; }
    if (!data || len == 0) { setErr(errbuf, errlen, @"Invalid metallib data"); return -2; }

    NSError* err = nil;
    // Copy bytes so dispatch_data can own and free them later.
    void* copy = malloc(len);
    if (!copy) { setErr(errbuf, errlen, @"Out of memory allocating metallib copy"); return -3; }
    memcpy(copy, data, len);
    dispatch_data_t d = dispatch_data_create(copy, len, NULL, DISPATCH_DATA_DESTRUCTOR_FREE);
    gLibrary = [gDevice newLibraryWithData:d error:&err];
    if (!gLibrary || err) { setErr(errbuf, errlen, err ? err.localizedDescription : @"Failed to load MTLLibrary"); return -3; }

    id<MTLFunction> fn = [gLibrary newFunctionWithName:@"eval_block_candidates"];
    if (!fn) { setErr(errbuf, errlen, @"Kernel eval_block_candidates not found"); return -4; }

    gPipeline = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    if (!gPipeline || err) { setErr(errbuf, errlen, err ? err.localizedDescription : @"Failed to create pipeline"); return -5; }

    return 0;
  }
}

int knapsack_metal_init_from_source(const char* src, size_t len, char* errbuf, int errlen) {
  @autoreleasepool {
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) { setErr(errbuf, errlen, @"Metal device unavailable"); return -1; }
    if (!src || len == 0) { setErr(errbuf, errlen, @"Invalid shader source"); return -2; }

    NSString* msl = [[NSString alloc] initWithBytes:src length:len encoding:NSUTF8StringEncoding];
    if (!msl) { setErr(errbuf, errlen, @"Failed to decode shader source as UTF-8"); return -3; }

    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    gLibrary = [gDevice newLibraryWithSource:msl options:opts error:&err];
    if (!gLibrary || err) { setErr(errbuf, errlen, err ? err.localizedDescription : @"Failed to compile MTLLibrary from source"); return -4; }

    id<MTLFunction> fn = [gLibrary newFunctionWithName:@"eval_block_candidates"];
    if (!fn) { setErr(errbuf, errlen, @"Kernel eval_block_candidates not found"); return -5; }

    gPipeline = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    if (!gPipeline || err) { setErr(errbuf, errlen, err ? err.localizedDescription : @"Failed to create pipeline"); return -6; }

    return 0;
  }
}

int knapsack_metal_eval(const MetalEvalIn* in, MetalEvalOut* out, char* errbuf, int errlen) {
  @autoreleasepool {
    if (!gDevice || !gPipeline) { setErr(errbuf, errlen, @"Metal not initialized"); return -1; }
    if (!in || !out || !in->candidates || !out->obj || !out->soft_penalty) {
      setErr(errbuf, errlen, @"Invalid input/output pointers"); return -2;
    }
    if (in->num_items < 0 || in->num_candidates < 0) {
      setErr(errbuf, errlen, @"Negative sizes not allowed"); return -3;
    }

    const uint32_t num_items = (uint32_t)in->num_items;
    const uint32_t num_cands = (uint32_t)in->num_candidates;
    const uint32_t bytes_per_cand = (num_items + 3u) / 4u; // 2 bits per item
    const size_t candBytes = (size_t)bytes_per_cand * (size_t)num_cands;

    id<MTLCommandQueue> q = [gDevice newCommandQueue];
    if (!q) { setErr(errbuf, errlen, @"Failed to create command queue"); return -4; }

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:gPipeline];

    id<MTLBuffer> candBuf = [gDevice newBufferWithBytes:in->candidates length:candBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> objBuf  = [gDevice newBufferWithLength:sizeof(float)*num_cands options:MTLResourceStorageModeShared];
    id<MTLBuffer> penBuf  = [gDevice newBufferWithLength:sizeof(float)*num_cands options:MTLResourceStorageModeShared];
    if (!candBuf || !objBuf || !penBuf) { setErr(errbuf, errlen, @"Failed to create buffers"); return -5; }

  [enc setBuffer:candBuf offset:0 atIndex:0];
  [enc setBuffer:objBuf  offset:0 atIndex:1];
  [enc setBuffer:penBuf  offset:0 atIndex:2];
  // Optional attribute buffers (values, weights, group caps)
    id<MTLBuffer> valBuf = nil;
    id<MTLBuffer> wgtBuf = nil;
    id<MTLBuffer> capBuf = nil;
    if (in->item_values && num_items > 0 && (!in->obj_attrs || in->num_obj_terms <= 0)) {
      valBuf = [gDevice newBufferWithBytes:in->item_values length:sizeof(float)*num_items options:MTLResourceStorageModeShared];
      [enc setBuffer:valBuf offset:0 atIndex:3];
    }
    if (in->item_weights && num_items > 0) {
      wgtBuf = [gDevice newBufferWithBytes:in->item_weights length:sizeof(float)*num_items options:MTLResourceStorageModeShared];
      [enc setBuffer:wgtBuf offset:0 atIndex:4];
    }
    if (in->group_capacities && in->num_groups > 0) {
      capBuf = [gDevice newBufferWithBytes:in->group_capacities length:sizeof(float)*in->num_groups options:MTLResourceStorageModeShared];
      [enc setBuffer:capBuf offset:0 atIndex:5];
    }

    // New: multi-term objective
    id<MTLBuffer> objAttrBuf = nil, objWBuf = nil;
    if (in->obj_attrs && in->obj_weights && in->num_obj_terms > 0 && num_items > 0) {
      const size_t objAttrCount = (size_t)in->num_obj_terms * (size_t)num_items;
      objAttrBuf = [gDevice newBufferWithBytes:in->obj_attrs length:sizeof(float)*objAttrCount options:MTLResourceStorageModeShared];
      objWBuf    = [gDevice newBufferWithBytes:in->obj_weights length:sizeof(float)*in->num_obj_terms options:MTLResourceStorageModeShared];
      [enc setBuffer:objAttrBuf offset:0 atIndex:6];
      [enc setBuffer:objWBuf    offset:0 atIndex:7];
    }

    // New: global soft constraints
    id<MTLBuffer> consAttrBuf = nil, consLimBuf = nil, consWBuf = nil, consPBuf = nil;
    if (in->cons_attrs && in->cons_limits && in->cons_weights && in->cons_powers && in->num_soft_constraints > 0 && num_items > 0) {
      const size_t consAttrCount = (size_t)in->num_soft_constraints * (size_t)num_items;
      consAttrBuf = [gDevice newBufferWithBytes:in->cons_attrs length:sizeof(float)*consAttrCount options:MTLResourceStorageModeShared];
      consLimBuf  = [gDevice newBufferWithBytes:in->cons_limits length:sizeof(float)*in->num_soft_constraints options:MTLResourceStorageModeShared];
      consWBuf    = [gDevice newBufferWithBytes:in->cons_weights length:sizeof(float)*in->num_soft_constraints options:MTLResourceStorageModeShared];
      consPBuf    = [gDevice newBufferWithBytes:in->cons_powers  length:sizeof(float)*in->num_soft_constraints options:MTLResourceStorageModeShared];
      [enc setBuffer:consAttrBuf offset:0 atIndex:8];
      [enc setBuffer:consLimBuf  offset:0 atIndex:9];
      [enc setBuffer:consWBuf    offset:0 atIndex:10];
      [enc setBuffer:consPBuf    offset:0 atIndex:11];
    }

  Uniforms U{num_items, num_cands, bytes_per_cand, (uint32_t)in->num_groups, in->penalty_coeff, in->penalty_power, (uint32_t) (in->obj_attrs && in->num_obj_terms>0 ? in->num_obj_terms : 0), (uint32_t) (in->cons_attrs && in->num_soft_constraints>0 ? in->num_soft_constraints : 0)};
    [enc setBytes:&U length:sizeof(U) atIndex:15];

    MTLSize grid = MTLSizeMake((NSUInteger)num_cands, 1, 1);
    NSUInteger maxT = gPipeline.maxTotalThreadsPerThreadgroup;
    MTLSize tg = MTLSizeMake((NSUInteger)MIN((NSUInteger)64, maxT), 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];

    // Copy results back into user-provided buffers (avoids passing Go memory to MTL).
    memcpy(out->obj,  objBuf.contents, sizeof(float)*num_cands);
    memcpy(out->soft_penalty, penBuf.contents, sizeof(float)*num_cands);
    return 0;
  }
}
