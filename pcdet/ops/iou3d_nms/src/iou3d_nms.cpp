/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "iou3d_nms.h"

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;


/*
 * Reusable NMS workspace.
 *
 * This changes only memory allocation strategy.
 * The NMS CUDA kernel, IoU calculation, CPU suppression order,
 * threshold and returned indices are unchanged.
 *
 * The current StreamDSGN config uses NMS_PRE_MAXSIZE=4096, so
 * preallocating workspace for at least 4096 boxes removes almost
 * all runtime allocation from steady-state inference.
 */
struct NmsWorkspace {
    unsigned long long *mask_gpu = nullptr;
    unsigned long long *mask_cpu = nullptr;

    size_t mask_capacity_elems = 0;

    int gpu_device = -1;

    std::vector<unsigned long long> remv_cpu;
};

static thread_local NmsWorkspace g_nms_workspace;


static void ensure_nms_workspace(
    size_t required_mask_elems,
    int required_col_blocks)
{
    NmsWorkspace &ws = g_nms_workspace;

    int current_device = -1;
    CHECK_ERROR(cudaGetDevice(&current_device));

    /*
     * Maximum workspace normally needed by this project's
     * NMS_PRE_MAXSIZE=4096:
     *
     * 4096 * ceil(4096 / 64) uint64 values = 262144 values
     *                                          = 2 MiB.
     */
    const size_t default_mask_elems =
        static_cast<size_t>(4096) *
        static_cast<size_t>(DIVUP(4096, THREADS_PER_BLOCK_NMS));

    size_t wanted_elems = required_mask_elems;
    if (wanted_elems < default_mask_elems) {
        wanted_elems = default_mask_elems;
    }

    /*
     * CUDA device changed: discard only the device-side workspace.
     * Pinned host memory is device independent and may still be reused.
     */
    if (ws.gpu_device != -1 && ws.gpu_device != current_device) {
        if (ws.mask_gpu != nullptr) {
            int restore_device = current_device;

            CHECK_ERROR(cudaSetDevice(ws.gpu_device));
            CHECK_ERROR(cudaFree(ws.mask_gpu));
            CHECK_ERROR(cudaSetDevice(restore_device));

            ws.mask_gpu = nullptr;
        }

        ws.gpu_device = current_device;

        /*
         * GPU buffer has to be allocated again for the new device.
         * Host capacity remains valid.
         */
        if (wanted_elems < ws.mask_capacity_elems) {
            wanted_elems = ws.mask_capacity_elems;
        }

        CHECK_ERROR(
            cudaMalloc(
                reinterpret_cast<void **>(&ws.mask_gpu),
                wanted_elems * sizeof(unsigned long long)
            )
        );

        /*
         * Host buffer may not exist yet.
         */
        if (ws.mask_cpu == nullptr) {
            CHECK_ERROR(
                cudaMallocHost(
                    reinterpret_cast<void **>(&ws.mask_cpu),
                    wanted_elems * sizeof(unsigned long long)
                )
            );
        }

        ws.mask_capacity_elems = wanted_elems;
    }

    if (ws.gpu_device == -1) {
        ws.gpu_device = current_device;
    }

    /*
     * First allocation or capacity growth.
     */
    if (
        ws.mask_gpu == nullptr ||
        ws.mask_cpu == nullptr ||
        ws.mask_capacity_elems < wanted_elems
    ) {
        size_t new_capacity = wanted_elems;

        if (ws.mask_gpu != nullptr) {
            CHECK_ERROR(cudaFree(ws.mask_gpu));
            ws.mask_gpu = nullptr;
        }

        if (ws.mask_cpu != nullptr) {
            CHECK_ERROR(cudaFreeHost(ws.mask_cpu));
            ws.mask_cpu = nullptr;
        }

        CHECK_ERROR(
            cudaMalloc(
                reinterpret_cast<void **>(&ws.mask_gpu),
                new_capacity * sizeof(unsigned long long)
            )
        );

        CHECK_ERROR(
            cudaMallocHost(
                reinterpret_cast<void **>(&ws.mask_cpu),
                new_capacity * sizeof(unsigned long long)
            )
        );

        ws.mask_capacity_elems = new_capacity;
    }

    /*
     * The current config requires at most 64 uint64 blocks for
     * 4096 candidates. Reserve this once to remove vector growth.
     */
    if (ws.remv_cpu.capacity() < 64) {
        ws.remv_cpu.reserve(64);
    }

    ws.remv_cpu.resize(required_col_blocks);

    memset(
        ws.remv_cpu.data(),
        0,
        required_col_blocks * sizeof(unsigned long long)
    );
}


void boxesoverlapLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap);
void boxesoverlapOnebyoneLauncher(const int num_a, const float *boxes_a, const float *boxes_b, float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou);
void boxesioubevOnebyoneLauncher(const int num_a, const float *boxes_a, const float *boxes_b, float *ans_iou);
void nmsLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);
void nmsNormalLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);


int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)

    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_overlap);

    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data<float>();
    const float * boxes_b_data = boxes_b.data<float>();
    float * ans_overlap_data = ans_overlap.data<float>();

    boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_overlap_data);

    return 1;
}

int boxes_overlap_bev_onebyone_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N)

    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_overlap);

    int num_a = boxes_a.size(0);

    const float * boxes_a_data = boxes_a.data<float>();
    const float * boxes_b_data = boxes_b.data<float>();
    float * ans_overlap_data = ans_overlap.data<float>();

    boxesoverlapOnebyoneLauncher(num_a, boxes_a_data, boxes_b_data, ans_overlap_data);

    return 1;
}

int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)
    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_iou);

    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data<float>();
    const float * boxes_b_data = boxes_b.data<float>();
    float * ans_iou_data = ans_iou.data<float>();

    boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

    return 1;
}

int boxes_iou_bev_onebyone_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N)
    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_iou);

    int num_a = boxes_a.size(0);

    const float * boxes_a_data = boxes_a.data<float>();
    const float * boxes_b_data = boxes_b.data<float>();
    float * ans_iou_data = ans_iou.data<float>();

    boxesioubevOnebyoneLauncher(num_a, boxes_a_data, boxes_b_data, ans_iou_data);

    return 1;
}

int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)
    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);

    int boxes_num = boxes.size(0);
    const float * boxes_data = boxes.data<float>();
    long * keep_data = keep.data<long>();

    const int col_blocks =
        DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    const size_t mask_elems =
        static_cast<size_t>(boxes_num) *
        static_cast<size_t>(col_blocks);

    ensure_nms_workspace(
        mask_elems,
        col_blocks
    );

    NmsWorkspace &ws = g_nms_workspace;

    /*
     * Identical CUDA NMS kernel.
     */
    nmsLauncher(
        boxes_data,
        ws.mask_gpu,
        boxes_num,
        nms_overlap_thresh
    );

    /*
     * Same D2H mask copy as before, but destination is persistent
     * pinned memory instead of a newly allocated std::vector.
     *
     * cudaMemcpy remains synchronous, preserving the original
     * execution dependency before the CPU suppression loop.
     */
    CHECK_ERROR(
        cudaMemcpy(
            ws.mask_cpu,
            ws.mask_gpu,
            mask_elems * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        )
    );

    int num_to_keep = 0;

    /*
     * CPU suppression loop is intentionally identical to original.
     */
    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(ws.remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;

            unsigned long long *p =
                ws.mask_cpu +
                static_cast<size_t>(i) *
                static_cast<size_t>(col_blocks);

            for (int j = nblock; j < col_blocks; j++){
                ws.remv_cpu[j] |= p[j];
            }
        }
    }

    if (cudaSuccess != cudaGetLastError()) {
        printf("Error!\n");
    }

    return num_to_keep;
}


int nms_normal_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)

    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);

    int boxes_num = boxes.size(0);
    const float * boxes_data = boxes.data<float>();
    long * keep_data = keep.data<long>();

    const int col_blocks =
        DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    const size_t mask_elems =
        static_cast<size_t>(boxes_num) *
        static_cast<size_t>(col_blocks);

    ensure_nms_workspace(
        mask_elems,
        col_blocks
    );

    NmsWorkspace &ws = g_nms_workspace;

    nmsNormalLauncher(
        boxes_data,
        ws.mask_gpu,
        boxes_num,
        nms_overlap_thresh
    );

    CHECK_ERROR(
        cudaMemcpy(
            ws.mask_cpu,
            ws.mask_gpu,
            mask_elems * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        )
    );

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(ws.remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;

            unsigned long long *p =
                ws.mask_cpu +
                static_cast<size_t>(i) *
                static_cast<size_t>(col_blocks);

            for (int j = nblock; j < col_blocks; j++){
                ws.remv_cpu[j] |= p[j];
            }
        }
    }

    if (cudaSuccess != cudaGetLastError()) {
        printf("Error!\n");
    }

    return num_to_keep;
}

