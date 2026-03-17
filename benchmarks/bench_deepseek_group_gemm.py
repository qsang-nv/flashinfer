"""Benchmark pure group GEMM kernels matching DeepSeek V3/R1 MoE configurations.

Reproduces the same workload as the TRT-LLM group GEMM benchmark table:
  - Group GEMM 1 (gate_up): [M, 7168] x [7168, 4096]  per expert
  - Group GEMM 2 (down):    [M, 2048] x [2048, 7168]  per expert

where M = batch_size_per_expert, and the number of groups = num_experts_per_gpu.

Usage:
    python bench_deepseek_group_gemm.py                          # all configs
    python bench_deepseek_group_gemm.py --ep 8                   # only EP8
    python bench_deepseek_group_gemm.py --gemm 1                 # only GEMM1
    python bench_deepseek_group_gemm.py --batch-sizes 64,128,256
    python bench_deepseek_group_gemm.py --backend cublas          # try cublas
    python bench_deepseek_group_gemm.py --peak-tflops 296
"""

import argparse
import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

HIDDEN_DIM = 7168
INTERMEDIATE_SIZE = 2048
TOTAL_EXPERTS = 256

EP_CONFIGS = {
    8: 32,
    16: 16,
    32: 8,
    64: 4,
    128: 2,
}

KNOWN_FP8_PEAK_TFLOPS = {
    "NVIDIA H20": 296,
    "NVIDIA H100": 1979,
    "NVIDIA H100 80GB HBM3": 1979,
    "NVIDIA H200": 1979,
    "NVIDIA A100": 624,
    "NVIDIA B200": 9000,
    "NVIDIA GB200": 9000,
}


def get_peak_tflops(dtype):
    name = torch.cuda.get_device_name(0)
    if dtype in (torch.float8_e4m3fn,):
        for key, val in KNOWN_FP8_PEAK_TFLOPS.items():
            if key in name:
                return val
    return None


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    amax = x.abs().amax().clamp(min=1e-12)
    scale = finfo.max / amax
    x_fp8 = (x * scale).clamp(min=finfo.min, max=finfo.max).to(dtype)
    return x_fp8, scale.float().reciprocal()


def bench_bmm_fp8(num_experts, batch_per_expert, d_in, d_out, backend, peak_tflops):
    """Use bmm_fp8 for uniform-batch grouped GEMM (FP8)."""
    A_bf16 = torch.randn(num_experts, batch_per_expert, d_in, device="cuda:0", dtype=torch.bfloat16)
    B_bf16 = torch.randn(num_experts, d_out, d_in, device="cuda:0", dtype=torch.bfloat16)

    A_fp8, A_inv_scale = to_float8(A_bf16)
    B_fp8_t = B_bf16.transpose(-2, -1).contiguous()
    B_fp8, B_inv_scale = to_float8(B_fp8_t)

    out_dtype = torch.bfloat16

    def run():
        return flashinfer.gemm.bmm_fp8(
            A=A_fp8, B=B_fp8,
            A_scale=A_inv_scale, B_scale=B_inv_scale,
            dtype=out_dtype, backend=backend,
        )

    measurements = bench_gpu_time(
        run,
        enable_cupti=True,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    median_ms = np.median(measurements)
    median_us = median_ms * 1000
    flops = 2 * num_experts * batch_per_expert * d_in * d_out
    tflops = flops / (median_ms * 1e-3) / 1e12

    sol_str = ""
    if peak_tflops:
        sol = tflops / peak_tflops * 100
        sol_str = f"  SOL={sol:6.2f}%"

    return median_us, tflops, sol_str


def bench_segment_gemm(num_experts, batch_per_expert, d_in, d_out, dtype, peak_tflops):
    """Use SegmentGEMMWrapper for grouped GEMM (supports FP8/BF16)."""
    W = torch.randn(num_experts, d_out, d_in, device="cuda:0").to(dtype)
    X = torch.randn(num_experts * batch_per_expert, d_in, device="cuda:0").to(dtype)
    Y = torch.empty(num_experts * batch_per_expert, d_out, dtype=torch.bfloat16, device="cuda:0")

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    segment_gemm = flashinfer.gemm.SegmentGEMMWrapper(workspace_buffer, backend="auto")
    seg_indptr = torch.arange(
        0, (num_experts + 1) * batch_per_expert, batch_per_expert,
        dtype=torch.int64, device="cuda:0",
    )

    def run():
        segment_gemm.run(X, W, num_experts, True, out=Y, seg_indptr=seg_indptr)

    measurements = bench_gpu_time(
        run,
        enable_cupti=True,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    median_ms = np.median(measurements)
    median_us = median_ms * 1000
    flops = 2 * num_experts * batch_per_expert * d_in * d_out
    tflops = flops / (median_ms * 1e-3) / 1e12

    sol_str = ""
    if peak_tflops:
        sol = tflops / peak_tflops * 100
        sol_str = f"  SOL={sol:6.2f}%"

    return median_us, tflops, sol_str


def run_benchmark(args):
    peak_tflops = args.peak_tflops or get_peak_tflops(torch.float8_e4m3fn)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Backend: {args.backend}")
    if peak_tflops:
        print(f"Peak TFLOPS (for SOL): {peak_tflops}")
    else:
        print("Peak TFLOPS: unknown (use --peak-tflops to set)")
    print()

    ep_list = [args.ep] if args.ep else sorted(EP_CONFIGS.keys())
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else [8, 16, 32, 64, 128, 256, 512]
    gemm_list = [args.gemm] if args.gemm else [1, 2]

    gemm_configs = {
        1: ("GEMM1 gate_up", HIDDEN_DIM, 2 * INTERMEDIATE_SIZE),
        2: ("GEMM2 down", INTERMEDIATE_SIZE, HIDDEN_DIM),
    }

    use_bmm = args.backend in ("cublas", "cudnn", "cutlass")

    for gemm_id in gemm_list:
        label, d_in, d_out = gemm_configs[gemm_id]
        print("=" * 100)
        print(f"{label}: [M, {d_in}] x [{d_in}, {d_out}]")
        print("=" * 100)

        col_width = 30
        header = f"{'EP':>4} {'Experts/GPU':>11} |"
        for bs in batch_sizes:
            header += f" {'M=' + str(bs):^{col_width}} |"
        print(header)
        print("-" * len(header))

        for ep in ep_list:
            num_experts = EP_CONFIGS[ep]
            row = f"EP{ep:>3} {num_experts:>11} |"

            for bs in batch_sizes:
                try:
                    if use_bmm:
                        us, tflops, sol_str = bench_bmm_fp8(
                            num_experts, bs, d_in, d_out, args.backend, peak_tflops
                        )
                    else:
                        us, tflops, sol_str = bench_segment_gemm(
                            num_experts, bs, d_in, d_out, torch.float8_e4m3fn, peak_tflops
                        )
                    cell = f"{us:7.1f}us {tflops:6.1f}T{sol_str}"
                except Exception as e:
                    cell = f"ERR: {str(e)[:20]}"
                row += f" {cell:^{col_width}} |"

            print(row)

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek V3/R1 MoE group GEMM kernels"
    )
    parser.add_argument(
        "--ep", type=int, default=None, choices=list(EP_CONFIGS.keys()),
        help="Expert parallelism degree (default: all)",
    )
    parser.add_argument(
        "--batch-sizes", type=str, default=None,
        help="Comma-separated batch_size_per_expert values (default: 8,16,32,64,128,256,512)",
    )
    parser.add_argument(
        "--gemm", type=int, default=None, choices=[1, 2],
        help="Which GEMM to test: 1=gate_up, 2=down (default: both)",
    )
    parser.add_argument(
        "--backend", type=str, default="cublas",
        choices=["cublas", "cudnn", "segment_gemm"],
        help="GEMM backend: cublas/cudnn use bmm_fp8, segment_gemm uses SegmentGEMMWrapper (default: cublas)",
    )
    parser.add_argument(
        "--peak-tflops", type=float, default=None,
        help="Peak TFLOPS for SOL calculation (default: auto-detect)",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
