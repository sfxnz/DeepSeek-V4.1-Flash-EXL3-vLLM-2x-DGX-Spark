K-1 attempt 1: DSV41_MHC_DET_SPLITS=16 self-disarmed on both ranks at load:
  dsv41: mhc det lever is OFF: target: prepare failed: RuntimeError('Assertion error
  (/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:43): d.scalar_type() == torch::kFloat'); stock kernels stay
(attempt1-boot.txt, attempt1-lever-lines.txt). AUDIT=strict flagged it; every other K lever engaged on both ranks.

Cause: prepare() runs from the load hooks (DeepseekV4Model.finalize_mhc_broadcast_weights, DSpark
load_weights), which vLLM calls inside set_default_torch_dtype(model dtype = bf16)
(model_loader/base_loader.py:53-65). The self-tests allocated their DeepGEMM reference outputs
with a bare torch.empty and their inputs with a bare torch.randn, so under the loader they were bf16
and DeepGEMM's tf32_hc_prenorm_gemm asserted fp32 outputs. The kernel_study GPU checks ran with the
default fp32 dtype and never hit it. Serving code (det_pre_delayed, det_post) already names every dtype.

Fix: the self-tests name their dtype (fp32 via _randn(g, *shape); fp32 reference outputs). Inputs are
the same values as before under an fp32 default. Unit test test_allocations_name_their_dtype pins it
(fails on 66bc26c: 12 bare calls; passes on the fix).

GPU repro (spark1, serve down, GPU free; image review-e14; kernel_study/mhc_det/lever_smoke.py main()
with mhc_det.prepare wrapped in torch.set_default_dtype(bf16), gpu-repro-driver.py.txt; patch dir mounted
at /opt/dsv41-patch as run.sh does, real layer 0/7 mHC weights):
  66bc26c mhc_det.py : same assertion, lever OFF, FAIL (rc 1)  gpu-repro-head66bc26c-stdout.txt
  fixed   mhc_det.py : "mhc det engaged: smoke: 5 fn packed", bitwise pre/post/broadcast, graph replays, PASS (rc 0)
                        gpu-repro-fixed-stdout.txt, gpu-repro-fixed-lever_smoke.json
CPU strict dry-run after the patch change (ARMS-r3 section 3): spark1 head on/off pass, spark2 worker on/off pass (dryrun/).
Unit tests: 798 OK, 14 skipped, 0 failures; kit/render.py --check rc 0.
