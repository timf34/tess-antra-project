# Generation runtime repair (8 October 2026)

Free-text generation did not start: the default vLLM 0.26.0 binary required
`libcudart.so.13`, despite using a CUDA 12.8 extra index for PyTorch. The pod
has an H200 and driver 570.195.03. Real activation/intervention data are
preserved locally and on the pod; no generated observations are being replaced.

The generation environment now explicitly selects the official vLLM 0.26.0
CUDA 12.9 wheel and PyTorch 2.11.0+cu129. NVIDIA's CUDA 12.9 compatibility
package 575.57.08 supplies supported PTX JIT libraries on driver 570. The
compatibility path applies only after capture, to the separate generation
process. The model revision, EasySteer revision, stimuli, splits, intervention
scales, and sampling design are unchanged. Runtime CUDA initialization and a
real GPU operation must pass before generation. Successful full generation
remains to be verified.

Sources: [vLLM release assets](https://github.com/vllm-project/vllm/releases/tag/v0.26.0)
and [NVIDIA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

The CUDA 12.9 runtime initialized and passed its GPU operation on the next run.
Full vLLM entrypoint import then failed because its transitive TorchCodec wheel
required `libnvrtc.so.13`. Generation still had not started. Pinning the official
TorchCodec 0.14.0 CPU wheel avoids this unrelated CUDA 13 media dependency; the
study uses text only. FFmpeg libraries are installed for its import, and the
runtime check now imports the actual LLM entrypoint. Full traceback was retained
privately. Pod restart was delayed by unavailable host GPU capacity.
