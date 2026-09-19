# Vision-on boot (SWA width 128, hf_config 1024)

`LANGUAGE_MODEL_ONLY=0`. `./run.sh` does not pass `--language-model-only`.
`hf_config.vision_max_n_token` stays 1024. `vision_n_layers` stays 32.

FlashInfer SM120 dual-cache prefill (`dispatch_dsv4_dual`) only instantiates
SWA `topk=128`. Window+1024=1152 and a clamp to 1024 both miss that table and
would crash the L.A.I.L ~81-token prefill (`num_tokens>64`). Decode SWA is
already 128 / DSpark 192.

`swa_image_tokens_for_dispatch` therefore returns 0 so the SWA *index row*
stays 128. That is not collapsing `vision_max_n_token` on the config object.

Vision tower load is confirmed by `FLASH_ATTN for vit attention` /
`MMEncoderAttention` in the engine log. Weights are ~0.93 GiB bf16.
