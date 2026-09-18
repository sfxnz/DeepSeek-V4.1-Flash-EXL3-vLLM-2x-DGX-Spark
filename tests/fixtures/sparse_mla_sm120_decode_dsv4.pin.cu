      + 2 * DSV4_N_WARPS * HPB * (int)sizeof(float)                   // sm_reduce
      + N_V_CHUNKS_LAUNCH * HPB * (int)sizeof(float)                  // sm_w_head_sc
      + 2 * HPB * (DSV4_BI + 16);                                     // sm_w_fp8 ×2 (vc parity)
