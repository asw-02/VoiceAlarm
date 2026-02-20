#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio resampling utility for real-time streams.
"""

import torch
import torchaudio.transforms as T

class StatefulResampler:
    """Resamples audio stream chunks while maintaining overlap to avoid artifacts."""
    
    def __init__(self, orig_sr, target_sr, overlap=256, device="cpu"):
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.overlap = overlap
        self.device = device

        self.resampler = T.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        ).to(device)

        # Overlap Buffer
        self.prev_tail = torch.zeros(overlap, dtype=torch.float32, device=device)

    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Processes a single audio chunk.
        chunk: 1D torch.Tensor (float32), orig_sr
        return: 1D torch.Tensor (float32), target_sr
        """
        if chunk.dtype != torch.float32:
            chunk = chunk.float()

        # Append overlap
        chunk = torch.cat([self.prev_tail, chunk])

        with torch.no_grad():
            out = self.resampler(chunk)

        # Remember new overlap
        self.prev_tail = chunk[-self.overlap:].detach()

        return out