import numpy as np
import torch
import torch.nn as nn

from typing import Iterable, Optional, Sequence, Tuple, Union
from src.QDDPM_torch import DiffusionModel, QDDPM

ZERO_NORM_THRESHOLD = 1e-12


def amplitude_encode_structure(descriptor: Union[Sequence[float], torch.Tensor], n: int) -> torch.Tensor:
    """
    Map a crystal structure descriptor into a normalized amplitude vector that
    fits the 2**n computational basis states used by QDDPM. Descriptors shorter
    than 2**n are zero-padded; longer descriptors are rejected to preserve
    reversibility during inverse generation.
    Returns a complex64 tensor of shape (2**n,). If the descriptor norm is below
    ZERO_NORM_THRESHOLD an unnormalized zero vector is returned.
    """
    target_dim = 2 ** n
    raw = torch.as_tensor(descriptor, dtype=torch.float32).flatten()
    if raw.numel() > target_dim:
        raise ValueError(f"Descriptor of length {raw.numel()} exceeds available amplitudes ({target_dim}).")

    padded = torch.zeros(target_dim, dtype=torch.complex64)
    padded[: raw.numel()] = raw.to(torch.complex64)
    norm = torch.linalg.norm(padded)
    if norm < ZERO_NORM_THRESHOLD:
        return padded
    return padded / norm


class CrystalInverseQDDPM(nn.Module):
    """
    A thin convenience wrapper that repurposes the QDDPM backbone for inverse
    design of crystalline materials. It provides:
      1) amplitude encoding utilities for structural descriptors
      2) forward diffusion of encoded structures
      3) backward sampling to recover candidate crystal descriptors
    The implementation intentionally mirrors the existing QDDPM API to keep
    integration with notebooks straightforward.
    """

    def __init__(self, n: int, na: int, T: int, L: int):
        super().__init__()
        self.n = n
        self.na = na
        self.T = T
        self.L = L
        self.backbone = QDDPM(n, na, T, L)
        self.n_tot = self.backbone.n_tot

    def encode_structures(self, descriptors: Iterable[Union[Sequence[float], torch.Tensor]]) -> torch.Tensor:
        """
        Convert a batch of crystal descriptors into normalized quantum states.
        Descriptors can be any real-valued sequences (e.g., concatenated lattice
        parameters, fractional coordinates, and atomic numbers).
        Returns:
            tensor of shape (batch_size, 2**n) with complex64 amplitudes.
        """
        encoded = [amplitude_encode_structure(d, self.n) for d in descriptors]
        return torch.stack(encoded)

    def diffuse_structures(
        self,
        encoded_structures: torch.Tensor,
        diff_hs: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """
        Apply the forward scrambling circuit to the encoded crystal structures,
        producing the noisy targets used by the backward denoising network.
        A fresh DiffusionModel is created to match the current batch size,
        which can differ between training and inference calls.
        Args:
            encoded_structures: tensor of shape (batch_size, 2**n) holding
                                 amplitude-encoded crystal descriptors.
            diff_hs: tensor of shape (T,) controlling diffusion circuit scaling.
            seed: PRNG seed forwarded to the scrambling circuit.
        Returns:
            tensor of shape (batch_size, 2**n) after forward diffusion.
        """
        diffusion = DiffusionModel(self.n, self.T, encoded_structures.shape[0])
        return diffusion.set_diffusionData_t(self.T, encoded_structures, diff_hs, seed)

    def inverse_generate(
        self,
        params_tot: Union[torch.Tensor, np.ndarray],
        batch_size: int,
        seed: int = 0,
        noisy_inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the backward denoising process to propose new crystal descriptors.
        Args:
            params_tot: learned circuit parameters for each backward step with
                        shape (T, 2*self.L*self.n_tot), matching the
                        constructor arguments.
            batch_size: number of samples to generate
            seed: randomness for Haar state initialization
            noisy_inputs: optional custom starting states at t = T; if omitted,
                          Haar random states are used.
        Returns:
            generated_states: tensor of shape (batch_size, 2**n) with complex
                              amplitudes on data qubits
            probabilities: tensor of shape (batch_size, 2**n) containing
                           measurement probabilities for each computational basis
        """
        if isinstance(params_tot, torch.Tensor):
            params_tot = params_tot.detach().cpu().numpy()

        if noisy_inputs is None:
            noisy_inputs = self.backbone.HaarSampleGeneration(batch_size, seed)

        # backDataGeneration returns (T+1, batch_size, 2**(n+na)) stacked over diffusion steps
        states = self.backbone.backDataGeneration(noisy_inputs, params_tot, batch_size)
        if states.dim() != 3:
            raise ValueError(
                f"Unexpected state tensor shape from backDataGeneration; expected (T+1, batch_size, 2**(n+na)), got {tuple(states.shape)}."
            )
        # states are filled in reverse time order, so index 0 is the denoised output at t=0
        generated_states = states[0, :, : 2 ** self.n]
        probabilities = torch.abs(generated_states) ** 2
        return generated_states, probabilities
