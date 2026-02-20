import torch
import torch.nn as nn

from typing import Iterable, Sequence, Union

from src.QDDPM_torch import DiffusionModel, QDDPM


def _amplitude_encode_structure(descriptor: Union[Sequence[float], torch.Tensor], n: int) -> torch.Tensor:
    """
    Map a crystal structure descriptor into a normalized amplitude vector that
    fits the 2**n computational basis states used by QDDPM. Descriptors shorter
    than 2**n are zero-padded; longer descriptors are rejected to preserve
    reversibility during inverse generation.
    """
    target_dim = 2 ** n
    raw = torch.as_tensor(descriptor, dtype=torch.float32).flatten()
    if raw.numel() > target_dim:
        raise ValueError(f"Descriptor of length {raw.numel()} exceeds available amplitudes ({target_dim}).")

    padded = torch.zeros(target_dim, dtype=torch.complex64)
    padded[: raw.numel()] = raw.to(torch.complex64)
    norm = torch.linalg.norm(padded)
    if norm == 0:
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

    def encode_structures(self, descriptors: Iterable[Union[Sequence[float], torch.Tensor]]) -> torch.Tensor:
        """
        Convert a batch of crystal descriptors into normalized quantum states.
        Descriptors can be any real-valued sequences (e.g., concatenated lattice
        parameters, fractional coordinates, and atomic numbers).
        """
        encoded = [_amplitude_encode_structure(d, self.n) for d in descriptors]
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
        """
        diffusion = DiffusionModel(self.n, self.T, encoded_structures.shape[0])
        return diffusion.set_diffusionData_t(self.T, encoded_structures, diff_hs, seed)

    def inverse_generate(
        self,
        params_tot,
        batch_size: int,
        seed: int = 0,
        noisy_inputs: torch.Tensor = None,
    ):
        """
        Run the backward denoising process to propose new crystal descriptors.
        Args:
            params_tot: learned circuit parameters for each backward step
            batch_size: number of samples to generate
            seed: randomness for Haar state initialisation
            noisy_inputs: optional custom starting states at t = T; if omitted,
                          Haar random states are used.
        Returns:
            generated_states: complex amplitudes on data qubits
            probabilities: measurement probabilities for each computational basis
        """
        if isinstance(params_tot, torch.Tensor):
            params_tot = params_tot.detach().cpu().numpy()

        if noisy_inputs is None:
            noisy_inputs = self.backbone.HaarSampleGeneration(batch_size, seed)

        states = self.backbone.backDataGeneration(noisy_inputs, params_tot, batch_size)
        generated_states = states[0, :, : 2 ** self.n]
        probabilities = torch.abs(generated_states) ** 2
        return generated_states, probabilities
