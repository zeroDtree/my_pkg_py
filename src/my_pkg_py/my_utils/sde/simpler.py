import functools
from .predictor import Predictor, NonePredictor, ReverseDiffusionPredictor
from .corrector import Corrector, NoneCorrector, LangevinCorrector
from .sde import SDE
import torch
from typing import Tuple


def shared_predictor_update_fn(x, t, sde, score_fn, predictor, use_probability_flow):
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, use_probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, use_probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, score_fn, corrector, snr, n_steps):
    if corrector is None:
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


def get_pc_sampler(
    sde,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_correct_steps=1,
    use_probability_flow=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_correct_steps: An integer. The number of corrector steps per predictor update.
      use_probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn, sde=sde, predictor=predictor, probability_flow=use_probability_flow
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn, sde=sde, corrector=corrector, s=snr, n_steps=n_correct_steps
    )

    def pc_sampler(score_fn):
        """The PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, score_fn=score_fn)
                x, x_mean = predictor_update_fn(x, vec_t, score_fn=score_fn)

            return inverse_scaler(x_mean if denoise else x), sde.N * (n_correct_steps + 1)

    return pc_sampler
