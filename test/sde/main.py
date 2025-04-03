from my_utils.sde.diffusion_sde import VPSDE
from my_utils.sde.predictor import ReverseDiffusionPredictor
from my_utils.sde.corrector import LangevinCorrector
import hydra
from omegaconf import OmegaConf, DictConfig
from my_utils.sde.simpler import get_pc_sampler


@hydra.main(version_base=None, config_name=None, config_path=None)
def main(cfg: DictConfig):
    cfg = dict()
    sde = VPSDE(beta_min=0.1, beta_max=20, N=cfg.model.num_scales)
    sampling_eps = 1e-3
    img_size = cfg.data.image_size
    channels = cfg.data.num_channels
    shape = (cfg.generate.batch_size, channels, img_size, img_size)
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    snr = 0.16
    n_steps = 1
    use_probability_flow = False

    sampling_fn = get_pc_sampler(
        sde,
        shape,
        predictor,
        corrector,
        inverse_scaler,
        snr,
        n_steps=n_steps,
        probability_flow=probability_flow,
        continuous=cfg.training.continuous,
        eps=sampling_eps,
        device=cfg.device,
    )


if __name__ == "__main__":
    main()
