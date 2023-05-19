import hydra
from omegaconf import OmegaConf

from ssac.trainer import Trainer


@hydra.main(version_base=None, config_path="ssac/configs", config_name="config")
def experiment(cfg):
    print(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    with Trainer(cfg, make_env) as trainer:
        trainer.train()


if __name__ == "__main__":
    experiment()
