import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="./config", config_name="config")
def my_app(cfg: DictConfig):
    # print(cfg)
    pass


if __name__ == "__main__":
    my_app()
