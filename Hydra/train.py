# train.py
import hydra
from omegaconf import DictConfig # OmegaConf 是 Hydra 的底层配置引擎

@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig):
    print(f"Learning Rate: {cfg.learning_rate}, Batch Size: {cfg.batch_size}")

if __name__ == "__main__":
    train()