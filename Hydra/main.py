import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", )
def my_app(cfg: DictConfig):
    print(f"环境: {cfg.env.log_level}")
    print(f"数据库: {cfg.db.host}")
    print(f"模型: {cfg.model.type}")

if __name__ == "__main__":
    my_app()