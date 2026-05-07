from src.models.load_model import load_model
from src.unified_predictor import run_unified_prediction


def run(args, yaml_cfg, model_cfg):
    model = load_model(args, model_cfg)
    run_unified_prediction(model, yaml_cfg, args)
    return model
