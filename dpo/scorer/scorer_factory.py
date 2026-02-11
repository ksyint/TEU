from dpo.scorer.aesthetic_scorer import AestheticScorer
from dpo.scorer.clip_scorer import CLIPScorer
from dpo.scorer.hps_scorer import HPSScorer


def create_scorer(cfg, device="cpu"):
    scorer_cfg = cfg.get("scorer", {})
    scorer_type = scorer_cfg.get("type", "clip")
    clip_model_name = scorer_cfg.get("clip_model_name", "openai/clip-vit-large-patch14")

    if scorer_type == "aesthetic":
        model_path = scorer_cfg.get("aesthetic_model_path", None)
        return AestheticScorer(
            clip_model_name=clip_model_name,
            model_path=model_path,
            device=device,
        )
    elif scorer_type == "clip":
        return CLIPScorer(
            clip_model_name=clip_model_name,
            device=device,
        )
    elif scorer_type == "hps":
        model_path = scorer_cfg.get("hps_model_path", None)
        return HPSScorer(
            clip_model_name=clip_model_name,
            model_path=model_path,
            device=device,
        )
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}")
