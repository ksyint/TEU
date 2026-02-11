import torch
import torch.nn.functional as F


def dpo_loss_sigmoid(policy_preferred_logps, policy_rejected_logps,
                     reference_preferred_logps, reference_rejected_logps, beta):
    preferred_rewards = beta * (policy_preferred_logps - reference_preferred_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    logits = preferred_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    preferred_reward_mean = preferred_rewards.detach().mean()
    rejected_reward_mean = rejected_rewards.detach().mean()
    return loss, preferred_reward_mean, rejected_reward_mean


def dpo_loss_hinge(policy_preferred_logps, policy_rejected_logps,
                   reference_preferred_logps, reference_rejected_logps, beta, margin=0.1):
    preferred_rewards = beta * (policy_preferred_logps - reference_preferred_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    loss = F.relu(margin - (preferred_rewards - rejected_rewards)).mean()
    return loss, preferred_rewards.detach().mean(), rejected_rewards.detach().mean()


def dpo_loss_ipo(policy_preferred_logps, policy_rejected_logps,
                 reference_preferred_logps, reference_rejected_logps, beta):
    log_ratio_preferred = policy_preferred_logps - reference_preferred_logps
    log_ratio_rejected = policy_rejected_logps - reference_rejected_logps
    loss = ((log_ratio_preferred - log_ratio_rejected) - 1 / (2 * beta)) ** 2
    loss = loss.mean()
    return loss, log_ratio_preferred.detach().mean(), log_ratio_rejected.detach().mean()


def compute_log_probs(noise_scheduler, model_pred, noise, timesteps):
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        raise NotImplementedError("v_prediction not supported for DPO yet")
    else:
        target = noise
    mse = -0.5 * F.mse_loss(model_pred.float(), target.float(), reduction="none")
    log_probs = mse.mean(dim=list(range(1, len(mse.shape))))
    return log_probs


def get_dpo_loss_fn(loss_type="sigmoid"):
    if loss_type == "sigmoid":
        return dpo_loss_sigmoid
    elif loss_type == "hinge":
        return dpo_loss_hinge
    elif loss_type == "ipo":
        return dpo_loss_ipo
    else:
        raise ValueError(f"Unknown DPO loss type: {loss_type}")
