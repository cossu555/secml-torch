import torch
from secmlt.optimization.constraints import LpConstraint
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from secmlt.trackers import (
    LossTracker,
    PredictionTracker,
)

device = 'cpu' if torch.cuda.is_available() else 'cpu'

def unavailable_gradients_indicator(model, attack, dataloader) -> bool:
    for x, y in dataloader:
        if device is not None:
            x = x.to(device); y = y.to(device)
        target = y if attack.y_target is None else torch.full_like(y, attack.y_target)
        x_ = x.detach().clone().requires_grad_(True)
        try:
            _, losses = attack.forward_loss(model, x_, target)
            (grad,) = torch.autograd.grad(losses.sum(), x_, allow_unused=True)
        except Exception:
            return True
        if grad is None or (not torch.isfinite(grad).all()):
            return True
        g_inf = grad.detach().abs().flatten(1).amax(dim=1)
        if (g_inf == 0).any().item():
            return True
    return False

def _perturb_loader(data_loader: DataLoader, radius: float) -> DataLoader:
    x_perturb, y_original = [], []
    for x, y in data_loader:
        x = x.to(device);y = y.to(device)
        x_pert = x + (torch.rand_like(x) - 0.5) * radius
        x_perturb.append(x_pert);y_original.append(y)
    x_all = torch.cat(x_perturb, dim=0);y_all = torch.cat(y_original, dim=0)
    return DataLoader(
        TensorDataset(x_all, y_all),
        batch_size=data_loader.batch_size,
        shuffle=False,
        num_workers=0,
    )

def unstable_predictions_indicator(attack,model,data_loader: DataLoader,gamma=10,radius=8/255) -> float:

    ce_unperturbed = []
    for x,y in data_loader:
        target = y if attack.y_target is None else torch.full_like(y, attack.y_target)
        scores, ce = attack.forward_loss(model, x, target)
        ce_unperturbed.append(ce)
    ce_unperturbed = torch.stack(ce_unperturbed, dim=0)

    results = []
    for _ in range(gamma):
        perturbed_loader = _perturb_loader(data_loader, radius)
        ce_perturbed = []
        for x,y in perturbed_loader:
            target = y if attack.y_target is None else torch.full_like(y, attack.y_target)
            _ , ce = attack.forward_loss(model, x, target)
            ce_perturbed.append(ce)
        ce_perturbed = torch.stack(ce_perturbed, dim=0)

        tmp = (ce_unperturbed - ce_perturbed).abs()
        tmp = tmp / ((ce_unperturbed).abs() + torch.finfo(ce_unperturbed.dtype).eps)
        results.append(tmp)
    metric = torch.stack(results, dim=0).mean().clip(max=1)
    return metric.item()

def silent_success_indicator(P_hist: torch.Tensor,y_adv: torch.Tensor,y0: torch.Tensor,y_target=None,) -> torch.Tensor:
    y0 = y0.to(device)
    y_adv = torch.as_tensor(y_adv, device=device)

    if y_target is None:
        returned_fail = (y_adv == y0)
        found_adv_along_path = (P_hist != y0[:, None]).any(dim=1)
    else:
        y_target = torch.as_tensor(y_target, device=device)
        if y_target.ndim == 0:
            y_target = y_target.expand_as(y0)
        else:
            y_target = y_target.to(device)

        returned_fail = (y_adv != y_target)
        found_adv_along_path = (P_hist == y_target[:, None]).any(dim=1)

    return returned_fail & found_adv_along_path

def incomplete_optimization_indicator(loss_hist: torch.Tensor, k: int = 10, mu: float = 0.01) -> float:
    if loss_hist.numel() == 0:
        return 0.0
    if loss_hist.ndim == 1:  # [T] -> [1, T]
        loss_hist = loss_hist.unsqueeze(0)
    row_min = loss_hist.min(dim=1, keepdim=True).values
    row_max = loss_hist.max(dim=1, keepdim=True).values
    denom = (row_max - row_min).clamp_min(1e-12)
    L = (loss_hist - row_min) / denom
    Lhat = torch.cummin(L, dim=1).values
    last = Lhat[:, -k:]
    D = last.max(dim=1).values - last.min(dim=1).values
    I4 = (D >= mu).float()
    return I4.mean().item()

def transfer_failure_indicator(transfer_scores, pred) -> float:
    I = (pred != transfer_scores).float()
    return I.mean().item()

def _reset_trackers(obj):
    if obj is None:
        return
    if isinstance(obj, (list, tuple)):
        for t in obj:
            _reset_trackers(t)
        return
    if hasattr(obj, "trackers"):
        _reset_trackers(obj.trackers)
    if hasattr(obj, "reset"):
        obj.reset()

def unconstrained_attack_failure_indicator(attack, model, dataloader) -> float:
    man = getattr(attack, "manipulation_function", None)

    old_radii = []
    if man is not None and hasattr(man, "perturbation_constraints"):
        for c in man.perturbation_constraints:
            if isinstance(c, LpConstraint):
                old_radii.append((c, c.radius.clone()))
                c.radius = torch.full_like(c.radius, float("inf"))

    try:
        _reset_trackers(attack.trackers[0].trackers)
        ds_unc = attack(model, dataloader)

        P_tracker = next(
            tr.get() for tr in attack.trackers[0].trackers
            if isinstance(tr, PredictionTracker)
        )

        y_targeted = attack.y_target
        y_o = []
        y_unc = []
        fails = 0
        total = 0

        for x_unc_b, y_o_b in ds_unc:
            y_unc_b = model.decision_function(x_unc_b).argmax(dim=1)

            y_unc_b = y_unc_b.to(device)
            y_o_b = y_o_b.to(device)

            y_o.append(y_o_b)
            y_unc.append(y_unc_b)

            if y_targeted is None:
                trigger = (y_unc_b == y_o_b)
            else:
                y_t_b = torch.as_tensor(y_targeted, device=device)
                if y_t_b.ndim == 0:
                    y_t_b = y_t_b.expand_as(y_o_b)
                trigger = (y_unc_b != y_t_b)

            fails += int(trigger.sum().item())
            total += trigger.numel()

        y_o = torch.cat(y_o, dim=0)
        y_unc = torch.cat(y_unc, dim=0)

        if silent_success_indicator(P_tracker, y_unc, y_o, y_targeted).any().item():
            return -1.0

        return fails / total if total > 0 else 0.0

    finally:
        for c, old_radius in old_radii:
            c.radius = old_radius

REJECT_CLASSES = [-1, 10]

def attack_fails(adv_pred, y0, target_label=None, transfer_scores=None) -> float:
    adv_pred = torch.as_tensor(adv_pred, device = device)
    y0 = torch.as_tensor(y0,device = device)

    if target_label is None:
        untarget_fail = adv_pred == y0
        targeted_fail = torch.zeros_like(untarget_fail)
    else:
        target_label = torch.as_tensor(target_label, device=device)
        targeted_fail = adv_pred != target_label
        untarget_fail = torch.zeros_like(targeted_fail)

    rejection_failure = adv_pred == -1

    if transfer_scores is not None:
        transfer_scores = torch.as_tensor(transfer_scores, device=device)
        reject_set = torch.as_tensor(REJECT_CLASSES, device=device)
        rejection_failure = rejection_failure | torch.isin(transfer_scores, reject_set)

    fail = untarget_fail | targeted_fail | rejection_failure
    return float(fail.float().mean().item())

def compute_indicators(attack,model,dataloader,surrogate_model=None,y_target=None,gamma=50,radius = 8/255):
    #pre-processing:
    print("starting pre-processing...")

    pred_tracker_obj = next(tr for tr in attack.trackers[0].trackers if isinstance(tr, PredictionTracker))
    loss_tracker_obj = next(tr for tr in attack.trackers[0].trackers if isinstance(tr, LossTracker))

    if pred_tracker_obj.get().numel() != 0:
        pred_tracker_obj.reset()
    if loss_tracker_obj.get().numel() != 0:
        loss_tracker_obj.reset()

    # ---------------------------
    # RUN 1: attack on real model
    # ---------------------------
    ds_adv = attack(model, dataloader)

    y_0_batched = []
    y_model_adv_batched = []

    for x, y in ds_adv:
        y_0_batched.append(y)
        y_model_adv_batched.append(model.decision_function(x).argmax(dim=1))

    y_0 = torch.cat(y_0_batched, dim=0)
    y_model_adv = torch.cat(y_model_adv_batched, dim=0)

    # save trackers from the REAL-MODEL run
    P_tracker = pred_tracker_obj.get()
    L_tracker = loss_tracker_obj.get()

    # --------------------------------
    # RUN 2: attack on surrogate model
    # --------------------------------
    if surrogate_model is not None:
        if pred_tracker_obj.get().numel() != 0:
            pred_tracker_obj.reset()
        if loss_tracker_obj.get().numel() != 0:
            loss_tracker_obj.reset()

        ds_adv_transfer = attack(surrogate_model, dataloader)

        y_surrogate_adv_batched = []
        y_model_transfer_batched = []

        for x, y in ds_adv_transfer:
            y_surrogate_adv_batched.append(surrogate_model.decision_function(x).argmax(dim=1))
            y_model_transfer_batched.append(model.decision_function(x).argmax(dim=1))

        y_surrogate_adv = torch.cat(y_surrogate_adv_batched, dim=0)
        y_model_transfer = torch.cat(y_model_transfer_batched, dim=0)
    else:
        y_surrogate_adv = None
        y_model_transfer = None

    print("end pre-processing\n")

    #trackers:
    print("obtaining trackers...")
    P_tracker = next(tr.get() for tr in attack.trackers[0].trackers if isinstance(tr, PredictionTracker))
    L_tracker = next(tr.get() for tr in attack.trackers[0].trackers if isinstance(tr, LossTracker))
    print("trackers obtained\n")

    #indicators:
    print("starting unavailable_gradients_indicator...")
    I1 = unavailable_gradients_indicator(model,attack,dataloader)
    print("end unavailable_gradients_indicator\n")

    print("staring unstable_predictions_indicator...")
    I2 = unstable_predictions_indicator(attack,model,dataloader,gamma=gamma, radius = radius)
    print("end unstable_predictions_indicator\n")

    print("staring silent_success_indicator...")
    I3 = bool(silent_success_indicator(P_tracker, y_model_adv, y_0, y_target).any().item())
    print("end silent_success_indicator\n")

    print("starting incomplete_optimization_indicator...")
    I4 = incomplete_optimization_indicator(L_tracker)
    print("end incomplete_optimization_indicator\n")

    if surrogate_model is not None:
        print("start transfer_failure_indicator..")
        I5 = transfer_failure_indicator(y_model_transfer, y_surrogate_adv)
        print("end transfer_failure_indicator\n")
    else: I5=None; print("no surrogate\n")

    print("starting unconstrained_attack_failure_indicator..")
    I6 = unconstrained_attack_failure_indicator(attack,model,dataloader)
    print("end unconstrained_attack_failure_indicator\n")

    print("starting Attack_fails...")
    Attack_fails = attack_fails(y_model_adv, y_0, target_label=attack.y_target)
    print("end Attack_fails\n")

    df = pd.DataFrame(data={
        'attack_fails': Attack_fails,
        'unavailable_gradients': I1,
        'unstable_predictions': I2,
        'silent_success': I3,
        'incomplete_optimization': I4,
        'transfer_failure': I5,
        'unconstrained_attack_failure': I6,
    }, index=[0])

    return df