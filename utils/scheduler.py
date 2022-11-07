import torch
import math
import numpy as np

# temp_scheduler = scheduler('cosine_warmup', args.temperature_min, args.temperature_min, args.epochs, len(train_loader), warmup_epochs=args.temp_warmup_epochs, start_warmup_value=args.temperature)
# kl_scheduler = scheduler('cosine_warmup', args.kl_div_loss_weight_max, args.kl_div_loss_weight_max, args.epochs, len(train_loader), warmup_epochs=args.kl_warmup_epochs)


# self.temp_scheduler = scheduler('cosine_warmup', self.args.temperature_min, self.args.temperature_min, self.args.epochs,
#                                 len(train_loader), warmup_epochs=self.args.temp_warmup_epochs, start_warmup_value=self.args.temperature)


# self.temp_scheduler = scheduler('cosine_warmup', self.temperature_min, self.temperature_min,
#                                 self.args.epochs, len(train_loader),
#                                 warmup_epochs=self.args.temp_warmup_epochs,
#                                 start_warmup_value=self.temperature)


def scheduler(scheduler, base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    # scheduler, cosine_warmup
    # base_value, 1e-4
    # final_value, 1e-6
    # epochs, total epoch
    # niter_per_ep, len(train_loader)
    # warmup_epochs=0,
    # start_warmup_value=0
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)

    if scheduler == "cosine_warmup":
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    elif scheduler == 'reciprocal_sqr_root':
        schedule = base_value * math.sqrt(warmup_iters) / np.sqrt(iters)
    # elif scheduler == 'sigmoid':
    #     schedule =
    else:
        NotImplementedError

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule