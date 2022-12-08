# --------------------------------------------------------
# Copyright. All Rights Reserved
# --------------------------------------------------------

# Most likely some version of this repo is the source:
# https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py

# General impression of this module is that it is quite buggy, and poorly maintained.
# - Buggy behavior when after_scheduler is not None. https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/26
# - The implementation currently requires calling .step() once before any optimizer steps are taken. The author of this function was probably not aware that pytorch does this implicitly in any case, and therefore wrote the scheduler assuming this practice, despite going against pytorch's recommendation, consequently generating a warning. The work-around is to simply take an optimizer step with zero gradient at the very beginning, followed by a manual initial scheduler step, followed by the training loop.
# - For the GH repo version, the above assumption also seems to cause some issues for after_scheduler == ReduceLROnPlateau(), and consequently this case is treated separately, which should not be needed. Furthermore, the separate function has a reported bug (https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/19).
# - One would rather rewrite everything, with greater care, better consistency with pytorch, and more compactly / readable.
# - Ideally, the whole thing could be implemented with pytorch LR scheduler primitives. E.g. LinearLR seems very appropriate. Also, one can chain schedulers with SequentialLR.
#   https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
#   https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html

from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0.
            if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self,
                 optimizer,
                 multiplier,
                 total_epoch,
                 after_scheduler=None):
        # Buggy behavior reported (but affecting only when multiplier > 1): https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/26
        if multiplier > 1:
            assert after_scheduler is None, "after_scheduler not accepted, due to buggy behavior as of now."
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError(
                'multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        # The super class _LRScheduler() initialization will result in self.last_epoch being incremented from -1 to 0:
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    # self.after_scheduler._last_lr = self._last_lr # Might make sense..?
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr() # Wrong, not updated. Does however only affect the case when self.after_scheduler is not None.
                # return self.after_scheduler.get_lr() # More sensible, and might solve https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/26
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            # Special case, linearly ramp up LR from 0 to base_lr
            # NOTE: If not having ever called .step() manually before the first actual gradient update (and implied .get_lr() call), self.last_epoch will still be 0, and the LR retrieved will also be 0.
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            # Regular case, linearly ramp up LR from base_lr to multiplie4r*base_lr
            return [
                base_lr *
                ((self.multiplier - 1.) * self.last_epoch / self.total_epoch +
                 1.) for base_lr in self.base_lrs
            ]

    def step(self, epoch=None, metrics=None):
        # This method will first be called by _LRScheduler._initial_step(), in turn called by _LRScheduler.__init__(). As a result, self.last_epoch is incremented from -1 to 0.
        # The next time .step() is called (by the user), self.last_epoch will increment from 0 to 1.
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr() # Weird..?
            # Better: (?)
            # self._last_lr = self.after_scheduler.get_lr()
            # self._last_epoch += 1
        else:
            # Why not call this even in the after_scheduler phase?
            return super(GradualWarmupScheduler, self).step(epoch)
