"""
@file: lr_schedules.py
Created on 30.09.18
@project: crazy_ara_refactor
@author: queensgambit

This file contains the description of the common Advanced Learning rates.
Code is based on:
https://mxnet.incubator.apache.org/tutorials/gluon/learning_rate_schedules_advanced.html
"""

import copy
import math
import matplotlib.pyplot as plt


def plot_schedule(schedule_fn, iterations=1500, ylabel="Learning Rate", ylim=None):
    """ Make graph to follow the learning rate per iteration"""
    # Iteration count starting at 1
    iterations = [i + 1 for i in range(iterations)]
    plt.scatter(iterations, [schedule_fn(i) for i in iterations])
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    plt.show()


class TriangularSchedule:
    """TODO: docstring"""

    def __init__(self, min_lr, max_lr, cycle_length, inc_fraction=0.5):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        inc_fraction: fraction of iterations spent in increasing stage (float)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.inc_fraction = inc_fraction

    def __call__(self, iteration):
        if iteration <= self.cycle_length * self.inc_fraction:
            unit_cycle = iteration * 1 / (self.cycle_length * self.inc_fraction)
        elif iteration <= self.cycle_length:
            unit_cycle = (self.cycle_length - iteration) * 1 / (self.cycle_length * (1 - self.inc_fraction))
        else:
            unit_cycle = 0

        return (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr


class LinearWarmUp:
    """TODO: docstring"""

    def __init__(self, schedule, start_lr, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        start_lr: learning rate used at start of the warm-up (float)
        length: number of iterations used for the warm-up (int)
        """
        self.schedule = schedule
        self.start_lr = start_lr
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.finish_lr = copy.copy(schedule)(0)
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.length:
            return iteration * (self.finish_lr - self.start_lr) / (self.length) + self.start_lr
        return self.schedule(iteration - self.length)


class CyclicalSchedule:
    """TODO: docstring"""

    def __init__(self, schedule_class, cycle_length, cycle_length_decay=1, cycle_magnitude_decay=1, **kwargs):
        """
        schedule_class: class of schedule, expected to take `cycle_length` argument
        cycle_length: iterations used for initial cycle (int)
        cycle_length_decay: factor multiplied to cycle_length each cycle (float)
        cycle_magnitude_decay: factor multiplied learning rate magnitudes each cycle (float)
        kwargs: passed to the schedule_class
        """
        self.schedule_class = schedule_class
        self.length = cycle_length
        self.length_decay = cycle_length_decay
        self.magnitude_decay = cycle_magnitude_decay
        self.kwargs = kwargs

    def __call__(self, iteration):
        cycle_idx = 0
        cycle_length = self.length
        idx = self.length
        while idx <= iteration:
            cycle_length = math.ceil(cycle_length * self.length_decay)
            cycle_idx += 1
            idx += cycle_length

        schedule = self.schedule_class(cycle_length=cycle_length, **self.kwargs)
        return schedule(iteration - idx + cycle_length) * self.magnitude_decay ** cycle_idx


class CosineAnnealingSchedule:
    """TODO: docstring"""

    def __init__(self, min_lr, max_lr, cycle_length):
        """
        min_lr: lower bound for learning rate (float)
        max_lr: upper bound for learning rate (float)
        cycle_length: iterations between start and finish (int)
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length

    def __call__(self, iteration):
        if iteration <= self.cycle_length:
            unit_cycle = (1 + math.cos(iteration * math.pi / self.cycle_length)) / 2
            return (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return self.min_lr


class LinearCoolDown:
    """TODO: docstring"""

    def __init__(self, schedule, finish_lr, start_idx, length):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        finish_lr: learning rate used at end of the cool-down (float)
        start_idx: iteration to start the cool-down (int)
        length: number of iterations used for the cool-down (int)
        """
        self.schedule = schedule
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.start_lr = copy.copy(self.schedule)(start_idx)
        self.finish_lr = finish_lr
        self.start_idx = start_idx
        self.finish_idx = start_idx + length
        self.length = length

    def __call__(self, iteration):
        if iteration <= self.start_idx:
            return self.schedule(iteration)
        if iteration <= self.finish_idx:
            return (iteration - self.start_idx) * (self.finish_lr - self.start_lr) / self.length + self.start_lr
        return self.finish_lr


class OneCycleSchedule:
    """TODO: docstring"""

    def __init__(self, start_lr, max_lr, cycle_length, cooldown_length=0, finish_lr=None):
        """
        start_lr: lower bound for learning rate in triangular cycle (float)
        max_lr: upper bound for learning rate in triangular cycle (float)
        cycle_length: iterations between start and finish of triangular cycle: 2x 'stepsize' (int)
        cooldown_length: number of iterations used for the cool-down (int)
        finish_lr: learning rate used at end of the cool-down (float)
        """
        if cooldown_length > 0 and finish_lr is None:
            raise ValueError("Must specify finish_lr when using cooldown_length > 0.")
        if cooldown_length == 0 and finish_lr:
            raise ValueError("Must specify cooldown_length > 0 when using finish_lr.")

        finish_lr = finish_lr if cooldown_length > 0 else start_lr
        schedule = TriangularSchedule(min_lr=start_lr, max_lr=max_lr, cycle_length=cycle_length)
        self.schedule = LinearCoolDown(schedule, finish_lr=finish_lr, start_idx=cycle_length, length=cooldown_length)

    def __call__(self, iteration):
        return self.schedule(iteration)


class OneCycleMomentumSchedule:
    """TODO: docstring"""

    def __init__(self, start_momentum, max_momentum, cycle_length, warmup_length=0, finish_momentum=None):
        """
        start_lr: lower bound for learning rate in triangular cycle (float)
        max_lr: upper bound for learning rate in triangular cycle (float)
        cycle_length: iterations between start and finish of triangular cycle: 2x 'stepsize' (int)
        cooldown_length: number of iterations used for the cool-down (int)
        finish_lr: learning rate used at end of the cool-down (float)
        """
        if warmup_length > 0 and finish_momentum is None:
            raise ValueError("Must specify finish_lr when using cooldown_length > 0.")
        if warmup_length == 0 and finish_momentum:
            raise ValueError("Must specify cooldown_length > 0 when using finish_lr.")

        finish_lr = finish_momentum if warmup_length > 0 else start_momentum
        schedule = TriangularSchedule(min_lr=start_momentum, max_lr=max_momentum, cycle_length=cycle_length)
        self.schedule = LinearCoolDown(schedule, finish_lr=finish_lr, start_idx=cycle_length, length=warmup_length)

    def __call__(self, iteration):
        return self.schedule(iteration)


class MomentumSchedule:
    """TODO: docstring"""

    def __init__(self, lr_schedule, min_lr, max_lr, min_momentum, max_momentum):
        self.lr_schedule = lr_schedule
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum

    def __call__(self, iteration):
        perc = (self.lr_schedule(iteration) - self.min_lr) / (self.max_lr - self.min_lr)  # calculate percentage factor
        # invert the percentage factor and apply it
        return self.max_momentum - perc * (self.max_momentum - self.min_momentum)


class ConstantSchedule:
    """
    Constant schedule which return the same value for every iteration.
    This can be used to comply with the same program code scheme as other schedulers.
    """

    def __init__(self, lr):
        """
        lr: Constant learning rate
        """
        self.lr = lr

    def __call__(self, iteration):

        return self.lr
