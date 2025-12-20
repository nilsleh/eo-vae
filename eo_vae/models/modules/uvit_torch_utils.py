# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
import torch.nn as nn
from einops import rearrange

ACTIVATIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'identity': nn.Identity,
}

###############################################################
# Wrappers / Containers
###############################################################


class Ref:
    """A class to store a reference, usefull to reference a torch module without including it into torch operations"""

    class Empty:
        pass

    def __init__(self, value=None):
        self._value = value

    def check(self):
        if self._value is Ref.Empty:
            raise ValueError(
                "Ref is empty. It could be a copied ref that hasn't been re-initialized."
            )

    def __call__(self, *args, **kwargs):
        self.check()
        return self._value(*args, **kwargs)

    def __getattr__(self, attr):
        if attr == 'value':
            self.check()
            return self._value
        elif attr == '_value':
            raise AttributeError('_value access but not defined!')
        else:
            return getattr(self._value, attr)

    def __setattr__(self, key, value):
        if key in ['value', '_value']:
            super().__setattr__('_value', value)
        else:
            self.check()
            setattr(self._value, key, value)

    def __getitem__(self, item):
        self.check()
        return self._value[item]

    def __repr__(self):
        return f'Ref({self._value})'

    def set(self, value):
        self._value = value

    def clear(self):
        self._value = Ref.Empty


class Frozen(nn.Module):
    """Wraps a pre-trained module, and freezes it.
    If allow_grad is True, gradients can computed through the module, but the weights are not updated.
    Parameters will be counted by count_parameters, but not by model.parameters().
    The wrapped module is not fully displayed, only its name.
    """

    def __init__(self, module, accelerator=None, allow_grad=True):
        super().__init__()
        freeze_model(module)
        # Prepare the model, but ensure accelerate doesn't keep track of it afterward
        if accelerator is not None:
            n_models = accelerator._models
            module = accelerator.prepare(module)
            accelerator._models.pop()
            assert accelerator._models == n_models

        self._module = (module.eval(),)
        self.allow_grad = allow_grad

    def __getattr__(self, name):
        if name == 'module':
            return self._module[0]
        return getattr(self._module[0], name)

    def forward(self, *args, **kwargs):
        assert not self.module.training
        if not self.allow_grad:
            with torch.no_grad():
                return self.module(*args, **kwargs)
        return self.module(*args, **kwargs)

    def __repr__(self):
        m = self.module
        name = []
        while m is not None:
            name.append(m.__class__.__name__)
            if hasattr(m, 'module'):
                m = m.module
            elif hasattr(m, 'model'):
                m = m.model
            elif hasattr(m, '_orig_mod'):
                m = m._orig_mod
            else:
                m = None
        name = '/'.join(name)
        return f'Frozen({name})'


class FrozenCopyRef(Frozen):
    """Wraps a copy of a pre-trained module, and freezes it.
    The parameters of the copy are updated at every forward pass.
    If ema is not 0, the parameters are updated with an exponential moving average instead of being copied.
    """

    def __init__(self, accelerator, module, allow_grad=True, ema=0):
        copy_module = deepcopy(module)
        self.initial_module = Ref(module)
        self.ema = ema
        super().__init__(accelerator, copy_module, allow_grad)

    @torch.no_grad()
    def update_copy(self):
        # Share memory
        for base_param, frozen_param in zip(
            self.initial_module.value.parameters(),
            self.module.parameters(),
            strict=True,
        ):
            if self.ema:
                frozen_param.sub_((1 - self.ema) * (frozen_param - base_param))
            else:
                frozen_param.data[:] = base_param.data

    def forward(self, *args, **kwargs):
        self.update_copy()
        return super().forward(*args, **kwargs)


###############################################################
# Parameters & types
###############################################################


def mark_initialized(model):
    for m in model.modules():
        m._w_init = True


def freeze_model(model, freeze=True, mark_init='auto'):
    for param in model.parameters():
        param.requires_grad = not freeze
    if mark_init == 'auto':
        mark_init = freeze
    if mark_init:
        mark_initialized(model)


def show_model_parameters(model):
    n_params = count_parameters(model, trainable=False)
    n_trainable = count_parameters(model, trainable=True)
    print(f'{n_params} parameters ({n_trainable} trainable)')


def format_parameter_count(cnp):
    if cnp >= 2**40:
        return f'{cnp / (2**40):.1f}T'
    if cnp >= 2**30:
        return f'{cnp / (2**30):.1f}B'
    if cnp >= 2**20:
        return f'{cnp / (2**20):.1f}M'
    if cnp >= 2**10:
        return f'{cnp / (2**10):.1f}K'
    return f'{cnp}'


def count_parameters(model, trainable=False, exclude=None, as_int=False):
    if model is None:
        return 0

    model = unwrap(model)
    cnp = sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)

    if not trainable:  # If not trainable, add Frozen modules
        for m in model.modules():
            if isinstance(m, Frozen):
                cnp += count_parameters(m.module, as_int=True)
    if exclude:
        if not isinstance(exclude, (list, tuple)):
            exclude = [exclude]
        for subpart in exclude:
            cnp -= count_parameters(subpart, trainable=trainable, as_int=True)
    if as_int:
        return cnp
    return format_parameter_count(cnp)


###############################################################
# Misc
###############################################################


def ensure_1d_tokens(tensor):
    if tensor.ndim == 3:
        return tensor
    elif tensor.ndim == 4:
        return rearrange(tensor, 'b c h w -> b (h w) c')
    else:
        raise ValueError(f'Tensor must be 3D or 4D, got {tensor.ndim}D tensor')


def unwrap(model, unw_ema=False):
    unw = lambda m: unwrap(m, unw_ema=unw_ema)
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return unw(model.module)
    elif unw_ema and hasattr(model, 'model') and hasattr(model, 'ema'):
        return unw(model.model)
    elif model.__class__.__name__ == 'AcceleratedScheduler':
        return unw(model.scheduler)
    elif hasattr(model, '_orig_mod'):
        return unw(model._orig_mod)
    return model


###############################################################
# Random generators
###############################################################


def reproducible_rand(accelerator, generator: torch.Generator, shape: tuple, fn=None):
    fn = fn or torch.randn
    noise = [
        fn(shape, generator=generator, device=accelerator.device)
        for _ in range(accelerator.num_processes)
    ]
    noise = noise[accelerator.process_index]
    return noise
