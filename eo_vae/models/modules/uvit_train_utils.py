# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

# from .optional_import import optional_import

# schedulefree = optional_import("schedulefree")

# from .main_utils import TaskState, ensure_path

###############################################################
# Initialization
###############################################################


@torch.no_grad()
def init_weights(
    model,
    method=None,
    nonlinearity='leaky_relu',
    init_embeds=False,
    scale=0.02,
    force=False,
    checkpoint=None,
    ckpt_module=None,
    ckpt_args=None,
    freeze=False,
):
    """Initialize weights of a model with a given method, when the model is not already initialized. Mark initialized modules with _w_init=True.
    method: str, default="auto"
        - "auto": use kaiming_uniform for conv and linear, and normal for embeddings
        - "kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "uniform", "normal"
        - float value: normal with the given scale
    method modifiers:
        - "+nonlinearity-leaky_relu", "+nonlinearity-relu", "+nonlinearity-tanh", "+nonlinearity-sigmoid"
        - "+scale-0.02", "+scale-0.1", etc.
        - "+conv-kaiming_uniform", "+conv-kaiming_normal", "+conv-xavier_uniform", "+conv-xavier_normal", "+conv-uniform", etc.

    """
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # If checkpoint is provided, load weights from the checkpoint
    if checkpoint is not None:
        n_mod = load_submodule(
            model, checkpoint, ckpt_module or '', **(ckpt_args or {})
        )
        mark_init_state(model, True)
        return n_mod

    # Parse method
    method = method.lower() if isinstance(method, str) else method
    if method in ['default', 'auto']:
        # Default initialization, mark initialized modules
        n_mod = mark_init_state(model, True)  # Keep default weights
        return n_mod

    if method in [None, 'none', 'skip']:
        # Don't initialize, module can be initialized by parent
        return

    if isinstance(method, (int, float)):
        scale = method
        method = 'normal'

    # Express modifiers
    conv_method = 'normal'  # default
    method_parts = method.split('+')
    for part in method_parts:
        if '-' in part:
            part, value = part.split('-', 1)
            if part == 'scale':
                scale = float(value)
            elif part == 'nonlinearity':
                nonlinearity = value
            elif part == 'conv':
                conv_method = value
            else:
                raise ValueError(f'Unknown part {part} in method {method}')
        else:
            method = part

    # Initialize weights function
    def init_layer_weights(l_method, layer_weight):
        if l_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(layer_weight, nonlinearity=nonlinearity)
        elif l_method == 'kaiming_normal':
            nn.init.kaiming_normal_(layer_weight, nonlinearity=nonlinearity)
        elif l_method == 'xavier_uniform':
            nn.init.xavier_uniform_(layer_weight)
        elif l_method == 'xavier_normal':
            nn.init.xavier_normal_(layer_weight)
        elif l_method == 'uniform':
            nn.init.uniform_(layer_weight, a=-scale, b=scale)
        elif l_method == 'normal':
            nn.init.normal_(layer_weight, mean=0.0, std=scale)
        elif l_method == 'zero':
            nn.init.zeros_(layer_weight)
        else:
            raise ValueError(f'Unknown weights initialization method {l_method}')

    # Initialize weights
    n_init = 0
    for m in model.modules():
        if (hasattr(m, '_w_init') and m._w_init) or force:
            continue
        if isinstance(m, nn.Conv2d):
            init_layer_weights(conv_method or method, m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init_layer_weights(method, m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            if init_embeds:
                nn.init.normal_(m.weight, mean=0)
                m.weight.data = (
                    nn.functional.normalize(m.weight.data, p=2, dim=-1) * scale
                )
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                m.bias.data.zero_()
            if m.weight is not None:
                m.weight.data.fill_(1.0)
        elif 'RMSNorm' in m.__class__.__name__:
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.data.fill_(1.0)
        else:
            m._w_init = True
            continue
        m._w_init = True
        n_init += 1
    return n_init


def mark_init_state(model, init_state):
    """Mark all modules of a model as initialized or not. Can be used to reinitialize the model, or protect weights of a submodule."""
    n_mod = 0
    for m in model.modules():
        if not hasattr(m, '_w_init') or m._w_init != init_state:
            n_mod += 1
        m._w_init = init_state
    return n_mod


@torch.no_grad()
def init_zero(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            m.weight.data.zero_()
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            m.weight.data.zero_()
        elif 'RMSNorm' in m.__class__.__name__:
            m.weight.data.zero_()
        elif hasattr(m, 'lambda1'):
            nn.init.zeros_(m.lambda1.data)
        m._w_init = True


# def load_submodule(model, weights_path, module_name, strict=True, accelerator=None, map_fn=None, default_load="ae_ema"):
#     warn = accelerator.warning if accelerator else lambda x: print("[WARNING", x)
#     assert isinstance(weights_path, (str, Path))
#     if len(Path(weights_path).parts) == 1:
#         model_type = "last"
#         if "@" in weights_path:
#             weights_path, model_type = str(weights_path).split("@", 1)
#         ckpt_path = TaskState().cfg.ckpt_dir
#         weights_path = Path(ckpt_path) / "jobs" / str(weights_path) / "checkpoints" / model_type / f"model_{default_load}.safetensors"

#     weights = safe_load_file(weights_path)

#     if module_name:
#         if not module_name.endswith("."):
#             module_name += "."
#         weights = {k.replace(module_name, ""): v for k, v in weights.items() if k.startswith(module_name)}
#         if map_fn is not None:
#             weights, unmaped_weight = {}, weights
#             for k, v in unmaped_weight.items():
#                 k, v = map_fn(k, v)
#                 if k:
#                     weights[k] = v

#     if not strict:
#         cur_weights = model.state_dict()
#         for k in list(weights.keys()):
#             if k not in cur_weights:
#                 del weights[k]
#                 warn(f"{k} not found in model state_dict, skipping loading.")
#             elif weights[k].shape != cur_weights[k].shape:
#                 warn(f"Shape mismatch for {k}: {weights[k].shape} != {cur_weights[k].shape}, skipping loading.")
#                 del weights[k]

#     model.load_state_dict(weights, strict=strict)
#     mark_init_state(model, True)

#     return len([v.numel() for v in weights.values()])


###############################################################
# Optimizers
###############################################################


def create_parameter_groups(models, weight_decay):
    if isinstance(models, dict):
        models = list(models.values())
    elif not isinstance(models, (list, tuple)):
        models = [models]

    penalized, not_penalized = [], []
    for m in models:
        for module in m.modules():
            if isinstance(module, nn.Embedding):
                not_penalized.append(module.weight)
            elif isinstance(module, nn.Linear):
                penalized.append(module.weight)
                not_penalized.append(module.bias)
            elif isinstance(module, nn.Conv2d):
                penalized.append(module.weight)
                not_penalized.append(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    not_penalized.append(module.bias)
                if module.weight is not None:
                    penalized.append(module.weight)
            elif (
                'Norm' in module.__class__.__name__
                or 'norm' in module.__class__.__name__
            ):
                if hasattr(module, 'weight') and module.weight is not None:
                    penalized.append(module.weight)

    seen_p_ids = set([id(p) for p in penalized + not_penalized])

    for m in models:
        for name, param in m.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in seen_p_ids:
                continue
            if (
                param.ndim < 2
                or 'ln' in name
                or 'bias' in name
                or 'token' in name
                or 'embed' in name
                or 'mask' in name
                or 'norm' in name
            ):
                not_penalized.append(param)
            else:
                penalized.append(param)

    penalized = [p for p in penalized if p is not None and p.requires_grad]
    not_penalized = [p for p in not_penalized if p is not None and p.requires_grad]
    optimizer_params = [
        {'params': penalized, 'weight_decay': weight_decay},
        {'params': not_penalized, 'weight_decay': 0.0},
    ]

    return optimizer_params


def build_optimizer(models, lr, weight_decay):
    optimizer_params = create_parameter_groups(models, weight_decay)
    optimizer = schedulefree.RAdamScheduleFree(optimizer_params, lr=lr)
    return optimizer


###############################################################
# Training utils
###############################################################


# def aggregate_losses(cfg, losses, error_on_nan=True):
#     assert isinstance(losses, dict)
#     # Keep losses with non-zero coeff
#     losses = {loss_name: loss_val for loss_name, loss_val in losses.items() if cfg.losses[loss_name] != 0 and loss_val is not None}

#     # Aggragate losses
#     for loss_name, loss_val in losses.items():
#         if loss_val.isnan().any():
#             if error_on_nan:
#                 raise ValueError(f"Loss {loss_name} is NaN (Losses: {losses})")
#             else:
#                 TaskState().accelerator.warning(f"Loss {loss_name} is NaN (Losses: {losses})")
#     sum_losses = torch.sum(torch.stack([loss_val * cfg.losses[loss_name] for loss_name, loss_val in losses.items()]))

#     return sum_losses, losses


def auto_compile(compile_mode, model):
    assert compile_mode in [True, False, 'jit', 'max']
    if compile_mode:
        if compile_mode == 'jit':
            model = torch.jit.script(model)
        elif compile_mode == 'max':
            model = torch.compile(model, mode='max-autotune', fullgraph=False)
        else:
            model = torch.compile(model, fullgraph=False)
    return model


###############################################################
# Checkpoints
###############################################################


def is_training_state_dir(path):
    if not path:
        return False
    path = Path(path)
    return (
        path.is_dir()
        and (path / 'model.safetensors').exists()
        and (path / 'optimizer.bin').exists()
    )


def find_checkpoint(path):
    check_dirs = [
        '.',
        'checkpoints/last',
        'checkpoints/best',
        'last',
        'best',
        'model-checkpoint',
    ]
    for d in check_dirs:
        if is_training_state_dir(Path(path) / d):
            return str(Path(path) / d)


def force_rename(src, dst):
    for i in range(10):
        try:
            if Path(dst).exists():
                if Path(dst).is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                else:
                    Path(dst).unlink()
            Path(src).rename(dst)
        except Exception as e:
            if i == 9:
                raise e
            time.sleep(1)
        else:
            break


# def save_training_state(state, path, allow_backup=True):
#     if path.exists() and allow_backup:
#         new_path = path.with_name(f"{path.name}.new")
#         back_path = path.with_name(f"{path.name}.back")
#         save_training_state(state, new_path, allow_backup=False)
#         with state.accelerator.sync_ctx() as main:
#             if main:
#                 force_rename(path, back_path)
#                 force_rename(new_path, path)
#                 shutil.rmtree(back_path, ignore_errors=True)

#         return

#     with state.accelerator.sync_ctx() as main:
#         path = Path(path)
#         if main:
#             ensure_path(path)

#     with state.accelerator.sync_ctx():
#         state.accelerator.save_state(str(path))

#     # Create symbolic links for each model
#     with state.accelerator.sync_ctx() as main:
#         if main:
#             for i, model_name in enumerate(state.registered_models):
#                 sd_name = "model.safetensors" if i == 0 else f"model_{i}.safetensors"
#                 sym_path = path / f"model_{model_name}.safetensors"
#                 if sym_path.exists() or sym_path.is_symlink():
#                     sym_path.unlink()
#                 assert (path / sd_name).exists(), f"Model checkpoint {path / sd_name} does not exist"
#                 sym_path.symlink_to(sd_name)


def load_training_state(state, state_path):
    # 1. Find an existing training state
    state_path = find_checkpoint(state_path)
    if not state_path or not is_training_state_dir(state_path):
        state.accelerator.print(f'No training state found to resume from {state_path}')
        return False

    # 2: Load training state
    state.accelerator.print(f'Resuming training from state {state_path}')

    for r in range(10):
        try:
            state.accelerator.load_state(
                str(state_path), strict=True, load_kwargs=dict(weights_only=False)
            )
        except Exception as e:
            if r == 9:
                raise e
        else:
            break

    return True
