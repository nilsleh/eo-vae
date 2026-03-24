"""Transport (flow matching) and Sampler for UNITE.

Copied from UNITE-tokenization-generation/modules/denoiser.py and adapted:
- Import paths updated to use local eo_vae package structure
- scipy dependency removed for sample_logit_normal (replaced with torch)
"""

import math

import numpy as np
import torch as th

from eo_vae.models.unite_core import path as path_module
from eo_vae.models.unite_core.integrators import ode, sde


class Transport:
    """Flow matching transport with linear coupling path (ICPlan)."""

    def __init__(self, train_eps, sample_eps, use_cosine_loss=False, use_lognorm=False,
                 partitial_train=None, partial_ratio=1.0, shift_lg=False,
                 lognorm_mu=0.0, lognorm_sigma=1.0):
        self.path_sampler = path_module.ICPlan()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.use_cosine_loss = use_cosine_loss
        self.use_lognorm = use_lognorm
        self.partitial_train = partitial_train
        self.partial_ratio = partial_ratio
        self.shift_lg = shift_lg
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma

    def prior_logp(self, z):
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(self, train_eps, sample_eps, *, diffusion_form='SBDM', sde=False,
                       reverse=False, eval=False, last_step_size=0.0):
        t0, t1 = 0, 1
        eps = train_eps if not eval else sample_eps
        if sde:
            t0 = eps if (diffusion_form == 'SBDM' and sde) else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def sample_logit_normal(self, mu, sigma, size=1):
        """Sample from logit-normal using torch (no scipy dependency)."""
        samples = th.randn(size) * sigma + mu
        return th.sigmoid(samples).float()

    def sample(self, x1, sp_timesteps=None, shifted_mu=0, timestep_shift=0.0):
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = self.sample_logit_normal(self.lognorm_mu, self.lognorm_sigma, size=x1.shape[0])
        t = t * (t1 - t0) + t0

        if sp_timesteps is not None:
            t = th.rand((x1.shape[0],)) * (sp_timesteps[1] - sp_timesteps[0]) + sp_timesteps[0]

        if timestep_shift > 0:
            t = timestep_shift * t / (1.0 + (timestep_shift - 1.0) * t)
        return t.to(x1), x0, x1

    def training_losses(self, model, x1, t=None, model_kwargs=None, sp_timesteps=None, shifted_mu=0):
        if model_kwargs is None:
            model_kwargs = {}
        if t is None:
            t, x0, x1 = self.sample(x1, sp_timesteps, shifted_mu)
        else:
            x0 = th.randn_like(x1)

        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        return {
            'pred': model_output,
            'model_output': model_output,
            'sampled_t': t,
            'xt': xt,
            'x1': x1,
        }

    def get_drift(self):
        def body_fn(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            assert model_output.shape == x.shape
            return model_output
        return body_fn

    def get_score(self):
        return lambda x, t, model, **kw: self.path_sampler.get_score_from_velocity(
            model(x, t, **kw), x, t
        )


class Sampler:
    """ODE/SDE sampler for the transport model."""

    def __init__(self, transport: Transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def _get_sde_diffusion_and_drift(self, *, diffusion_form='SBDM', diffusion_norm=1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )
        sde_drift = lambda x, t, model, **kw: (
            self.drift(x, t, model, **kw) + diffusion_fn(x, t) * self.score(x, t, model, **kw)
        )
        return sde_drift, diffusion_fn

    def sample_sde(self, *, sampling_method='Euler', diffusion_form='SBDM',
                   diffusion_norm=1.0, last_step='Mean', last_step_size=0.04, num_steps=250):
        sde_drift, sde_diffusion = self._get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            diffusion_form=diffusion_form, sde=True, eval=True,
            reverse=False, last_step_size=last_step_size or 0.0,
        )
        _sde = sde(sde_drift, sde_diffusion, t0=t0, t1=t1,
                   num_steps=num_steps, sampler_type=sampling_method)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            return xs

        return _sample

    def sample_ode(self, *, sampling_method='dopri5', num_steps=50, atol=1e-6,
                   rtol=1e-3, reverse=False, timestep_shift=0.0):
        if reverse:
            drift = lambda x, t, model, **kw: self.drift(
                x, th.ones_like(t) * (1 - t), model, **kw
            )
        else:
            drift = self.drift
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=reverse, last_step_size=0.0,
        )
        _ode = ode(
            drift=drift, t0=t0, t1=t1, sampler_type=sampling_method,
            num_steps=num_steps, atol=atol, rtol=rtol, timestep_shift=timestep_shift,
        )
        return _ode.sample
