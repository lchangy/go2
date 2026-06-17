# -*- coding: utf-8 -*-
'''
@File    : actor_critic_moe_cts.py
@Time    : 2025/12/30 21:06:46
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Mixture of Experts Concurrent Teacher Student Network
@Refer   : CTS https://arxiv.org/abs/2405.10830, Switch Transformers https://arxiv.org/abs/2101.03961
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.modules.utils import L2Norm, SimNorm, StudentMoEEncoder, MLP

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, token_dim, token_hidden_dim, channel_hidden_dim, activation='elu'):
        super().__init__()
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_mixing = MLP([num_tokens, token_hidden_dim, num_tokens], activation=activation)
        self.channel_norm = nn.LayerNorm(token_dim)
        self.channel_mixing = MLP([token_dim, channel_hidden_dim, token_dim], activation=activation)

    def forward(self, x):
        y = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mixing(y).transpose(1, 2)
        return x + self.channel_mixing(self.channel_norm(x))

class StableFiLMActor(nn.Module):
    def __init__(self, input_dim, conditioning_dim, hidden_dims, action_dim, activation='elu'):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = F.elu if activation == 'elu' else None
        if self.activation is None:
            raise AssertionError("StableFiLMActor currently supports elu activation only")

        dims = [input_dim, *hidden_dims]
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.film = nn.Linear(conditioning_dim, 2 * sum(hidden_dims))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, conditioning):
        film_params = self.film(conditioning)
        gammas, betas = torch.split(film_params, sum(self.hidden_dims), dim=-1)
        offset = 0
        h = x
        for layer, hidden_dim in zip(self.hidden_layers, self.hidden_dims):
            h = self.activation(layer(h))
            gamma = gammas[:, offset:offset + hidden_dim]
            beta = betas[:, offset:offset + hidden_dim]
            h = h * (1.0 + gamma) + beta
            offset += hidden_dim
        return self.output_layer(h)

class ActorCriticMoECTS(nn.Module):
    is_recurrent = False
    def __init__(self,  num_obs,
                        num_critic_obs,
                        num_actions,
                        num_envs,
                        history_length,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        teacher_encoder_hidden_dims=[512, 256],
                        student_encoder_hidden_dims=[512, 256, 256],
                        expert_num=8,
                        activation='elu',
                        init_noise_std=1.0,
                        latent_dim=32,
                        norm_type='l2norm',
                        privileged_height_start=None,
                        privileged_height_dim=0,
                        privileged_height_latent_dim=0,
                        height_encoder_hidden_dims=None,
                        teacher_context_mode=None,
                        use_teacher_mixer=False,
                        teacher_mixer_token_dim=64,
                        teacher_mixer_summary_tokens=1,
                        teacher_mixer_summary_aggregation='first',
                        teacher_mixer_num_blocks=2,
                        teacher_mixer_token_hidden_dim=64,
                        teacher_mixer_channel_hidden_dim=128,
                        use_actor_film=False,
                        detach_critic_context=False,
                        critic_use_privileged_obs=False,
                        use_stable_swav=False,
                        stable_latent_dim=8,
                        swav_num_prototypes=64,
                        swav_temperature=0.1,
                        swav_epsilon=0.05,
                        swav_sinkhorn_iters=3,
                        swav_privileged_noise_std=0.005,
                        swav_privileged_dropout_prob=0.02,
                        swav_height_start=None,
                        swav_height_dim=0,
                        swav_height_noise_std=0.02,
                        swav_height_dropout_prob=0.1,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        assert norm_type in ['l2norm', 'simnorm'], f"Normalization type {norm_type} not supported!"
        super().__init__()
        self.num_actions = num_actions
        self.history_length = history_length
        self.privileged_height_start = privileged_height_start
        self.privileged_height_dim = privileged_height_dim
        self.privileged_height_latent_dim = privileged_height_latent_dim
        self.use_privileged_height_encoder = privileged_height_dim > 0 and privileged_height_latent_dim > 0
        self.num_obs = num_obs
        self.num_critic_obs = num_critic_obs
        if teacher_context_mode is None:
            teacher_context_mode = "mixer" if use_teacher_mixer else "current_privileged"
        valid_teacher_context_modes = (
            "current_privileged",
            "latest_privileged_history",
            "flat_privileged_history",
            "mixer",
        )
        if teacher_context_mode not in valid_teacher_context_modes:
            raise ValueError(
                f"teacher_context_mode must be one of {valid_teacher_context_modes}, got {teacher_context_mode}"
            )
        if teacher_mixer_summary_aggregation not in ("first", "mean", "concat"):
            raise ValueError(
                "teacher_mixer_summary_aggregation must be one of ('first', 'mean', 'concat'), "
                f"got {teacher_mixer_summary_aggregation}"
            )
        self.teacher_context_mode = teacher_context_mode
        self.use_teacher_mixer = teacher_context_mode == "mixer"
        self.requires_privileged_history = teacher_context_mode in (
            "latest_privileged_history",
            "flat_privileged_history",
            "mixer",
        ) or use_stable_swav
        self.use_actor_film = use_actor_film
        self.detach_critic_context = detach_critic_context
        self.critic_use_privileged_obs = critic_use_privileged_obs
        self.use_stable_swav = use_stable_swav
        self.stable_latent_dim = stable_latent_dim
        self.swav_temperature = swav_temperature
        self.swav_epsilon = swav_epsilon
        self.swav_sinkhorn_iters = swav_sinkhorn_iters
        self.swav_privileged_noise_std = swav_privileged_noise_std
        self.swav_privileged_dropout_prob = swav_privileged_dropout_prob
        self.swav_height_start = swav_height_start
        self.swav_height_dim = swav_height_dim
        self.swav_height_noise_std = swav_height_noise_std
        self.swav_height_dropout_prob = swav_height_dropout_prob
        if self.swav_height_start is not None and self.swav_height_dim > 0:
            self.swav_height_end = self.swav_height_start + self.swav_height_dim
            assert self.swav_height_end <= num_critic_obs, (
                f"SwAV height slice [{self.swav_height_start}:{self.swav_height_end}] exceeds "
                f"num_critic_obs={num_critic_obs}"
            )
        else:
            self.swav_height_end = None
        self.teacher_mixer_summary_tokens = teacher_mixer_summary_tokens
        self.teacher_mixer_summary_aggregation = teacher_mixer_summary_aggregation
        if self.use_stable_swav or self.use_actor_film:
            assert 0 < stable_latent_dim < latent_dim, (
                f"stable_latent_dim must be in (0, latent_dim), got {stable_latent_dim=} and {latent_dim=}"
            )

        compact_num_critic_obs = num_critic_obs
        if self.use_privileged_height_encoder:
            assert privileged_height_start is not None, "privileged_height_start must be set when height encoder is enabled"
            privileged_height_end = privileged_height_start + privileged_height_dim
            assert privileged_height_end <= num_critic_obs, (
                f"height slice [{privileged_height_start}:{privileged_height_end}] exceeds "
                f"num_critic_obs={num_critic_obs}"
            )
            if height_encoder_hidden_dims is None:
                height_encoder_hidden_dims = [64, 32]
            self.privileged_height_end = privileged_height_end
            self.height_encoder = MLP(
                [privileged_height_dim, *height_encoder_hidden_dims, privileged_height_latent_dim],
                activation=activation,
            )
            compact_num_critic_obs = num_critic_obs - privileged_height_dim + privileged_height_latent_dim
        else:
            self.privileged_height_end = None
            self.height_encoder = nn.Identity()
        self.compact_num_critic_obs = compact_num_critic_obs
        if self.use_teacher_mixer:
            if teacher_mixer_summary_aggregation == "concat":
                teacher_context_dim = teacher_mixer_token_dim * teacher_mixer_summary_tokens
            else:
                teacher_context_dim = teacher_mixer_token_dim
        elif teacher_context_mode == "flat_privileged_history":
            teacher_context_dim = history_length * num_critic_obs
        else:
            teacher_context_dim = compact_num_critic_obs

        if self.use_teacher_mixer:
            self.teacher_mixer_num_tokens = history_length * 2 + teacher_mixer_summary_tokens
            self.obs_token_proj = nn.Linear(num_obs, teacher_mixer_token_dim)
            self.privileged_token_proj = nn.Linear(compact_num_critic_obs, teacher_mixer_token_dim)
            self.teacher_summary_token = nn.Parameter(
                torch.zeros(1, teacher_mixer_summary_tokens, teacher_mixer_token_dim)
            )
            self.teacher_token_type_embedding = nn.Embedding(3, teacher_mixer_token_dim)
            self.teacher_time_embedding = nn.Embedding(history_length, teacher_mixer_token_dim)
            self.teacher_mixer = nn.Sequential(
                *[
                    MixerBlock(
                        self.teacher_mixer_num_tokens,
                        teacher_mixer_token_dim,
                        teacher_mixer_token_hidden_dim,
                        teacher_mixer_channel_hidden_dim,
                        activation=activation,
                    )
                    for _ in range(teacher_mixer_num_blocks)
                ]
            )
            type_ids = [0] * teacher_mixer_summary_tokens
            for _ in range(history_length):
                type_ids.extend([1, 2])
            self.register_buffer(
                "teacher_mixer_type_ids",
                torch.tensor(type_ids, dtype=torch.long).unsqueeze(0),
                persistent=False,
            )
            self.register_buffer(
                "teacher_mixer_time_ids",
                torch.arange(history_length, dtype=torch.long).repeat_interleave(2).unsqueeze(0),
                persistent=False,
            )
        else:
            self.teacher_mixer_num_tokens = None
            self.obs_token_proj = nn.Identity()
            self.privileged_token_proj = nn.Identity()
            self.teacher_mixer = nn.Identity()
            self.teacher_token_type_embedding = nn.Identity()
            self.teacher_time_embedding = nn.Identity()

        mlp_input_dim_t = teacher_context_dim
        mlp_input_dim_s = num_obs * history_length
        actor_latent_input_dim = latent_dim - stable_latent_dim if self.use_actor_film else latent_dim
        mlp_input_dim_a = actor_latent_input_dim + num_obs
        critic_context_dim = num_critic_obs if self.critic_use_privileged_obs else teacher_context_dim
        mlp_input_dim_c = latent_dim + critic_context_dim

        # History
        self.register_buffer("history", torch.zeros((num_envs, history_length, num_obs)), persistent=False)

        # Teacher encoder
        self.teacher_encoder = nn.Sequential(
            MLP([mlp_input_dim_t, *teacher_encoder_hidden_dims, latent_dim], activation=activation),
            L2Norm() if norm_type == 'l2norm' else SimNorm()
        )

        # Student MoE encoder
        self.student_moe_encoder = StudentMoEEncoder(
            expert_num=expert_num,
            input_dim=mlp_input_dim_s,
            hidden_dims=student_encoder_hidden_dims,
            output_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
        )

        # Policy
        if self.use_actor_film:
            self.actor = StableFiLMActor(
                mlp_input_dim_a,
                stable_latent_dim,
                actor_hidden_dims,
                num_actions,
                activation=activation,
            )
        else:
            self.actor = MLP([mlp_input_dim_a, *actor_hidden_dims, num_actions], activation=activation)

        # Value function
        self.critic = MLP([mlp_input_dim_c, *critic_hidden_dims, 1], activation=activation)
        if self.use_stable_swav:
            self.swav_prototypes = nn.Linear(stable_latent_dim, swav_num_prototypes, bias=False)
        else:
            self.swav_prototypes = nn.Identity()

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        if self.use_privileged_height_encoder:
            print(f"Height Encoder: {self.height_encoder}")
            print(f"Compact privileged obs dim: {num_critic_obs} -> {compact_num_critic_obs}")
        if self.use_teacher_mixer:
            print(f"Teacher Mixer: {self.teacher_mixer}")
            print(f"Teacher mixer tokens: {self.teacher_mixer_num_tokens}, token dim: {teacher_mixer_token_dim}")
            print(
                f"Teacher mixer summary tokens: {self.teacher_mixer_summary_tokens}, "
                f"aggregation: {self.teacher_mixer_summary_aggregation}"
            )
        if self.use_stable_swav:
            print(
                f"Stable SwAV: stable_dim={stable_latent_dim}, prototypes={swav_num_prototypes}, "
                f"temperature={swav_temperature}, epsilon={swav_epsilon}, sinkhorn_iters={swav_sinkhorn_iters}"
            )
            print(
                "Stable SwAV augmentation: "
                f"priv_noise={swav_privileged_noise_std}, priv_dropout={swav_privileged_dropout_prob}, "
                f"height_slice={self.swav_height_start}:{self.swav_height_end}, "
                f"height_noise={swav_height_noise_std}, height_dropout={swav_height_dropout_prob}"
            )
        print(f"Teacher context mode: {self.teacher_context_mode}")
        print(f"Detach critic context: {self.detach_critic_context}")
        print(f"Critic uses privileged obs: {self.critic_use_privileged_obs}")
        if self.use_actor_film:
            print(
                f"Stable FiLM actor: stable_dim={stable_latent_dim}, "
                f"dynamic_dim={latent_dim - stable_latent_dim}, actor_input_dim={mlp_input_dim_a}"
            )
        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student MoE Encoder: {self.student_moe_encoder}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        self.history[dones > 0] = 0.0

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def actor_forward(self, latent, obs):
        if self.use_actor_film:
            stable_latent = latent[:, :self.stable_latent_dim]
            dynamic_latent = latent[:, self.stable_latent_dim:]
            dynamic_latent_and_obs = torch.cat([dynamic_latent, obs], dim=1)
            return self.actor(dynamic_latent_and_obs, stable_latent)
        latent_and_obs = torch.cat([latent, obs], dim=1)
        return self.actor(latent_and_obs)

    def update_distribution(self, latent, obs):
        mean = self.actor_forward(latent, obs)
        self.distribution = Normal(mean, mean*0. + self.std)

    def teacher_context_parameters(self):
        yield from self.height_encoder.parameters()
        if self.use_stable_swav:
            yield from self.swav_prototypes.parameters()
        if not self.use_teacher_mixer:
            return
        yield from self.obs_token_proj.parameters()
        yield from self.privileged_token_proj.parameters()
        yield self.teacher_summary_token
        yield from self.teacher_token_type_embedding.parameters()
        yield from self.teacher_time_embedding.parameters()
        yield from self.teacher_mixer.parameters()

    def encode_privileged_obs(self, privileged_obs):
        if not self.use_privileged_height_encoder:
            return privileged_obs

        height_latent = self.height_encoder(
            privileged_obs[:, self.privileged_height_start:self.privileged_height_end]
        )
        return torch.cat(
            [
                privileged_obs[:, :self.privileged_height_start],
                height_latent,
                privileged_obs[:, self.privileged_height_end:],
            ],
            dim=1,
        )

    def encode_teacher_context(self, privileged_obs, history=None, privileged_history=None):
        if self.teacher_context_mode == "current_privileged":
            return self.encode_privileged_obs(privileged_obs)
        if self.teacher_context_mode == "latest_privileged_history":
            assert privileged_history is not None, (
                "privileged_history is required when teacher_context_mode='latest_privileged_history'"
            )
            privileged_history = privileged_history.reshape(
                privileged_history.shape[0],
                self.history_length,
                self.num_critic_obs,
            )
            return self.encode_privileged_obs(privileged_history[:, -1])
        if self.teacher_context_mode == "flat_privileged_history":
            assert privileged_history is not None, (
                "privileged_history is required when teacher_context_mode='flat_privileged_history'"
            )
            return privileged_history.reshape(privileged_history.shape[0], -1)

        assert history is not None, "history is required when teacher mixer is enabled"
        assert privileged_history is not None, "privileged_history is required when teacher mixer is enabled"
        batch_size = history.shape[0]
        obs_history = history.reshape(batch_size, self.history_length, self.num_obs)
        privileged_history = privileged_history.reshape(batch_size, self.history_length, self.num_critic_obs)
        compact_privileged_history = self.encode_privileged_obs(
            privileged_history.reshape(-1, self.num_critic_obs)
        ).view(batch_size, self.history_length, self.compact_num_critic_obs)

        obs_tokens = self.obs_token_proj(obs_history)
        privileged_tokens = self.privileged_token_proj(compact_privileged_history)
        paired_tokens = torch.stack([obs_tokens, privileged_tokens], dim=2).reshape(
            batch_size,
            self.history_length * 2,
            -1,
        )
        summary_token = self.teacher_summary_token.expand(batch_size, -1, -1)
        tokens = torch.cat([summary_token, paired_tokens], dim=1)
        tokens = tokens + self.teacher_token_type_embedding(self.teacher_mixer_type_ids)
        tokens[:, self.teacher_mixer_summary_tokens:] = (
            tokens[:, self.teacher_mixer_summary_tokens:]
            + self.teacher_time_embedding(self.teacher_mixer_time_ids)
        )
        summary_outputs = self.teacher_mixer(tokens)[:, :self.teacher_mixer_summary_tokens]
        if self.teacher_mixer_summary_aggregation == "concat":
            return summary_outputs.reshape(batch_size, -1)
        if self.teacher_mixer_summary_aggregation == "mean":
            return summary_outputs.mean(dim=1)
        return summary_outputs[:, 0]

    def encode_teacher_latent(self, privileged_obs, history=None, privileged_history=None):
        return self.teacher_encoder(
            self.encode_teacher_context(
                privileged_obs,
                history=history,
                privileged_history=privileged_history,
            )
        )

    def _augment_privileged_for_swav(self, privileged_obs):
        augmented = privileged_obs.clone()
        if self.swav_privileged_dropout_prob > 0.0:
            dropout_mask = torch.rand_like(augmented) < self.swav_privileged_dropout_prob
            feature_mean = augmented.mean(dim=0, keepdim=True)
            augmented = torch.where(dropout_mask, feature_mean, augmented)
        if self.swav_privileged_noise_std > 0.0:
            augmented = augmented + torch.randn_like(augmented) * self.swav_privileged_noise_std
        if self.swav_height_end is not None:
            height = augmented[:, self.swav_height_start:self.swav_height_end]
            if self.swav_height_dropout_prob > 0.0:
                dropout_mask = torch.rand_like(height) < self.swav_height_dropout_prob
                height_mean = height.mean(dim=1, keepdim=True)
                height = torch.where(dropout_mask, height_mean, height)
            if self.swav_height_noise_std > 0.0:
                height = height + torch.randn_like(height) * self.swav_height_noise_std
            augmented[:, self.swav_height_start:self.swav_height_end] = height
        return augmented

    @torch.no_grad()
    def normalize_swav_prototypes(self):
        if not self.use_stable_swav:
            return
        weight = self.swav_prototypes.weight.data
        self.swav_prototypes.weight.copy_(F.normalize(weight, dim=1))

    @torch.no_grad()
    def _sinkhorn_assignments(self, logits):
        q = torch.exp(logits / self.swav_epsilon).t()
        q = q / torch.clamp(q.sum(), min=1e-12)
        num_prototypes, batch_size = q.shape
        for _ in range(self.swav_sinkhorn_iters):
            q = q / torch.clamp(q.sum(dim=1, keepdim=True), min=1e-12)
            q = q / num_prototypes
            q = q / torch.clamp(q.sum(dim=0, keepdim=True), min=1e-12)
            q = q / batch_size
        q = q * batch_size
        return q.t()

    def _stable_dynamic_corr(self, latent):
        dynamic_dim = latent.shape[1] - self.stable_latent_dim
        if dynamic_dim <= 0 or latent.shape[0] < 2:
            return latent.new_tensor(0.0)
        stable = latent[:, :self.stable_latent_dim]
        dynamic = latent[:, self.stable_latent_dim:]
        stable = (stable - stable.mean(dim=0, keepdim=True)) / torch.clamp(stable.std(dim=0, keepdim=True), min=1e-6)
        dynamic = (dynamic - dynamic.mean(dim=0, keepdim=True)) / torch.clamp(dynamic.std(dim=0, keepdim=True), min=1e-6)
        corr = stable.t().matmul(dynamic) / (latent.shape[0] - 1)
        return corr.pow(2).mean()

    def stable_swav_loss(self, privileged_history):
        if not self.use_stable_swav:
            return None, {}
        assert privileged_history is not None, "privileged_history is required when stable SwAV is enabled"
        privileged_history = privileged_history.reshape(
            privileged_history.shape[0],
            self.history_length,
            self.num_critic_obs,
        )
        if self.history_length < 2:
            zero = privileged_history.new_tensor(0.0)
            return zero, {
                "stable_swav_loss": 0.0,
                "stable_proto_entropy": 0.0,
                "stable_proto_usage": 0.0,
                "stable_dynamic_corr": 0.0,
            }

        view_a = self._augment_privileged_for_swav(privileged_history[:, -1])
        view_b = self._augment_privileged_for_swav(privileged_history[:, -2])
        latent_a = self.encode_teacher_latent(view_a)
        latent_b = self.encode_teacher_latent(view_b)
        stable_a = F.normalize(latent_a[:, :self.stable_latent_dim], dim=1)
        stable_b = F.normalize(latent_b[:, :self.stable_latent_dim], dim=1)

        self.normalize_swav_prototypes()
        logits_a = self.swav_prototypes(stable_a)
        logits_b = self.swav_prototypes(stable_b)
        with torch.no_grad():
            assignments_a = self._sinkhorn_assignments(logits_a.detach())
            assignments_b = self._sinkhorn_assignments(logits_b.detach())

        log_probs_a = F.log_softmax(logits_a / self.swav_temperature, dim=1)
        log_probs_b = F.log_softmax(logits_b / self.swav_temperature, dim=1)
        loss = -0.5 * (
            (assignments_a * log_probs_b).sum(dim=1).mean()
            + (assignments_b * log_probs_a).sum(dim=1).mean()
        )

        with torch.no_grad():
            prototype_usage = 0.5 * (assignments_a.mean(dim=0) + assignments_b.mean(dim=0))
            entropy_raw = -(
                prototype_usage * torch.log(torch.clamp(prototype_usage, min=1e-8))
            ).sum()
            prototype_entropy = entropy_raw / np.log(prototype_usage.numel())
            effective_usage = torch.exp(entropy_raw) / prototype_usage.numel()
            stable_dynamic_corr = 0.5 * (
                self._stable_dynamic_corr(latent_a) + self._stable_dynamic_corr(latent_b)
            )
        return loss, {
            "stable_swav_loss": loss.detach().item(),
            "stable_proto_entropy": prototype_entropy.item(),
            "stable_proto_usage": effective_usage.item(),
            "stable_dynamic_corr": stable_dynamic_corr.item(),
        }
    
    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.encode_teacher_latent(
                privileged_obs,
                history=history,
                privileged_history=kwargs.get("privileged_history"),
            )
        else:
            with torch.no_grad():
                latent, _ = self.student_moe_encoder(history)
        self.update_distribution(latent, obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent, _ = self.student_moe_encoder(self.history.flatten(1))
        actions_mean = self.actor_forward(latent, obs)
        return actions_mean

    def evaluate(self, privileged_obs, history, is_teacher, **kwargs):
        teacher_context = self.encode_teacher_context(
            privileged_obs,
            history=history,
            privileged_history=kwargs.get("privileged_history"),
        )
        if is_teacher:
            latent = self.teacher_encoder(teacher_context)
        else:
            latent, _ = self.student_moe_encoder(history)
        if self.critic_use_privileged_obs:
            critic_context = privileged_obs
        else:
            critic_context = teacher_context.detach() if self.detach_critic_context else teacher_context
        x = torch.cat([latent.detach(), critic_context], dim=1)
        value = self.critic(x)
        return value
