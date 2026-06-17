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

class FiLMActor(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, action_dim, activation='elu'):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = F.elu if activation == 'elu' else None
        if self.activation is None:
            raise AssertionError("FiLMActor currently supports elu activation only")

        dims = [input_dim, *hidden_dims]
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))]
        )
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.film = nn.Linear(latent_dim, 2 * sum(hidden_dims))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, latent):
        film_params = self.film(latent)
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
                        use_teacher_mixer=False,
                        teacher_mixer_token_dim=64,
                        teacher_mixer_num_blocks=2,
                        teacher_mixer_token_hidden_dim=64,
                        teacher_mixer_channel_hidden_dim=128,
                        use_actor_film=False,
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
        self.use_teacher_mixer = use_teacher_mixer
        self.use_actor_film = use_actor_film

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
        teacher_context_dim = teacher_mixer_token_dim if self.use_teacher_mixer else compact_num_critic_obs

        if self.use_teacher_mixer:
            self.teacher_mixer_num_tokens = history_length * 2 + 1
            self.obs_token_proj = nn.Linear(num_obs, teacher_mixer_token_dim)
            self.privileged_token_proj = nn.Linear(compact_num_critic_obs, teacher_mixer_token_dim)
            self.teacher_summary_token = nn.Parameter(torch.zeros(1, 1, teacher_mixer_token_dim))
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
            type_ids = [0]
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
        mlp_input_dim_a = latent_dim + num_obs
        mlp_input_dim_c = latent_dim + teacher_context_dim

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
            self.actor = FiLMActor(
                mlp_input_dim_a,
                latent_dim,
                actor_hidden_dims,
                num_actions,
                activation=activation,
            )
        else:
            self.actor = MLP([mlp_input_dim_a, *actor_hidden_dims, num_actions], activation=activation)

        # Value function
        self.critic = MLP([mlp_input_dim_c, *critic_hidden_dims, 1], activation=activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        if self.use_privileged_height_encoder:
            print(f"Height Encoder: {self.height_encoder}")
            print(f"Compact privileged obs dim: {num_critic_obs} -> {compact_num_critic_obs}")
        if self.use_teacher_mixer:
            print(f"Teacher Mixer: {self.teacher_mixer}")
            print(f"Teacher mixer tokens: {self.teacher_mixer_num_tokens}, token dim: {teacher_mixer_token_dim}")
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
        latent_and_obs = torch.cat([latent, obs], dim=1)
        if self.use_actor_film:
            return self.actor(latent_and_obs, latent)
        return self.actor(latent_and_obs)

    def update_distribution(self, latent, obs):
        mean = self.actor_forward(latent, obs)
        self.distribution = Normal(mean, mean*0. + self.std)

    def teacher_context_parameters(self):
        yield from self.height_encoder.parameters()
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
        if not self.use_teacher_mixer:
            return self.encode_privileged_obs(privileged_obs)

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
        tokens[:, 1:] = tokens[:, 1:] + self.teacher_time_embedding(self.teacher_mixer_time_ids)
        return self.teacher_mixer(tokens)[:, 0]

    def encode_teacher_latent(self, privileged_obs, history=None, privileged_history=None):
        return self.teacher_encoder(
            self.encode_teacher_context(
                privileged_obs,
                history=history,
                privileged_history=privileged_history,
            )
        )
    
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
        x = torch.cat([latent.detach(), teacher_context], dim=1)
        value = self.critic(x)
        return value
