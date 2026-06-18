# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
import itertools
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS
from rsl_rl.algorithms.cts import CTS

class MoECTS(CTS):
    model: ActorCriticMoECTS
    def __init__(self,
                model,
                num_envs,
                history_length,
                num_learning_epochs=1,
                num_mini_batches=1,
                clip_param=0.2,
                gamma=0.998,
                lam=0.95,
                value_loss_coef=1.0,
                entropy_coef=0.0,
                load_balance_coef=0.01,
                stable_swav_coef=0.0,
                use_ema_teacher=False,
                ema_teacher_decay=0.995,
                ema_teacher_warmup_updates=100,
                learning_rate=1e-3,
                student_encoder_learning_rate=1e-3,
                max_grad_norm=1.0,
                use_clipped_value_loss=True,
                schedule="fixed",
                desired_kl=0.01,
                teacher_env_ratio=0.75,
                device='cpu',
                ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.history_length = history_length

        # CTS components
        self.model = model
        self.model.to(self.device)
        self.storage = None # initialized later
        teacher_context_params = list(self.model.teacher_context_parameters())
        params1 = [
            {"params": self.model.teacher_encoder.parameters()},
            {"params": self.model.critic.parameters()},
            {"params": self.model.actor.parameters()},
            {"params": self.model.std}
        ]
        if teacher_context_params:
            params1.insert(0, {"params": teacher_context_params})
        self.optimizer1 = optim.Adam(params1, lr=learning_rate)
        self.optimizer2 = optim.Adam(self.model.student_moe_encoder.parameters(), lr=student_encoder_learning_rate)
        self.transition = RolloutStorageCTS.Transition()
        self.tb_metrics = {}

        # CTS parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.load_balance_coef = load_balance_coef
        self.stable_swav_coef = stable_swav_coef
        self.use_ema_teacher = use_ema_teacher
        self.ema_teacher_decay = ema_teacher_decay
        self.ema_teacher_warmup_updates = ema_teacher_warmup_updates
        self.ema_teacher_updates = 0
        self.ema_teacher_effective_decay = 0.0
        self.ema_model = None
        if self.use_ema_teacher:
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.teacher_num_envs = max(int(num_envs * teacher_env_ratio), 1)
        self.student_num_envs = num_envs - self.teacher_num_envs
        student_env_ratio = 1 - teacher_env_ratio
        self.teacher_env_idxs = torch.tensor([i for i in range(num_envs) if i % int(1/student_env_ratio) != 0], device=self.device)
        self.student_env_idxs = torch.tensor([i for i in range(num_envs) if i % int(1/student_env_ratio) == 0], device=self.device)
        assert len(self.teacher_env_idxs) == self.teacher_num_envs, f"{len(self.teacher_env_idxs)=} != {self.teacher_num_envs=}"
        assert len(self.student_env_idxs) == self.student_num_envs, f"{len(self.student_env_idxs)=} != {self.student_num_envs=}"

    def _current_ema_teacher_decay(self):
        if not self.use_ema_teacher:
            return 0.0
        if self.ema_teacher_warmup_updates <= 0:
            return self.ema_teacher_decay
        warmup_fraction = min(
            1.0,
            float(self.ema_teacher_updates + 1) / float(self.ema_teacher_warmup_updates),
        )
        return self.ema_teacher_decay * warmup_fraction

    @torch.no_grad()
    def _update_ema_teacher(self):
        if not self.use_ema_teacher:
            return
        decay = self._current_ema_teacher_decay()
        online_params = dict(self.model.named_parameters())
        for name, ema_param in self.ema_model.named_parameters():
            online_param = online_params.get(name)
            if online_param is None:
                continue
            ema_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)
        self.ema_teacher_updates += 1
        self.ema_teacher_effective_decay = decay

    @torch.no_grad()
    def reset_ema_teacher(self):
        if not self.use_ema_teacher:
            return
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_teacher_updates = 0
        self.ema_teacher_effective_decay = 0.0

    def ema_teacher_state_dict(self):
        if not self.use_ema_teacher:
            return None
        return {
            "model_state_dict": self.ema_model.state_dict(),
            "updates": self.ema_teacher_updates,
            "effective_decay": self.ema_teacher_effective_decay,
        }

    def load_ema_teacher_state_dict(self, state):
        if not self.use_ema_teacher:
            return
        assert state is not None and "model_state_dict" in state, "EMA teacher state is required when EMA teacher is enabled"
        self.ema_model.load_state_dict(state["model_state_dict"])
        self.ema_teacher_updates = int(state["updates"])
        self.ema_teacher_effective_decay = float(state["effective_decay"])
        self.ema_model.eval()

    def encode_teacher_latent_target(self, privileged_obs, history=None, privileged_history=None):
        target_model = self.ema_model if self.use_ema_teacher else self.model
        return target_model.encode_teacher_latent(
            privileged_obs,
            history=history,
            privileged_history=privileged_history,
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_latent_loss = 0
        mean_latent_stable_mse = 0.0
        mean_latent_dynamic_mse = 0.0
        mean_load_balance_loss = 0
        mean_latent_l2 = 0
        mean_latent_cosine = 0
        mean_ema_teacher_online_l2 = 0.0
        mean_ema_teacher_online_cosine = 0.0
        ema_teacher_metric_count = 0
        mean_gate_entropy = 0
        mean_gate_usage = None
        mean_stable_swav_loss = 0.0
        mean_stable_proto_entropy = 0.0
        mean_stable_proto_usage = 0.0
        mean_stable_dynamic_corr = 0.0
        stable_swav_update_count = 0
        film_metrics = {}
        film_metrics_recorded = False
        mean_film_grad_norm = 0.0
        film_grad_norm_count = 0
        priv_history_last_abs_error_sum = 0.0
        priv_history_last_abs_error_max = 0.0
        priv_history_last_abs_error_count = 0
        assert not self.model.is_recurrent
        include_privileged_history = getattr(self.model, "requires_privileged_history", False)
        data = list(
            self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
                include_privileged_history=include_privileged_history,
            )
        )
        def unpack_sample(sample):
            if include_privileged_history:
                (
                    obs_batch, privileged_obs_batch, actions_batch, history_batch, privileged_history_batch,
                    target_values_batch, advantages_batch, returns_batch,
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                    hid_states_batch, masks_batch
                ) = sample
            else:
                (
                    obs_batch, privileged_obs_batch, actions_batch, history_batch,
                    target_values_batch, advantages_batch, returns_batch,
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                    hid_states_batch, masks_batch
                ) = sample
                privileged_history_batch = None
            return (
                obs_batch,
                privileged_obs_batch,
                actions_batch,
                history_batch,
                privileged_history_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            )
        teacher_samples = self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        student_samples = self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        def track_privileged_history_alignment(privileged_obs_batch, privileged_history_batch):
            nonlocal priv_history_last_abs_error_sum
            nonlocal priv_history_last_abs_error_max
            nonlocal priv_history_last_abs_error_count
            if privileged_history_batch is None:
                return
            latest_privileged = privileged_history_batch.view(
                privileged_history_batch.shape[0],
                self.history_length,
                self.model.num_critic_obs,
            )[:, -1]
            diff = (latest_privileged - privileged_obs_batch).abs()
            priv_history_last_abs_error_sum += diff.mean().item()
            priv_history_last_abs_error_max = max(
                priv_history_last_abs_error_max,
                diff.max().item(),
            )
            priv_history_last_abs_error_count += 1

        for sample in data:
            (
                obs_batch, privileged_obs_batch, actions_batch, history_batch, privileged_history_batch,
                target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                hid_states_batch, masks_batch
            ) = unpack_sample(sample)
            track_privileged_history_alignment(privileged_obs_batch, privileged_history_batch)
            def get_results(start, end, is_teacher):
                kwargs = {}
                if privileged_history_batch is not None:
                    kwargs["privileged_history"] = privileged_history_batch[start:end]
                self.model.act(
                    obs_batch[start:end],
                    privileged_obs_batch[start:end],
                    history_batch[start:end],
                    is_teacher,
                    **kwargs,
                )
                actions_log_prob = self.model.get_actions_log_prob(actions_batch[start:end])
                value = self.model.evaluate(
                    privileged_obs_batch[start:end],
                    history_batch[start:end],
                    is_teacher,
                    **kwargs,
                )
                mu = self.model.action_mean
                sigma = self.model.action_std
                entropy = self.model.entropy
                return actions_log_prob, value, mu, sigma, entropy
            teacher_results = get_results(0, teacher_samples, True)
            student_results = get_results(teacher_samples, teacher_samples + student_samples, False)
            results = []
            for x1, x2 in zip(teacher_results, student_results):
                results.append(torch.cat([x1, x2], dim=0))
            actions_log_prob_batch = results[0]
            value_batch = results[1]
            mu_batch = results[2]
            sigma_batch = results[3]
            entropy_batch = results[4]

            if self.model.use_actor_film and not film_metrics_recorded:
                with torch.no_grad():
                    teacher_privileged_history = None
                    if privileged_history_batch is not None:
                        teacher_privileged_history = privileged_history_batch[:teacher_samples]
                    teacher_latent = self.model.encode_teacher_latent(
                        privileged_obs_batch[:teacher_samples],
                        history=history_batch[:teacher_samples],
                        privileged_history=teacher_privileged_history,
                    )
                    student_latent, _ = self.model.student_moe_encoder(
                        history_batch[teacher_samples:teacher_samples + student_samples]
                    )
                    film_latent = torch.cat([teacher_latent, student_latent], dim=0)
                    film_obs = obs_batch[:teacher_samples + student_samples]
                    film_metrics = self.model.actor_film_metrics(film_latent, film_obs)
                    film_metrics_recorded = True

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(
                            sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) +
                                torch.square(old_mu_batch - mu_batch)
                            ) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer1.param_groups:
                        param_group['lr'] = self.learning_rate


            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_losses = torch.max(surrogate, surrogate_clipped)
            teacher_surrogate_loss = surrogate_losses[:teacher_samples].mean()
            student_surrogate_loss = surrogate_losses[teacher_samples:].mean()
            surrogate_loss = teacher_surrogate_loss + student_surrogate_loss
            # surrogate_loss = teacher_surrogate_loss

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            # teacher_value_loss = value_losses[:teacher_samples].mean()
            # student_value_loss = value_losses[teacher_samples:].mean()
            # value_loss = teacher_value_loss  # + student_value_loss

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            stable_swav_loss, stable_swav_metrics = self.model.stable_swav_loss(privileged_history_batch)
            if stable_swav_loss is not None and self.stable_swav_coef > 0.0:
                loss = loss + self.stable_swav_coef * stable_swav_loss
            if stable_swav_loss is not None:
                mean_stable_swav_loss += stable_swav_metrics.get("stable_swav_loss", 0.0)
                mean_stable_proto_entropy += stable_swav_metrics.get("stable_proto_entropy", 0.0)
                mean_stable_proto_usage += stable_swav_metrics.get("stable_proto_usage", 0.0)
                mean_stable_dynamic_corr += stable_swav_metrics.get("stable_dynamic_corr", 0.0)
                stable_swav_update_count += 1

            # Gradient step
            self.optimizer1.zero_grad()
            loss.backward()
            if self.model.use_actor_film:
                mean_film_grad_norm += self.model.actor_film_grad_norm()
                film_grad_norm_count += 1
            params_to_clip = itertools.chain.from_iterable(g['params'] for g in self.optimizer1.param_groups)
            nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
            self.optimizer1.step()
            self.model.normalize_swav_prototypes()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()

        self._update_ema_teacher()
        
        for sample in data:
            (
                obs_batch, privileged_obs_batch, actions_batch, history_batch, privileged_history_batch,
                target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                hid_states_batch, masks_batch
            ) = unpack_sample(sample)
            # Student encoder update
            student_latent, gating_weights = self.model.student_moe_encoder(history_batch[teacher_samples:])
            with torch.no_grad():
                teacher_privileged_history = None
                if privileged_history_batch is not None:
                    teacher_privileged_history = privileged_history_batch[teacher_samples:]
                teacher_latent = self.encode_teacher_latent_target(
                    privileged_obs_batch[teacher_samples:],
                    history=history_batch[teacher_samples:],
                    privileged_history=teacher_privileged_history,
                )
                if self.use_ema_teacher:
                    online_teacher_latent = self.model.encode_teacher_latent(
                        privileged_obs_batch[teacher_samples:],
                        history=history_batch[teacher_samples:],
                        privileged_history=teacher_privileged_history,
                    )
                    mean_ema_teacher_online_l2 += torch.norm(
                        online_teacher_latent - teacher_latent,
                        dim=1,
                    ).mean().item()
                    mean_ema_teacher_online_cosine += F.cosine_similarity(
                        online_teacher_latent,
                        teacher_latent,
                        dim=1,
                    ).mean().item()
                    ema_teacher_metric_count += 1
            latent_loss = (teacher_latent - student_latent).pow(2).mean()
            latent_error = teacher_latent - student_latent
            stable_dim = getattr(self.model, "stable_latent_dim", 0)
            if 0 < stable_dim < latent_error.shape[1]:
                latent_stable_mse = latent_error[:, :stable_dim].pow(2).mean()
                latent_dynamic_mse = latent_error[:, stable_dim:].pow(2).mean()
            else:
                latent_stable_mse = latent_loss.new_tensor(0.0)
                latent_dynamic_mse = latent_loss
            latent_l2 = torch.norm(teacher_latent - student_latent, dim=1).mean()
            latent_cosine = F.cosine_similarity(teacher_latent, student_latent, dim=1).mean()

            # Load balance loss
            mean_usage = torch.mean(gating_weights, dim=0)
            target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[1])
            load_balance_loss = torch.mean((mean_usage - target_usage).pow(2))
            gate_entropy = -(gating_weights * torch.log(gating_weights + 1e-8)).sum(dim=1).mean()
            # load_balance_loss = torch.sum(mean_usage.pow(2)) * gating_weights.shape[1]  # Switch Transformer style

            student_loss = latent_loss + self.load_balance_coef * load_balance_loss

            self.optimizer2.zero_grad()
            student_loss.backward()
            nn.utils.clip_grad_norm_(self.model.student_moe_encoder.parameters(), self.max_grad_norm)
            self.optimizer2.step()

            mean_latent_loss += latent_loss.item()
            mean_latent_stable_mse += latent_stable_mse.item()
            mean_latent_dynamic_mse += latent_dynamic_mse.item()
            mean_load_balance_loss += load_balance_loss.item()
            mean_latent_l2 += latent_l2.item()
            mean_latent_cosine += latent_cosine.item()
            mean_gate_entropy += gate_entropy.item()
            if mean_gate_usage is None:
                mean_gate_usage = mean_usage.detach()
            else:
                mean_gate_usage += mean_usage.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_latent_loss /= num_updates
        mean_latent_stable_mse /= num_updates
        mean_latent_dynamic_mse /= num_updates
        mean_load_balance_loss /= num_updates
        mean_latent_l2 /= num_updates
        mean_latent_cosine /= num_updates
        mean_gate_entropy /= num_updates
        mean_gate_usage = mean_gate_usage / num_updates
        if priv_history_last_abs_error_count > 0:
            priv_history_last_abs_error_mean = (
                priv_history_last_abs_error_sum / priv_history_last_abs_error_count
            )
        else:
            priv_history_last_abs_error_mean = 0.0
        if stable_swav_update_count > 0:
            mean_stable_swav_loss /= stable_swav_update_count
            mean_stable_proto_entropy /= stable_swav_update_count
            mean_stable_proto_usage /= stable_swav_update_count
            mean_stable_dynamic_corr /= stable_swav_update_count
        if film_grad_norm_count > 0:
            mean_film_grad_norm /= film_grad_norm_count
        if ema_teacher_metric_count > 0:
            mean_ema_teacher_online_l2 /= ema_teacher_metric_count
            mean_ema_teacher_online_cosine /= ema_teacher_metric_count
        self.tb_metrics = {
            "latent_l2": mean_latent_l2,
            "latent_cosine_similarity": mean_latent_cosine,
            "latent_stable_mse": mean_latent_stable_mse,
            "latent_dynamic_mse": mean_latent_dynamic_mse,
            "gate_entropy": mean_gate_entropy,
            "gate_usage": mean_gate_usage.detach().cpu(),
            "gate_usage_max": mean_gate_usage.max().item(),
            "gate_usage_min": mean_gate_usage.min().item(),
            "gate_usage_std": mean_gate_usage.std(unbiased=False).item(),
            "priv_history_last_abs_error_mean": priv_history_last_abs_error_mean,
            "priv_history_last_abs_error_max": priv_history_last_abs_error_max,
            "stable_swav_loss": mean_stable_swav_loss,
            "stable_proto_entropy": mean_stable_proto_entropy,
            "stable_proto_usage": mean_stable_proto_usage,
            "stable_dynamic_corr": mean_stable_dynamic_corr,
            "ema_teacher_enabled": float(self.use_ema_teacher),
            "ema_teacher_decay": self.ema_teacher_decay,
            "ema_teacher_effective_decay": self.ema_teacher_effective_decay,
            "ema_teacher_updates": float(self.ema_teacher_updates),
            "ema_teacher_online_l2": mean_ema_teacher_online_l2,
            "ema_teacher_online_cosine": mean_ema_teacher_online_cosine,
        }
        if self.model.use_actor_film:
            self.tb_metrics.update({
                f"film_{key}": value
                for key, value in film_metrics.items()
            })
            self.tb_metrics["film_grad_norm"] = mean_film_grad_norm
            self.tb_metrics["film_param_norm"] = self.model.actor_film_param_norm()
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss, mean_load_balance_loss
