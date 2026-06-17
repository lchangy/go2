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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import CTS, MoENGCTS, MCPCTS, ACMoECTS, DualMoECTS, MoECTS
from rsl_rl.modules import ActorCriticCTS, ActorCriticMoENGCTS, ActorCriticMCPCTS, ActorCriticACMoECTS, ActorCriticDualMoECTS, ActorCriticMoECTS
from rsl_rl.env import VecEnv

import yaml
import numpy as np
from pathlib import Path
from legged_gym.utils.helpers import class_to_dict
from typing import Union
from legged_gym.utils.exporter import export_policy_as_jit

def numpy_representer(dumper, data):
    return dumper.represent_float(float(data))

def numpy_int_representer(dumper, data):
    return dumper.represent_int(int(data))

# Add the numpy representer to yaml
yaml.add_representer(np.float32, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.float64, numpy_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int32, numpy_int_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(np.int64, numpy_int_representer, Dumper=yaml.SafeDumper)


class OnPolicyRunnerCTS:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        history_length = train_cfg["history_length"]
        actor_critic_class = eval(self.cfg["policy_class_name"])
        model: Union[ActorCriticCTS, ActorCriticMoENGCTS, ActorCriticMCPCTS, ActorCriticACMoECTS, ActorCriticDualMoECTS, ActorCriticMoECTS] = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            self.env.num_envs,
            history_length,
            **self.policy_cfg).to(self.device)
        self.use_privileged_history = getattr(model, "requires_privileged_history", False)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: Union[CTS, MoENGCTS, MCPCTS, ACMoECTS, DualMoECTS, MoECTS] = alg_class(model, self.env.num_envs, history_length, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        privileged_history_shape = [history_length * self.env.num_privileged_obs] if self.use_privileged_history else None
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            privileged_history_shape=privileged_history_shape,
        )

        # init history
        self.history = torch.zeros((self.env.num_envs, history_length, self.env.num_obs), device=self.device)
        self.privileged_history = None
        if self.use_privileged_history:
            self.privileged_history = torch.zeros(
                (self.env.num_envs, history_length, self.env.num_privileged_obs),
                device=self.device,
            )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
        if self.log_dir is not None and self.env.cfg.env.test is False:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            all_cfg = {"train_cfg": train_cfg, "env_cfg": class_to_dict(self.env.cfg)}
            yaml.safe_dump(all_cfg, open(os.path.join(self.log_dir, 'config.yaml'), 'w'))
        
        # robogauge client
        try:
            if not train_cfg['robogauge']['enabled']:
                raise ImportError("config disabled")
            from robogauge.scripts.client import RoboGaugeClient
            self.robogauge_client = RoboGaugeClient(f"http://127.0.0.1:{train_cfg['robogauge']['port']}")
        except Exception as e:
            print(f"[INFO] RoboGauge client could not be initialized: {e}, disabling RoboGauge interface.")
            self.robogauge_client = None

    def _write_domain_rand_metrics(self, it):
        if hasattr(self.env, "payload_masses"):
            payload = self.env.payload_masses.detach().float()
            self.writer.add_scalar('DomainRand/payload_mass_mean', payload.mean().item(), it)
            self.writer.add_scalar('DomainRand/payload_mass_min', payload.min().item(), it)
            self.writer.add_scalar('DomainRand/payload_mass_max', payload.max().item(), it)
        if hasattr(self.env, "base_com_offsets"):
            com = self.env.base_com_offsets.detach().float()
            com_norm = torch.norm(com, dim=1)
            self.writer.add_scalar('DomainRand/base_com_x_mean', com[:, 0].mean().item(), it)
            self.writer.add_scalar('DomainRand/base_com_y_mean', com[:, 1].mean().item(), it)
            self.writer.add_scalar('DomainRand/base_com_z_mean', com[:, 2].mean().item(), it)
            self.writer.add_scalar('DomainRand/base_com_norm_mean', com_norm.mean().item(), it)
            self.writer.add_scalar('DomainRand/base_com_norm_max', com_norm.max().item(), it)
        if hasattr(self.env, "motor_strengths"):
            motor_strength = self.env.motor_strengths.detach().float()
            self.writer.add_scalar('DomainRand/motor_strength_mean', motor_strength.mean().item(), it)
            self.writer.add_scalar('DomainRand/motor_strength_std', motor_strength.std(unbiased=False).item(), it)
        if hasattr(self.env, "friction_coeffs"):
            friction = self.env.friction_coeffs.detach().float()
            self.writer.add_scalar('DomainRand/friction_mean', friction.mean().item(), it)
            self.writer.add_scalar('DomainRand/friction_min', friction.min().item(), it)
            self.writer.add_scalar('DomainRand/friction_max', friction.max().item(), it)
        if hasattr(self.env, "get_current_range"):
            payload_cfg = getattr(self.env.cfg.domain_rand, "payload_mass_curriculum", None)
            com_cfg = getattr(self.env.cfg.domain_rand, "base_com_curriculum", None)
            if payload_cfg is not None:
                payload_range = self.env.get_current_range(payload_cfg, self.env.cfg.domain_rand.added_mass_range)
                self.writer.add_scalar('DomainRand/payload_range_low', payload_range[0], it)
                self.writer.add_scalar('DomainRand/payload_range_high', payload_range[1], it)
            if com_cfg is not None:
                com_range = self.env.get_current_range(com_cfg, self.env.cfg.domain_rand.added_base_com_range)
                self.writer.add_scalar('DomainRand/base_com_range_abs', max(abs(com_range[0]), abs(com_range[1])), it)

    def _write_tracking_metrics(self, it):
        if not all(hasattr(self.env, attr) for attr in ("commands", "base_lin_vel", "base_ang_vel")):
            return
        commands = self.env.commands.detach().float()
        base_lin_vel = self.env.base_lin_vel.detach().float()
        base_ang_vel = self.env.base_ang_vel.detach().float()
        lin_error = commands[:, :2] - base_lin_vel[:, :2]
        yaw_error = commands[:, 2] - base_ang_vel[:, 2]
        self.writer.add_scalar('Command/cmd_x_abs_mean', commands[:, 0].abs().mean().item(), it)
        self.writer.add_scalar('Command/cmd_y_abs_mean', commands[:, 1].abs().mean().item(), it)
        self.writer.add_scalar('Command/cmd_yaw_abs_mean', commands[:, 2].abs().mean().item(), it)
        self.writer.add_scalar('Command/zero_cmd_ratio', (torch.norm(commands[:, :3], dim=1) < 0.1).float().mean().item(), it)
        self.writer.add_scalar('Tracking/lin_vel_error_mean', torch.norm(lin_error, dim=1).mean().item(), it)
        self.writer.add_scalar('Tracking/lin_vel_error_rmse', torch.sqrt(torch.mean(lin_error.pow(2))).item(), it)
        self.writer.add_scalar('Tracking/yaw_vel_error_mean', yaw_error.abs().mean().item(), it)
        self.writer.add_scalar('Tracking/yaw_vel_error_rmse', torch.sqrt(torch.mean(yaw_error.pow(2))).item(), it)
        self.writer.add_scalar('State/base_lin_vel_x_mean', base_lin_vel[:, 0].mean().item(), it)
        self.writer.add_scalar('State/base_lin_vel_y_mean', base_lin_vel[:, 1].mean().item(), it)
        self.writer.add_scalar('State/base_ang_vel_z_mean', base_ang_vel[:, 2].mean().item(), it)

    def _write_moe_cts_metrics(self, it):
        metrics = getattr(self.alg, "tb_metrics", {})
        if not metrics:
            return
        gate_usage = metrics.get("gate_usage")
        if gate_usage is not None:
            for expert_id, usage in enumerate(gate_usage):
                self.writer.add_scalar(f'MoE/expert_{expert_id}_usage', float(usage), it)
        if "gate_entropy" in metrics:
            self.writer.add_scalar('MoE/gate_entropy', metrics["gate_entropy"], it)
        if "gate_usage_max" in metrics:
            self.writer.add_scalar('MoE/max_expert_usage', metrics["gate_usage_max"], it)
        if "gate_usage_min" in metrics:
            self.writer.add_scalar('MoE/min_expert_usage', metrics["gate_usage_min"], it)
        if "gate_usage_std" in metrics:
            self.writer.add_scalar('MoE/usage_std', metrics["gate_usage_std"], it)
        if "latent_l2" in metrics:
            self.writer.add_scalar('CTS/latent_l2', metrics["latent_l2"], it)
        if "latent_cosine_similarity" in metrics:
            self.writer.add_scalar('CTS/latent_cosine_similarity', metrics["latent_cosine_similarity"], it)
        if "priv_history_last_abs_error_mean" in metrics:
            self.writer.add_scalar(
                'Debug/priv_history_last_abs_error_mean',
                metrics["priv_history_last_abs_error_mean"],
                it,
            )
        if "priv_history_last_abs_error_max" in metrics:
            self.writer.add_scalar(
                'Debug/priv_history_last_abs_error_max',
                metrics["priv_history_last_abs_error_max"],
                it,
            )
        if "stable_swav_loss" in metrics:
            self.writer.add_scalar('SwAV/stable_loss', metrics["stable_swav_loss"], it)
        if "stable_proto_entropy" in metrics:
            self.writer.add_scalar('SwAV/prototype_entropy', metrics["stable_proto_entropy"], it)
        if "stable_proto_usage" in metrics:
            self.writer.add_scalar('SwAV/prototype_usage', metrics["stable_proto_usage"], it)
        if "stable_dynamic_corr" in metrics:
            self.writer.add_scalar('SwAV/stable_dynamic_corr', metrics["stable_dynamic_corr"], it)
        film_prefix = "film_"
        for key, value in metrics.items():
            if key.startswith(film_prefix):
                self.writer.add_scalar(f'FiLM/{key[len(film_prefix):]}', value, it)

    def _write_reward_metrics(self, locs):
        count = locs.get('rollout_reward_count', 0)
        if count > 0:
            reward_mean = locs['rollout_reward_sum'] / count
            reward_var = max(locs['rollout_reward_sq_sum'] / count - reward_mean * reward_mean, 0.0)
            self.writer.add_scalar('Reward/step_mean', reward_mean, locs['it'])
            self.writer.add_scalar('Reward/step_std', reward_var ** 0.5, locs['it'])
            self.writer.add_scalar('Reward/step_min', locs['rollout_reward_min'], locs['it'])
            self.writer.add_scalar('Reward/step_max', locs['rollout_reward_max'], locs['it'])
        teacher_count = locs.get('rollout_teacher_reward_count', 0)
        if teacher_count > 0:
            self.writer.add_scalar(
                'Reward/teacher_step_mean',
                locs['rollout_teacher_reward_sum'] / teacher_count,
                locs['it'],
            )
        student_count = locs.get('rollout_student_reward_count', 0)
        if student_count > 0:
            self.writer.add_scalar(
                'Reward/student_step_mean',
                locs['rollout_student_reward_sum'] / student_count,
                locs['it'],
            )
        if hasattr(self.env, "reward_curriculum_scales"):
            for reward_name, scale in sorted(self.env.reward_curriculum_scales.items()):
                self.writer.add_scalar(f'RewardCurriculum/{reward_name}_scale', scale, locs['it'])
        term_sums = locs.get('rollout_reward_term_sums', {})
        term_abs_sums = locs.get('rollout_reward_term_abs_sums', {})
        term_counts = locs.get('rollout_reward_term_counts', {})
        reward_log_scale = 1.0 / max(getattr(self.env, "dt", 1.0), 1e-8)
        for reward_name in sorted(term_sums.keys()):
            count = term_counts.get(reward_name, 0)
            if count == 0:
                continue
            self.writer.add_scalar(
                f'RewardTermsFinal/{reward_name}_mean',
                term_sums[reward_name] / count * reward_log_scale,
                locs['it'],
            )
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        assert privileged_obs is not None
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
        if self.privileged_history is not None:
            self.privileged_history = torch.cat([self.privileged_history[:, 1:], privileged_obs.unsqueeze(1)], dim=1)
        self.alg.model.train() # switch to train mode (for dropout for example)

        ep_infos = []
        teacher_rewbuffer = deque(maxlen=100)
        teacher_lenbuffer = deque(maxlen=100)
        student_rewbuffer = deque(maxlen=100)
        student_lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.start_learning_iteration = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            rollout_reward_sum = 0.0
            rollout_reward_sq_sum = 0.0
            rollout_reward_min = float("inf")
            rollout_reward_max = float("-inf")
            rollout_reward_count = 0
            rollout_teacher_reward_sum = 0.0
            rollout_teacher_reward_count = 0
            rollout_student_reward_sum = 0.0
            rollout_student_reward_count = 0
            rollout_reward_term_sums = {}
            rollout_reward_term_abs_sums = {}
            rollout_reward_term_counts = {}
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if self.privileged_history is not None:
                        actions = self.alg.act(
                            obs,
                            privileged_obs,
                            self.history.flatten(1),
                            self.privileged_history.flatten(1),
                        )
                    else:
                        actions = self.alg.act(obs, privileged_obs, self.history.flatten(1))
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs, rewards, dones = obs.to(self.device), privileged_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    if 'time_outs' in infos:
                        infos['time_outs'] = infos['time_outs'].to(self.device)
                    reward_values = rewards.detach().float()
                    rollout_reward_sum += reward_values.sum().item()
                    rollout_reward_sq_sum += reward_values.pow(2).sum().item()
                    rollout_reward_min = min(rollout_reward_min, reward_values.min().item())
                    rollout_reward_max = max(rollout_reward_max, reward_values.max().item())
                    rollout_reward_count += reward_values.numel()
                    ti, si = self.alg.teacher_env_idxs, self.alg.student_env_idxs
                    rollout_teacher_reward_sum += reward_values[ti].sum().item()
                    rollout_teacher_reward_count += ti.numel()
                    rollout_student_reward_sum += reward_values[si].sum().item()
                    rollout_student_reward_count += si.numel()
                    for reward_name, term_values in getattr(self.env, "last_final_reward_terms", {}).items():
                        term_values = term_values.detach().float()
                        rollout_reward_term_sums[reward_name] = rollout_reward_term_sums.get(reward_name, 0.0) + term_values.sum().item()
                        rollout_reward_term_abs_sums[reward_name] = rollout_reward_term_abs_sums.get(reward_name, 0.0) + term_values.abs().sum().item()
                        rollout_reward_term_counts[reward_name] = rollout_reward_term_counts.get(reward_name, 0) + term_values.numel()
                    self.history[dones > 0] = 0.0
                    self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
                    if self.privileged_history is not None:
                        self.privileged_history[dones > 0] = 0.0
                        self.privileged_history = torch.cat([self.privileged_history[:, 1:], privileged_obs.unsqueeze(1)], dim=1)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.shape[0]:
                            ti = self.alg.teacher_env_idxs
                            teacher_ids = new_ids[torch.isin(new_ids, ti)]
                            student_ids = new_ids[~torch.isin(new_ids, ti)]
                            teacher_rewbuffer.extend(cur_reward_sum[teacher_ids].cpu().numpy().tolist())
                            teacher_lenbuffer.extend(cur_episode_length[teacher_ids].cpu().numpy().tolist())
                            student_rewbuffer.extend(cur_reward_sum[student_ids].cpu().numpy().tolist())
                            student_lenbuffer.extend(cur_episode_length[student_ids].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                if self.cfg["algorithm_class_name"] in ["ACMoECTS", "DualMoECTS"]:
                    self.alg.compute_returns(obs, privileged_obs, self.history.flatten(1))
                else:
                    if self.privileged_history is not None:
                        self.alg.compute_returns(
                            privileged_obs,
                            self.history.flatten(1),
                            self.privileged_history.flatten(1),
                        )
                    else:
                        self.alg.compute_returns(privileged_obs, self.history.flatten(1))
            
            if self.cfg["algorithm_class_name"] in ["CTS", "MCPCTS"]:
                mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss = self.alg.update()
            elif self.cfg["algorithm_class_name"] in ["MoECTS", "MoENGCTS", "ACMoECTS"]:
                mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss, mean_load_balance_loss = self.alg.update()
            elif self.cfg["algorithm_class_name"] == "DualMoECTS":
                mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss, mean_load_balance_loss, mean_actor_load_balance_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration += 1
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), it, False)
            ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), it, True)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if 'terrain' in key:
                    self.writer.add_scalar('Terrain/' + key, value, locs['it'])
                else:
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if 'mcp' not in self.cfg["algorithm_class_name"].lower():
            mean_std = self.alg.model.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/latent', locs['mean_latent_loss'], locs['it'])
        if 'mean_load_balance_loss' in locs:
            self.writer.add_scalar('Loss/load_balance', locs['mean_load_balance_loss'], locs['it'])
        if 'mean_actor_load_balance_loss' in locs:
            self.writer.add_scalar('Loss/actor_load_balance', locs['mean_actor_load_balance_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        if 'mcp' not in self.cfg["algorithm_class_name"].lower():
            self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['teacher_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_teacher_reward', statistics.mean(locs['teacher_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_teacher_episode_length', statistics.mean(locs['teacher_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_teacher_reward/time', statistics.mean(locs['teacher_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_teacher_episode_length/time', statistics.mean(locs['teacher_lenbuffer']), self.tot_time)
        if len(locs['student_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_student_reward', statistics.mean(locs['student_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_student_episode_length', statistics.mean(locs['student_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_student_reward/time', statistics.mean(locs['student_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_student_episode_length/time', statistics.mean(locs['student_lenbuffer']), self.tot_time)
        if len(locs['teacher_rewbuffer']) > 0 and len(locs['student_rewbuffer']) > 0:
            teacher_reward = statistics.mean(locs['teacher_rewbuffer'])
            student_reward = statistics.mean(locs['student_rewbuffer'])
            teacher_len = statistics.mean(locs['teacher_lenbuffer'])
            student_len = statistics.mean(locs['student_lenbuffer'])
            self.writer.add_scalar('CTS/teacher_student_reward_gap', teacher_reward - student_reward, locs['it'])
            self.writer.add_scalar('CTS/teacher_student_episode_length_gap', teacher_len - student_len, locs['it'])
        self._write_domain_rand_metrics(locs['it'])
        self._write_tracking_metrics(locs['it'])
        self._write_moe_cts_metrics(locs['it'])
        self._write_reward_metrics(locs)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                      'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                      f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                      f"""{'Latent loss:':>{pad}} {locs['mean_latent_loss']:.4f}\n""")
        if 'mean_load_balance_loss' in locs:
            log_string += f"""{'Load balance loss:':>{pad}} {locs['mean_load_balance_loss']:.4f}\n"""
        if 'mean_actor_load_balance_loss' in locs:
            log_string += f"""{'Actor load balance loss:':>{pad}} {locs['mean_actor_load_balance_loss']:.4f}\n"""
        if 'mcp' not in self.cfg["algorithm_class_name"].lower():
            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        if len(locs['teacher_rewbuffer']):
            log_string += (f"""{'Mean teacher reward:':>{pad}} {statistics.mean(locs['teacher_rewbuffer']):.2f}\n"""
                           f"""{'Mean teacher episode length:':>{pad}} {statistics.mean(locs['teacher_lenbuffer']):.2f}\n""")
        if len(locs['student_rewbuffer']):
            log_string += (f"""{'Mean student reward:':>{pad}} {statistics.mean(locs['student_rewbuffer']):.2f}\n"""
                           f"""{'Mean student episode length:':>{pad}} {statistics.mean(locs['student_lenbuffer']):.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration - self.start_learning_iteration) * (
                               locs['tot_iter'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, it, last_model, infos=None):
        torch.save({
            'model_state_dict': self.alg.model.state_dict(),
            'optimizer1_state_dict': self.alg.optimizer1.state_dict(),
            'optimizer2_state_dict': self.alg.optimizer2.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)
        self.update_robogauge(it, last_model)
    
    def update_robogauge(self, it, last_model):
        if self.robogauge_client is None:
            return

        try:
            if it % 500 == 0 or last_model:
                # export jit model
                jit_dir = os.path.join(self.log_dir, 'jit_models')
                jit_path = os.path.join(jit_dir, f'policy_jit_{it}.pt')
                export_policy_as_jit(self.alg.model, jit_dir, filename=f'policy_jit_{it}.pt')
                # upload to robogauge
                task_name = 'go2_moe'  # Both cts, moe-cts actor return a tuple `action, (latent, ...)`
                self.robogauge_client.submit_task(
                    model_path=jit_path,
                    step=it,
                    task_name=task_name,
                    experiment_name=self.cfg["experiment_name"]
                )
        except Exception as e:
            print(f"[WARN] RoboGauge submit failed at step {it}: {e}")
            return
        check_times = 1
        if last_model:
            check_times = int(1e9)  # keep checking until manually stopped
        while check_times > 0:
            check_times -= 1
            try:
                self.robogauge_client.monitor_tasks()
            except Exception as e:
                print(f"[WARN] RoboGauge monitor failed at step {it}: {e}")
                break
            results_dir = os.path.join(self.log_dir, 'robogauge_results')
            os.makedirs(results_dir, exist_ok=True)
            result_received = False
            for task_id, resp in self.robogauge_client.response_data.items():
                if not isinstance(resp, dict):
                    print(f"[WARN] RoboGauge returned an invalid response for task {task_id}: {resp}")
                    continue
                results = resp.get('results')
                step = resp.get('step', it)
                if results is None:
                    print(f"[WARN] RoboGauge returned empty results for task {task_id} at step {step}.")
                    continue
                scores = results.get('scores')
                if scores is None:
                    print(f"[WARN] RoboGauge results for task {task_id} at step {step} do not contain 'scores'.")
                    continue
                if step == it:
                    result_received = True
                for key, val in scores.items():
                    self.writer.add_scalar(f'RoboGauge/{key}', val, step)
                results_path = os.path.join(results_dir, f'results_{step}.yaml')
                with open(results_path, 'w', encoding='utf-8') as f:
                    yaml.dump(results, f, allow_unicode=True, sort_keys=False)
            
            if last_model and result_received:
                print(f"RoboGauge result for step {it} received. Exiting wait loop.")
                break

            if check_times > 0:
                print("Sleeping for 1 minute before checking RoboGauge results again...")
                time.sleep(60)  # wait for 1 minute before checking again

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.model.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer1.load_state_dict(loaded_dict['optimizer1_state_dict'])
            self.alg.optimizer2.load_state_dict(loaded_dict['optimizer2_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.model.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.model.to(device)
        return self.alg.model.act_inference
