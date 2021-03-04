import numpy as np
import pandas as pd
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
from srl_wrapper import SRLEnv
import copy
import matplotlib.pyplot as plt

import utils
from logger import Logger
from video import VideoRecorder

from curl_sac import CurlSacAgent
from sac_ae import SacAeAgent
from cfrl_sac import CfrlSacAgent
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--environment', type=str, default='mobile')
    parser.add_argument('--srl_model', type=str, default='raw_pixels')
    parser.add_argument('--renders', default=False, action='store_true')
    parser.add_argument('--is_discrete', default=False, action='store_true')
    parser.add_argument('--force_down', default=True, action='store_true')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--reward_type', default='dist', type=str)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--pre_training_steps', default=200000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--eval_ul_freq', default=3000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lam', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lam', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, replay_buffer, video, num_episodes, L, csv_dir, log_csv,
             image_dir, step, args, goal_sample=None):
    all_ep_rewards = []
    all_ep_distance = []
    image_dir = utils.make_dir(os.path.join(image_dir, str(step)))

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            image_log_dir = utils.make_dir(os.path.join(image_dir, str(i)))
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                # center crop image
                if episode_step % 100 == 0:
                    observation  = env.render("rgb_array")
                    plt.imsave(image_log_dir + "/result_" + str(episode_step) + ".png", observation)
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                    goal_obs = utils.center_crop_image(goal_sample, args.image_size)
                else:
                    goal_obs = goal_sample
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs, goal_obs)
                    else:
                        action = agent.select_action(obs, goal_obs)
                obs, reward, done, distance = env.step(action)
                if args.reward_type == 'dist':
                    reward = agent.dist_reward(obs, goal_sample)
                video.record(env)
                episode_reward += reward
                episode_step += 1
                if done:
                    observation  = env.render("rgb_array")
                    plt.imsave(image_log_dir + "/result_final.png", observation)

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
            all_ep_distance.append(distance)

        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        mean_ep_distance = np.mean(all_ep_distance)
        best_ep_distance = np.max(all_ep_distance)
        std_ep_distance = np.std(all_ep_distance)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)
        L.log('eval/' + prefix + 'mean_distance_to_goal', mean_ep_distance, step)
        L.log('eval/' + prefix + 'best_distance_to_goal', best_ep_distance, step)

        # Log to csv.
        log_csv["step"].append(step)
        log_csv["mean_reward"].append(mean_ep_reward)
        log_csv["mean_distance_to_goal"].append(mean_ep_distance)
        log_csv["std_distance_to_goal"].append(std_ep_distance)
        pd.DataFrame(log_csv).to_csv(csv_dir + "/log.csv", index=False)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            pre_training_steps=args.pre_training_steps

        )

    elif args.agent == 'sac_ae':
        return SacAeAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lam=args.decoder_latent_lam,
            decoder_weight_lam=args.decoder_weight_lam,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            pre_training_steps=args.pre_training_steps
        )

    elif args.agent == 'cfrl_sac':
        return CfrlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim,
            pre_training_steps=args.pre_training_steps

        )

    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)

    goal_env = SRLEnv(args.action_repeat,
                 args.environment,
                 args.srl_model,
                 args.pre_transform_image_size,
                 args.pre_transform_image_size,
                 args.renders,
                 args.is_discrete,
                 args.force_down)

    goal_env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        goal_env = utils.FrameStack(goal_env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H:%M:%S", ts)
    env_name = args.environment
    exp_name = env_name + '-' + args.agent + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.work_dir = args.work_dir + '/'  + exp_name

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))
    pre_buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'pre_buffer'))
    csv_dir = utils.make_dir(os.path.join(args.work_dir, 'csv'))
    image_dir = utils.make_dir(os.path.join(args.work_dir, 'image'))

    log_csv = {"step": [], "mean_reward": [], "mean_distance_to_goal": [],
               "std_distance_to_goal": []}

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = goal_env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        if args.agent == 'sac_ae':
            pre_aug_obs_shape = obs_shape
        else:
            pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
    else:
        obs_shape = goal_env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    if args.reward_type == 'dist':
        success_samples = goal_env.get_goal_image()
        goal_env.close()

        def sample_goal():
            success_sample = random.choice(success_samples)
            if args.encoder_type == 'pixel':
                frames = []
                for _ in range(args.frame_stack):
                    frames.append(success_sample)

                return np.concatenate(frames, axis=0)
            else:
                return success_sample

    env = SRLEnv(args.action_repeat,
                 args.environment,
                 args.srl_model,
                 args.pre_transform_image_size,
                 args.pre_transform_image_size)


    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    pre_replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.pre_training_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                pre_replay_buffer.save(pre_buffer_dir)

        if done:
            obs = env.reset()
            done = False
            episode_step = 0
            episode += 1

        # sample action for data collection
        action = env.action_space.sample()
        if args.environment == 'kuka':
            action[2] = -abs(action[2])

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(pre_replay_buffer, L, step, enc_train=True)

        next_obs, reward, done, distance = env.step(action)
        goal_obs = sample_goal()
        if args.reward_type == 'dist':
            reward = agent.dist_reward(next_obs, goal_obs)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        pre_replay_buffer.add(obs, action, reward, next_obs, goal_obs, done_bool)

        obs = next_obs
        episode_step += 1

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            if args.reward_type == 'dist':
                evaluate(env, agent, replay_buffer, video, args.num_eval_episodes,
                L, csv_dir, log_csv, image_dir, step, args, sample_goal())
            else:
                evaluate(env, agent, replay_buffer, video, args.num_eval_episodes,
                L, csv_dir, log_csv, image_dir, step, args, None)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            goal_obs = sample_goal()
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs, goal_obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step, enc_train=False)

        next_obs, reward, done, distance = env.step(action)
        if args.reward_type == 'dist':
            reward = agent.dist_reward(next_obs, goal_obs)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, goal_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
