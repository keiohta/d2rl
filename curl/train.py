import os
import time
import argparse

import numpy as np
import torch
import dmc2gym
import wandb
import tensorflow as tf
from tf2rl.misc.prepare_output_dir import prepare_output_dir
from tf2rl.misc.initialize_logger import initialize_logger

import utils
from curl_sac import CurlSacAgent


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--env', default='cheetah',
                        choices=['cheetah', 'finger', 'cartpole', 'reacher', 'walker', 'ball', 'humanoid', 'bring_ball',
                                 'bring_peg', 'insert_ball', 'insert_peg'])
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=501000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)  # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2,
                        type=int)  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--num_mlp_layers', default=4, type=int)
    parser.add_argument('--logdir', type=str, default="results")

    args = parser.parse_args()
    return args


def evaluate(env, agent, num_episodes, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        avg_test_return = 0.
        avg_test_steps = 0
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                avg_test_steps += 1
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            avg_test_return += episode_reward
            all_ep_rewards.append(episode_reward)

        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)

        logger.log({
            'mean_reward': mean_ep_reward,
            'max_reward': best_ep_reward})
        return avg_test_return / num_episodes, avg_test_steps / num_episodes

    return run_eval_loop(sample_stochastically=False)


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
            num_mlp_layers=args.num_mlp_layers,
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def main():
    args = parse_args()

    dm_envs = {
        'finger': ['finger', 'spin'],
        'cartpole': ['cartpole', 'swingup'],
        'reacher': ['reacher', 'easy'],
        'cheetah': ['cheetah', 'run'],
        'walker': ['walker', 'walk'],
        'ball': ['ball_in_cup', 'catch'],
        'humanoid': ['humanoid', 'stand'],
        'bring_ball': ['manipulator', 'bring_ball'],
        'bring_peg': ['manipulator', 'bring_peg'],
        'insert_ball': ['manipulator', 'insert_ball'],
        'insert_peg': ['manipulator', 'insert_peg'],
    }

    if args.env == 'cartpole':
        args.action_repeat = 8
    elif args.env in ['finger', 'walker']:
        args.action_repeat = 2
    else:
        args.action_repeat = 4

    args.domain_name, args.task_name = dm_envs[args.env]

    global logger
    logger = wandb.init(
        project='d2rl',
        config=args,
        dir='wandb_logs',
        group='{}_{}'.format(args.domain_name, args.task_name))

    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat
    )

    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) + '-b' \
               + str(args.batch_size) + '-s' + str(args.seed) + '-' + args.encoder_type

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
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

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    logdir = prepare_output_dir(args, user_specified_dir=args.logdir)
    model_dir = os.path.join(logdir, "models")
    buffer_dir = os.path.join(logdir, "buffer")

    console_logger = initialize_logger(output_dir=logdir)
    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()

    tf.summary.experimental.set_step(0)

    episode_start_time = time.perf_counter()
    for step in range(args.num_train_steps):
        # evaluate agent periodically
        if step % args.eval_freq == 0:
            tf.summary.experimental.set_step(step)
            avg_test_return, avg_test_steps = evaluate(env, agent, args.num_eval_episodes, step, args)
            console_logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                step, avg_test_return, args.num_eval_episodes))
            tf.summary.scalar(
                name="Common/average_test_return", data=avg_test_return)
            tf.summary.scalar(
                name="Common/average_test_episode_length", data=avg_test_steps)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            tf.summary.experimental.set_step(step)
            fps = episode_step / (time.perf_counter() - episode_start_time)
            console_logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                episode, step, episode_step, episode_reward, fps))
            tf.summary.scalar(name="Common/training_return", data=episode_reward)
            tf.summary.scalar(name="Common/training_episode_length", data=episode_step)
            obs = env.reset()
            episode_start_time = time.perf_counter()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, step)

        next_obs, reward, done, _ = env.step(action)
        episode_step += 1

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
