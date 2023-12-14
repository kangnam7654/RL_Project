from pathlib import Path
import argparse
import torch
import torch.nn as nn
import numpy as np
from models.QNet import QNet
from game.snake import SnakeGame
import random
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_size",
        type=int,
        default=84,
        help="The common width and height for all images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The number of images per batch"
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["sgd", "adam"], default="adam"
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument(
        "--replay_memory_size",
        type=int,
        default=5000,
        help="Number of epoches between testing phases",
    )
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    return args


def pre_processing(image, device, width=256, height=256):
    image = cv2.cvtColor(cv2.resize(image, (width, height)), cv2.COLOR_BGR2GRAY)
    image = torch.from_numpy(image)
    image = (image / 127.5) - 1
    return image.unsqueeze(0).to(device).type(torch.float32)


def main(args):
    if torch.backends.mps.is_available():
        device = "mps" 
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = QNet().to(device)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    game_state = SnakeGame()
    image, reward, terminal = game_state.frame(1)

    image = pre_processing(image, device)
    state = torch.cat([image for _ in range(4)], 0).unsqueeze(0)
    replay_memory = []
    iteration = 0

    while iteration < args.num_iters:
        prediction = model(state)
        epsilon = args.final_epsilon + (
            (args.num_iters - iteration)
            * (args.initial_epsilon - args.final_epsilon)
            / args.num_iters
        )

        random_action = np.random.rand() <= epsilon

        if random_action:
            action = torch.tensor((np.random.randint(0, 4)))
            print("RANDOM ACTION!!!")
        else:
            action = torch.argmax(prediction)

        next_image, reward, terminal = game_state.frame(action.item())
        next_image = pre_processing(next_image, device)
        next_state = torch.cat((state[0, 1:, :, :], next_image)).unsqueeze(0)

        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > args.replay_memory_size:
            del replay_memory[0]

        batch = random.sample(replay_memory, min(len(replay_memory), args.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(
            *batch
        )
        state_batch = torch.cat(tuple(state for state in state_batch))

        action_list = []
        for action in action_batch:
            one_hot = [0, 0, 0, 0]
            one_hot[action.item()] = 1
            action_list.append(one_hot)
        action_batch = torch.tensor(action_list, dtype=torch.float32)

        reward_batch = torch.tensor(reward_batch).unsqueeze(0)
        next_state_batch = torch.cat([state for state in next_state_batch])

        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)

        current_predcition_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            [
                reward if terminal else reward + 0.99 * torch.max(prediction)
                for reward, terminal, prediction in zip(
                    reward_batch, terminal_batch, next_prediction_batch
                )
            ]
        )
        q_value = torch.sum(current_predcition_batch * action_batch, dim=1)

        optimizer.zero_grad()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iteration += 1
        print(
            "Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                iteration,
                args.num_iters,
                action,
                loss,
                epsilon,
                reward,
                torch.max(prediction),
            )
        )
        if (iteration + 1) % 50000 == 0:
            if not Path(args.saved_path).is_dir():
                Path(args.saved_path).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "{}/snake{}.pt".format(args.saved_path, iteration + 1))
    torch.save(model.state_dict(), "{}/snake.pt".format(args.saved_path))


if __name__ == "__main__":
    args = get_args()
    main(args)
