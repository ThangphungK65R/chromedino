"""
@tác giả: Việt Nguyễn <nhviet1009@gmail.com>
"""
import argparse
import torch

from src.model import DeepQNetwork
from src.env import ChromeDino
import cv2


def get_args():
    parser = argparse.ArgumentParser(
        """Triển khai Deep Q Network để chơi Chrome Dino""")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--fps", type=int, default=60, help="frames per second")
    parser.add_argument("--output", type=str, default="output/chrome_dino.mp4", help="the path to output video")

    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)
    model = DeepQNetwork()
    checkpoint_path = "{}/chrome_dino.pth".format(opt.saved_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Thêm map_location để nạp model lên CPU
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    env = ChromeDino()
    state, raw_state, _, _ = env.step(0, True)
    state = torch.cat(tuple(state for _ in range(4)))[None, :, :, :]
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps, (600, 150))
    done = False
    while not done:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        next_state, raw_next_state, reward, done = env.step(action, True)
        out.write(raw_next_state)
        next_state = torch.cat((state[0, 1:, :, :], next_state))[None, :, :, :]
        state = next_state


if __name__ == "__main__":
    opt = get_args()
    test(opt)
