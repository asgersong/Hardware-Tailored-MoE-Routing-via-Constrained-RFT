import multiprocessing as mp
from typing import Callable, Dict

import torch

from rft_moe.utils.common import set_seed


def actor_loop(
    rank: int,
    model_builder: Callable[[], torch.nn.Module],
    rollout_fn: Callable[[torch.nn.Module], Dict],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    set_seed(42 + rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model_builder().to(device)
    model.eval()
    while True:
        msg = task_queue.get()
        if msg == "STOP":
            break
        state_dict = msg
        model.load_state_dict(state_dict, strict=False)
        rollout = rollout_fn(model)
        result_queue.put(rollout)


def learner_loop(
    world_size: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update_fn: Callable[[torch.nn.Module, torch.optim.Optimizer, Dict], None],
    rollout_fn: Callable[[torch.nn.Module], Dict],
    max_iters: int = 100,
):
    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    processes = []
    for rank in range(1, world_size):
        p = mp.Process(
            target=actor_loop,
            args=(rank, lambda: type(model)(model.config), rollout_fn, task_queue, result_queue),
        )
        p.start()
        processes.append(p)

    try:
        for _ in range(max_iters):
            state_dict = model.state_dict()
            for _ in range(world_size - 1):
                task_queue.put(state_dict)
            local_rollout = rollout_fn(model)
            rollouts = [local_rollout]
            for _ in range(world_size - 1):
                rollouts.append(result_queue.get())
            update_fn(model, optimizer, rollouts)
    finally:
        for _ in processes:
            task_queue.put("STOP")
        for p in processes:
            p.join()

