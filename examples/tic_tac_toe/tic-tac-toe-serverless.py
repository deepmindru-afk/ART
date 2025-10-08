import argparse
import asyncio
import os
import random

from dotenv import load_dotenv
from art.serverless.backend import ServerlessBackend
from rollout import TicTacToeScenario, rollout

import art

load_dotenv()

random.seed(42)

STEPS = 1000
GENERATE_BENCHMARKS = False

async def main_0(api_key: str, entity: str, project: str, model_name: str):
    backend = ServerlessBackend(
        base_url="https://api.training.wandb.ai/v1",
        api_key=api_key,
    )

    model = art.TrainableModel(
        name=model_name,
        project=project,
        entity=entity,
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )


    print("registering")
    await model.register(backend)

    print(model.get_inference_name())

    print("training")
    for i in range(await model.get_step(), STEPS):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, TicTacToeScenario(step=i)) for _ in range(96)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=5e-5))


    lora_model = art.Model(
        name=f"deployed-{model.name}-{i}",
        project="tic-tac-toe",
        inference_api_key=os.environ["WANDB_API_KEY"],
        inference_base_url="https://api.inference.wandb.ai/",
        inference_model_name=model.get_inference_name(),
    )

    print("Starting a rollout using the deployed model!")
    traj = await rollout(lora_model, TicTacToeScenario(step=0))

    print(traj)

    if GENERATE_BENCHMARKS:
        gpt_4o_mini = art.Model(
            name="gpt-4o-mini",
            project="tic-tac-toe",
            inference_model_name="gpt-4o-mini",
            inference_api_key=os.getenv("OPENROUTER_API_KEY"),
            inference_base_url="https://api.openrouter.ai/v1",
        )

        gpt_4o = art.Model(
            name="gpt-4o",
            project="tic-tac-toe",
            inference_model_name="gpt-4o",
            inference_api_key=os.getenv("OPENROUTER_API_KEY"),
            inference_base_url="https://api.openrouter.ai/v1",
        )

        async def benchmark_comparison_model(comparison_model: art.Model):
            trajectories = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(comparison_model, TicTacToeScenario(step=0))
                        for _ in range(12)
                    )
                    for _ in range(1)
                ),
                pbar_desc=f"gather {comparison_model.name}",
                max_exceptions=1,
            )
            await comparison_model.log(
                trajectories,
                split="val",
            )

        await benchmark_comparison_model(gpt_4o_mini)
        await benchmark_comparison_model(gpt_4o)

async def main_1():
    backend = ServerlessBackend(
        base_url="https://api.training.wandb.ai/v1",
        # base_url="http://166.19.34.97:8000/v1",
        api_key=os.environ["WANDB_API_KEY"],
    )

    model = art.TrainableModel(
        name="agent-005",
        project="tic-tac-toe",
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )


    print("registering")
    await model.register(backend)

    lora_model = art.Model(
        name=f"deployed-{model.name}",
        project="tic-tac-toe",
        inference_api_key=os.environ["WANDB_API_KEY"],
        # inference_base_url="https://api.inference.wandb.ai/v1",
        inference_base_url="https://api.inference.coreweave.com/v1",
        inference_model_name="wandb-artifact:///davidcorbittihs-test-guide/tic-tac-toe/agent-002:step0",
    )

    print("Starting a rollout using the deployed model!")
    traj = await rollout(lora_model, TicTacToeScenario(step=0))

    print(len(traj.messages_and_choices))

    # Make 1000 simultaneous rollouts
    tasks = [rollout(lora_model, TicTacToeScenario(step=0)) for _ in range(1000)]
    results = await asyncio.gather(*tasks)
    print(len(results))


# Determine whether to run first or second from CLI argument
parser = argparse.ArgumentParser(description="Train a model to play Tic-Tac-Toe")
parser.add_argument("--run-first", action="store_true", help="Run the first model")
parser.add_argument("--run-second", action="store_true", help="Run the second model")
parser.add_argument("--run-third", action="store_true", help="Run the third model")
parser.add_argument("--run-fourth", action="store_true", help="Run the fourth model")



args = parser.parse_args()

if args.run_first:
    asyncio.run(main_0(api_key=os.environ["WANDB_API_KEY"], entity="davidcorbittihs-test-guides", project="tic-tac-toe", model_name="ageng-007"))
if args.run_second:
    asyncio.run(main_0(api_key=os.environ["WANDB_API_KEY_2"], entity="wandb", project="tic-tac-toe-2", model_name="agent-008"))
if args.run_third:
    asyncio.run(main_0(api_key=os.environ["WANDB_API_KEY"], entity="wandb", project="davidcorbittihs-test-guide", model_name="agent-009"))
if args.run_fourth:
    asyncio.run(main_0(api_key=os.environ["WANDB_API_KEY_2"], entity="wandb", project="tic-tac-toe-2", model_name="agent-010"))



# if __name__ == "__main__":
    # asyncio.run(main_0(api_key=os.environ["WANDB_API_KEY"], entity="davidcorbittihs-test-guide", project="tic-tac-toe"))
    # asyncio.run(main_0(api_key=os.environ["WANDB_API_KEY_2"], entity="wandb", project="tic-tac-toe-2"))
    # asyncio.run(main_1())
    # asyncio.run(main())
