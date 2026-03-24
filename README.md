# MAZE RUNNER

Your goal is to build a policy that solves mazes. We provide a gym environment
that will generate mazes and allow the policy to interact with them.

The policy is given:
- The entire pixel view of the world including the goal square and its current location

The policy will return:
- What direction to move

We offer you a very basic untrained policy (single layer) and the means to run it.

Your goal is to improve it.


## Running this repo

To see the policy attempt a maze:

`uv run viz`

This starts a local web server and opens the maze in your browser (default `http://127.0.0.1:8000/`). Use **Play** / **Pause**, **Restart** (same maze, agent reset), **New Maze**, and the **Policy** selector to swap between the baseline model and a `random_choice` policy. The size and random sliders apply on **New Maze**; changing policy keeps the same maze and resets the robot to start. The policy steps every 0.2s while playing until success, then pauses.

---

To run bulk evaluation of the policy:

`uv run stats`

This will run the model and a `random_choice` baseline on the same 100 mazes by default (headless, at speed), show a progress bar, print a side-by-side comparison table, and save it to `stats/stats-compare-$timestamp.csv`


## The mazes

Mazes are generated procedurally. A few global variables (in python code) control this:
```
SIZE = 16 #(side in pixels)
RANDOM = 55 #(percentage, 0 random is simplest, 100 is most)
```

The start and end positions are randomly picked from the outer perimeter of the maze. A path is then drawn via random walk from start to end. `RANDOM` controls both how direct that main walk is and how much fractal corridor branching grows off the carved route. Every block of path drawn is constrained to:
1. Be continuous and connect on one side from a previous block
2. Not form a walkable 2x2 (e.g. our maze walkways are narrow)

Generation is also capped at 50% walkable density, and side-corridor growth prefers not to reconnect into existing paths, which keeps high-random mazes from collapsing into overly connected checkerboard patterns.

You are welcome to adjust these parameters during the exercise. You have to decide how to balance performing well on simple mazes versus performing worse (but admirably) on hard mazes

## Requirements

This repo should be runnable with CPU or any common GPU (e.g. Mac, Nvidia, AMD). The dependencies should be easy to install on Mac or Windows or Linux.

You will not be judged by the compute (e.g. if you only train on a CPU this will not penalize you), but you will be judged by how you choose to deploy the compute.

You have the hour of this interview to work on this, and will be judged on your ability to come up with training approaches, execute them, and reflect and learn from them.