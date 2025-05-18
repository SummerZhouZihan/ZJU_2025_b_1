## ZJU_2025_b



## Key dependencies based (Lower versions are not guaranteed to be feasible):

`python`: 3.8.20

`numpy`: 1.23.1

`gym`: 0.19.0

`gymnasium`:1.1.1

`pillow`: 10.4.0

`torch`: 1.10.0+cu113

`torchaudio`: 0.10.0+cu113

`torchvision`: 0.11.1+cu113

## Explanation of Document

- `agent`/`buffer`/`maddpg`/`networks`: Refer to Phil's work -> [PhilMADDPG](https://github.com/philtabor/Multi-Agent-Reinforcement-Learning);

- `sim_env`: Customized Multi-UAV round-up environment;

- `main`: Main loop to train agents;

- `main_evaluate`: Only rendering part is retained in `main`, in order to evaluate models (a set of feasible models is provided in `tmp/maddpg/UAV_Round_up`;

- `math_tool`: some math-relevant functions.

- `clean`:清除tmp文件夹中的历史记录
