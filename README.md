# READ ME

## Baseline

**file:** gym-pybullet-drones/gym_pybullet_drones/examples/RRT_trial.py

**objective:** Develop a baseline, which is a simple RRT

**how to run:**

Use the commond below to run the code:

```bash
python RRT_trial.py
```

I also added a debug configuration while I was trying to fix some problems, which I didn't manage to fix... but it can show the trees RRT explores, which probably would be useful in the future. So if you want to see the debug process, use the following command:

```bash
python RRT_trial.py --debug
```
*Reminder: the running time would be super long if you use debug.*

**Probelm:**

After some debugging, I found the problem is when the path recheaes the goal, the code gets stuck, more specifically, in function ```nearest_neighbor```. 