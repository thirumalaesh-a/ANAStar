import numpy as np
from anastar import *
from astar import *
import time
# envs_goal_inc = {'blocked.json':        [(1.8, -1.8, -np.pi/2), (.4,.4,np.pi/4)],
#                  'pr2doorway.json':     [(2.6, -1.3, -np.pi/2), (.5,.1,np.pi/4)], 
#                  'move_several_10.json':[(1.8, -1.8, -np.pi), (.2,.1,np.pi/4)]
#                  }

# heuristics = ['octile','euclidean']

envs_goal_inc = {'blocked.json':        [(1.8, -1.8, -np.pi/2), (.4,.4,np.pi/4)],
                 'pr2doorway.json':     [(2.6, -1.3, -np.pi/2), (.5,.1,np.pi/4)]
                #  'move_several_10.json':[(1.8, -1.8, -np.pi), (.2,.1,np.pi/4)]
                 }

heuristics = ['octile']

print("Estimated RunTime: 3-7 minutes")
start_time = time.time()
with open("log_demo.txt", 'w') as f:
    f.write(f"Start time: {start_time}")
    for env, goal_inc in envs_goal_inc.items():
        for h in heuristics:
            print(f"Running ANA*: \n Env: {env} \n Heuristic: {h} \n Goal: {goal_inc[0]} \n")
            f.write(f"Running ANA*: \n Env: {env} \n Heuristic: {h} \n Goal: {goal_inc[0]} \n")
            c, t, ts, cs = anastar(env, goal_inc[0], goal_inc[1], h)
            f.write(f" Lowest Cost = {c} \n Time: {t} \n")
            for d,g in zip(ts, cs):
                f.write(f" Cost = {g} \n Time: {d} \n")
            f.write("\n")
            print(f"Running A*: \n Env: {env} \n Heuristic: {h} \n Goal: {goal_inc[0]}")
            f.write(f"Running A*: \n Env: {env} \n Heuristic: {h} \n Goal: {goal_inc[0]} \n")
            c, t = astar(env, goal_inc[0], goal_inc[1], h)
            f.write(f"Cost = {c} \n Time: {t} \n")
        
    end_time = time.time()
    f.write(f"End time: {end_time} \n")
    f.write(f"Total Demo time: {end_time - start_time} \n")

    f.close()



