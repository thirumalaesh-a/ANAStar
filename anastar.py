import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, joint_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
#########################
class Node():
        def __init__(self, parent=None, position= None):
                self.parent = parent
                self.position = position

                self.g = 99999999999999
                self.h = None
                self.e = None

def h_fn(node, goal, h_fn_val):
        if h_fn_val=='octile':
            xdiff = np.abs(node.position[0] - goal.position[0])
            ydiff = np.abs(node.position[1] - goal.position[1])
            cheby_h = np.max([xdiff, ydiff])
            octile_h = cheby_h + ((np.sqrt(2) - 1)*np.min([xdiff, ydiff]))
            h_rot = (np.min(np.array(np.abs(goal.position[2]-node.position[2]), 2*np.pi - np.abs(goal.position[2] - node.position[2]))))**2
            h_tran = octile_h 
            dist = np.sqrt(h_rot + h_tran)

            return dist

        elif h_fn_val=='euclidean':
            h_trans = (node.position[0] - goal.position[0])**2 + (node.position[1] - goal.position[1])**2
            h_rot = (np.min(np.array(np.abs(node.position[2]-goal.position[2]), 2*np.pi - np.abs(node.position[2] - goal.position[2]))))**2
            dist = np.sqrt(h_rot + h_trans)

            return dist
        else: 
            assert False

def c_fn(node1, node2):
    x1, y1 = node1.position[0], node1.position[1]
    x2, y2 = node2.position[0], node2.position[1]
    c_rot = (np.min(np.array(np.abs(node1.position[2]-node2.position[2]), 2*np.pi - np.abs(node1.position[2] - node2.position[2]))))**2
    c_tran = (x2-x1)**2 + (y2-y1)**2
    dist = np.sqrt(c_rot + c_tran)
    return dist

def a_fn(node1, node2):
    x1, y1 = node1.position[0], node1.position[1]
    x2, y2 = node2.position[0], node2.position[1]
    c_rot = (np.min(np.array(np.abs(node1.position[2]-node2.position[2]), 2*np.pi - np.abs(node1.position[2] - node2.position[2]))))**2
    c_tran = (x2-x1)**2 + (y2-y1)**2
    dist = np.sqrt(c_tran)
    return dist

def e_fn(node, GG):
    num = GG - node.g
    assert node.h!=None
    assert node.g!=None
    den = node.h    
    return num/den

def get_expand_nodes(node, incrmts, goal_node, h_fn_val, n=8):
    neighbor_nodes = []
    translation_discretization_x_8_a = incrmts[0]
    translation_discretization_y_8_b = incrmts[1]
    rotation_discretization_8 = incrmts[2]

    increments_8 = np.array([[1, 0, 0],   [-1, 0, 0],  [0, 1, 0],  [0, -1, 0], [0, 0, 1], [0, 0, -1], 
                                [1, 1, 0],   [1, -1, 0],  [-1, 1, 0], [-1, -1, 0],
                                [1, 0, 1],   [1, 0, -1],  [-1, 0, 1], [-1, 0, -1], 
                                [0, 1, 1],   [0, 1, -1],  [0, -1, 1], [0, -1, -1],
                                [1, 1, 1],   [-1, 1, 1],  [1, -1, 1], [1, 1, -1], 
                                [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
                                [-1, -1, -1]], dtype=np.float32)

    increments_8[:,0]*=translation_discretization_x_8_a
    increments_8[:,1]*=translation_discretization_y_8_b
    increments_8[:,2]*=rotation_discretization_8

    if n==8:
        for i in increments_8:
                x,y,t = node.position[0]+i[0], node.position[1]+i[1], node.position[2]+i[2]
                xyt = np.array([x,y,t])
                new_node = Node(None, np.round(xyt, 3))
                new_node.h = h_fn(new_node, goal_node, h_fn_val)
                neighbor_nodes.append(new_node)
        return np.array(neighbor_nodes)

def argmax(open_list):
    nn, mi = open_list[0], 0
    me, mn = nn[0], nn[1]
    for i, nn in enumerate(open_list):
        e, n = nn[0], nn[1]
        if e>me:
            me = e
            mn = n
            mi = i

    return mi, (me, mn)

def goal_check(node, goal, tol=0.1):
    x1, y1 = node.position[0], node.position[1]
    x2, y2 = goal.position[0], goal.position[1]
    t1, t2 = node.position[2], goal.position[2]
    if abs(x1-x2) < tol and abs(y1-y2)<tol and (t1-t2)<tol:
        return True
    else:
        return False

def is_in(olist, node):
    for ind, (i,n) in enumerate(olist):
        if (n.position[0]==node.position[0]) and (n.position[1]==node.position[1]) and (n.position[2]==node.position[2]):
            return True, i, ind
    return False, None, None

def improve_soln(GG, EE, open_list, closed_list, collision_fn, free_set, collision_set, goal, incrmts, h_fn_val):
    
    while(len(open_list)!=0):

        sii, s = argmax(open_list)
        del open_list[sii]
        closed_list.append(s)
        s = s[1]
        if s.e < EE:
            EE = s.e

        if goal_check(s, goal):
            GG = s.g
            return s, True, GG, EE, open_list, closed_list, free_set, collision_set
        
        neighbors = get_expand_nodes(s, incrmts, goal, h_fn_val)
        for n in neighbors:
            c_n = c_fn(s, n)
            
            n_in_closed,_,_ = is_in(closed_list, n)
            if (s.g + c_n) < n.g and not collision_fn(n.position) and not n_in_closed: 
                
                n.g = s.g + c_n
                n.parent = s
                
                if n.g + n.h < GG:
                    n.e = e_fn(n, GG)
                    isin, _, ind =  is_in(open_list, n)
                    if not isin:
                        open_list.append((n.e, n))
                    else:
                        del open_list[ind]
                        open_list.append((n.e, n))

            elif collision_fn(n.position):
                collision_set.add((n.position[0], n.position[1], 0.5))

            elif (s.g + c_n) < n.g and n_in_closed:
                free_set.add((n.position[0], n.position[1], 0.5))
                
    return s, False, GG, EE, open_list, closed_list, free_set, collision_set
    

def report_e(term, atleast_one_path):
    if term and not atleast_one_path:
        print("path_found!!!")
    elif not term and atleast_one_path:
        print("Possible path already found")
    elif term and atleast_one_path:
        print("New path found")
    elif not term and not atleast_one_path:
        print("Path not found")

def update_open(open_list, GG):
    updated_open = []
    for e,s in open_list:
        s.e = e_fn(s, GG)
        updated_open.append((s.e, s))
    return updated_open

def prune(open, GG):
    pruned_open = []
    for e,s in open:
        if s.g + s.h < GG:
            pruned_open.append((e,s))
    return pruned_open

def anastar(env, goal_config, incrmts, h_fn_val='euclidean'):
    connect(use_gui=True)
    robots, obstacles = load_env(env)

    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
   
    PR2 = robots['pr2']

    GG, EE, open_list, closed_list = 99999999999999,99999999999999,[],[]

    start_node = Node(None, start_config)
    goal_node = Node(None, goal_config)
    # goal_node.h=0
    # goal_node.e = float('inf')

    start_node.g = 0
    start_node.h = h_fn(start_node, goal_node, h_fn_val)
    start_node.e = e_fn(start_node, GG)

    all_paths = {}

    collision_set = set()
    free_set = set()
    all_paths_set = {}
    times = []

    draw_sphere_marker((goal_config[0], goal_config[1], 0.5), 0.1, (0, 0, 0, 1))
    atleast_one_path = False

    temp_time = []
    start_time = time.time()
    time_for_list = start_time
    
    open_list.append((start_node.e, start_node))
    while(len(open_list)!=0):
        s, term, GG, EE, open_list, closed_list, free_set, collision_set = improve_soln(GG, EE, open_list, closed_list, collision_fn, free_set, collision_set, goal_node, incrmts, h_fn_val)
        report_e(term, atleast_one_path)
        if term:
            temp_time_val = time.time()
            times.append(temp_time_val-time_for_list)
            atleast_one_path = True
            
            path = []
            path_set = set()
            total_cost = 0
            
            goalreached = s

            path.append(goalreached.position)            
            path_set.add((goalreached.position[0], goalreached.position[1], 0.5))    

            while(goalreached.parent!=None):
                
                total_cost+=a_fn(goalreached.parent, goalreached)
                path.append(goalreached.parent.position)
                path_set.add((goalreached.parent.position[0], goalreached.parent.position[1], 0.5))
                goalreached = goalreached.parent

            all_paths[total_cost] = path
            all_paths_set[total_cost] = path_set
            path.reverse()

            # print("Cost of Path= ",total_cost)
            temp_time_val-= time.time()
            temp_time_val*=-1
            temp_time.append(temp_time_val)
            time_for_list = time.time()
            closed_list = []

        open_list = update_open(open_list, GG)
        open_list = prune(open_list, GG)

    end_time = time.time()

    costs = [i for i in all_paths.keys()]

    lowest_cost = min(costs)
    # print("Lowest Cost= ", lowest_cost
    path = all_paths[lowest_cost]
    paths_set = set()
    
    for _,points in all_paths_set.items():
        for point in points:
            paths_set.add(point)
    
    for point in all_paths[lowest_cost]:
        point = (point[0], point[1], 0.5)
        draw_sphere_marker(point, 0.09, (0, 1, 0, 1))
    
    for point in paths_set:
        if point not in all_paths_set[lowest_cost]:
            draw_sphere_marker(point, 0.09, (0, 0, 0, 1))

    for point in collision_set:
        if point not in paths_set:
            draw_sphere_marker(point, 0.09, (1, 0, 0, 1))

    for point in free_set:
        if point not in collision_set and point not in paths_set:
            draw_sphere_marker(point, 0.09, (0, 0, 1, 1))

    temp=0
    for i in temp_time:
        temp+=i
    run_time = (end_time - start_time) - temp 


    print("Planner run time: ", run_time)
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # wait_if_gui()
    
    # left = -2.0
    # right = 2.0
    # bottom = -1.5
    # top = 1.5

    # near = 5
    # far = 50.0

    # projection_matrix = pybullet.computeProjectionMatrix(left, right, bottom, top, near, far)
    # view_matrix = pybullet.computeViewMatrix([2, 2, 20], [0, 0, 0.5], [0, 0, 1])
    # _, _, rgb_array, _, _ = pybullet.getCameraImage(
    # 1024,
    # 1024,
    # view_matrix,
    # projection_matrix,
    # renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

    # image = Image.fromarray(rgb_array)

    # image.save(f"{env[:7]}_ANASTAR1_{h_fn_val}.png")
    time.sleep(3)
    disconnect()

    return lowest_cost, run_time, times, costs

# anastar('blocked.json', (1.8, -1.8, -np.pi/2), (.4,.4,np.pi/4), 'euclidean')
# anastar('pr2doorway.json', (2.6, -1.3, -np.pi/2), (.5,.1,np.pi/4), 'octile')
# anastar('pr2doorway.json', (2.6, -1.3, -np.pi/2), (.5,.1,np.pi/4), 'euclidean')
# anastar('move_several_10.json', (1.8, -1.8, -np.pi), (.2,.1,np.pi/4), 'octile')
# anastar('move_several_10.json', (1.8, -1.8, -np.pi), (.2,.1,np.pi/4), 'euclidean')

# anastar('pr2doorway.json', (2.6, -1.3, -np.pi/2), (.1,.1,np.pi/4), 'octile')
# anastar('move_several_10.json', (1.8, -1.8, -np.pi), (.1,.1,np.pi/4), 'octile')
