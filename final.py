import numpy as np
import time
%matplotlib qt
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import Planner
from pyrr import geometric_tests as gt
from pqdict import pqdict
import math
import plotly
from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.obstacle_generation import obstacle_generator
from src.utilities.plotting import Plot
from src.rrt.rrt_star_bid_h import RRTStarBidirectionalHeuristic

def tic():
  return time.time()

def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def load_map(fname):
  '''
  Loads the bounady and blocks from map file fname.
  
  boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  
  blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  '''
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks

def draw_block_list(ax,blocks):
  '''
  Subroutine used by draw_map() to display the environment blocks
  '''
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h

def draw_map(boundary, blocks, start, goal):
  '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])  
  return fig, ax, hb, hs, hg

def state_from_indices(x,y,z,x_coord,y_coord,z_coord):
    '''
    a mapping between the coordinates and their indices
    '''
    return np.array([x_coord[x],y_coord[y],z_coord[z]])

def is_blocked(blocks,next_node):
    '''
    checks if the point lies inside the blocks
    '''
    valid=False
    for k in range(blocks.shape[0]):
        if( next_node[0] > blocks[k,0] and next_node[0] < blocks[k,3] and\
              next_node[1] > blocks[k,1] and next_node[1] < blocks[k,4] and\
              next_node[2] > blocks[k,2] and next_node[2] < blocks[k,5] ):
            valid = True
            break
    return valid

def discretize_map(boundary,blocks,start,goal):
    '''
    This function discretizes the map with a resolutiomn of 0.2
    '''
    
    x_coord=np.arange(boundary[0,0],boundary[0,3],0.2)
    if(not(abs(x_coord[-1]-boundary[0,3])<1e-4)):
        x_coord=np.append(x_coord,x_coord[-1]+0.1)
    y_coord=np.arange(boundary[0,1],boundary[0,4],0.2)
    if(not(abs(y_coord[-1]-boundary[0,4])<1e-4)):
        y_coord=np.append(y_coord,y_coord[-1]+0.1)
    z_coord=np.arange(boundary[0,2],boundary[0,5],0.2)
    if(not(abs(z_coord[-1]-boundary[0,5])<1e-4)):
        z_coord=np.append(z_coord,z_coord[-1]+0.1)
    st=0
    en=0
   
    #construct value array
    g=np.zeros((x_coord.shape[0],y_coord.shape[0],z_coord.shape[0]))
    
    for x in range(x_coord.shape[0]):
        for y in range(y_coord.shape[0]):
            for z in range(z_coord.shape[0]):
                next_node=state_from_indices(x,y,z,x_coord,y_coord,z_coord)
                blocked=is_blocked(blocks,next_node)
                if(blocked==True):
                    g[x,y,z]=-10 #set the value of g at -10 if x,y,z lie inside the obstacle
                else:
                    point=np.array([x_coord[x],y_coord[y],z_coord[z]])
                    if(np.linalg.norm((point-start),2)<=0.2 and st!=1):
                        st=1
                        start_tuple=(x,y,z)
                        g[x,y,z]=0
                    else:
                        if(np.linalg.norm((point-goal),2)<=0.2 and en!=1):
                            en=1
                            goal_tuple=(x,y,z)
                            g[x,y,z]=np.inf
                        else:
                            g[x,y,z]=np.inf #set the rest coordinates value to infty
    return g,start_tuple,goal_tuple,x_coord,y_coord,z_coord

def collision_detection(line,aabb):
    '''
    check collision for each block
    '''
    u1=-np.inf
    u2=np.inf
    p=[line[0]-line[3],line[3]-line[0],line[1]-line[4],line[4]-line[1],line[2]-line[5],line[5]-line[2]]
    q=[line[0]-aabb[0],aabb[3]-line[0],line[1]-aabb[1],aabb[4]-line[1],line[2]-aabb[2],aabb[5]-line[2]]
    for i in range(6):
        if(p[i]==0):
            if(q[i]<0):
                return False
        else:
            t=q[i]/p[i]
            if(p[i]<0 and u1<t):
                u1=t
            elif(p[i]>0 and u2>t):
                u2=t
    
    if(u1>u2 or u1>1 or u1<0):
        return False
    
    return True        

def check_collision(path,blocks):
    '''
    check collision for all the blocks
    '''
    is_true=False
    for i in range(path.shape[0]-1):
        for b in blocks:
            is_true=collision_detection(path[i:i+2,:].reshape((-1,1)),b)
            if(is_true==True):
                return True
    return False

def get_children(node,x_coord,y_coord,z_coord,g,blocks):
    '''
    gets the  children fo the current node in A*
    '''
    children=[]
    for x in [-1,0,1]:
        if((node[0]+x)>=x_coord[0] and (node[0]+x)<x_coord.shape[0]): # to check if the new node is inside the bounds
            for y in [-1,0,1]:
                if((node[1]+y)>=y_coord[0] and (node[1]+y)<y_coord.shape[0]):
                    for z in [-1,0,1]:
                        if((node[2]+z)>=z_coord[0] and (node[2]+z)<z_coord.shape[0]):
                            if(x==0 and y==0 and z==0):
                                continue
                            else:
                                #check if the nod elies inside an obstacle
                                if(g[node[0]+x,node[1]+y,node[2]+z]<0):
                                    continue
                                else:
                                    #construct a line to detect collision
                                    line=np.array([node[0],node[1],node[2],node[0]+x,node[1]+y,node[2]+z])
                                    detect=False
                                    for b in blocks:
                                        #check if an edge is feasible between these nodes
                                        if(collision_detection(line,b)==True):
                                            detect=True
                                            break
                                            #append the node and cij if an edge is possible to contruct
                                    if(detect!=True):
                                        cij=np.linalg.norm((x_coord[node[0]]-x_coord[node[0]+x]\
                                                            ,y_coord[node[1]]-y_coord[node[1]+y],\
                                                            z_coord[node[2]]-z_coord[node[2]+z]))
                                        children.append(((node[0]+x,node[1]+y,node[2]+z),cij))
    return children
   
def weighted_a_star(g,start_tuple,goal_tuple,x_coord,y_coord,z_coord,eps,blocks):
    '''
    finds the path whcih is e-suboptimal
    algorithms is given the project report
    '''
    openlist=pqdict()
    closedlist=[]
    fs=eps*np.linalg.norm((x_coord[start_tuple[0]]-x_coord[goal_tuple[0]],\
                           y_coord[start_tuple[1]]-y_coord[goal_tuple[1]],\
                           z_coord[start_tuple[2]]-z_coord[goal_tuple[2]]))
    openlist.update({start_tuple:fs})
    node_index=(x_coord.shape[0]+1,y_coord.shape[0]+1,z_coord.shape[0]+1)
    Parent={}
    #Weighted A-star algorithm, pseudocode given in the report
    while(goal_tuple not in closedlist):
        current_tuple=openlist.pop()
        children=get_children(current_tuple,x_coord,y_coord,z_coord,g,blocks)
        closedlist.append(current_tuple)
        for (child,cij) in children:
            if(child not in closedlist):
                temp_val=cij+g[current_tuple[0],current_tuple[1],current_tuple[2]]
                if(g[child[0],child[1],child[2]]>temp_val):
                    g[child[0],child[1],child[2]]=temp_val
                    Parent.update({child:current_tuple})
                    f=g[child[0],child[1],child[2]]+eps*np.linalg.norm((x_coord[child[0]]-x_coord[goal_tuple[0]],\
                                                                       y_coord[child[1]]-y_coord[goal_tuple[1]],\
                                                                       z_coord[child[2]]-z_coord[goal_tuple[2]]))
                    if(child in list(openlist.keys())):
                        openlist[child]=f
                    else:
                        openlist.update({child:f})
    return Parent,closedlist,g

def get_path(start_tuple,goal_tuple,Parent,x_coord,y_coord,z_coord):
    '''
    returns the np array path from path indices
    '''
    path=[]
    val=goal_tuple
    arr=np.array([x_coord[val[0]],y_coord[val[1]],z_coord[val[2]]])
    path.append(arr)
    while (val!=start_tuple):
            val_new=Parent[val]
            val=val_new
            arr=np.array([x_coord[val[0]],y_coord[val[1]],z_coord[val[2]]])
            path.append(arr)
    return np.array(path[::-1])

def pathlength(filename):
    '''
    Gets path length
    '''
    arr=np.load(filename+'.npy')
    length=0
    for i in range(arr.shape[0]-1):
        length+=np.linalg.norm((arr[i+1]-arr[i]))
    return length

def runtest_a_star(start,goal,filename,eps):
    '''
    returns the path and the path length for a star
    '''
    boundary, blocks = load_map('./maps/'+filename+'.txt')
    #inflating the obstacles
    blocks[:,0]=blocks[:,0]-0.2
    blocks[:,1]=blocks[:,1]-0.2
    blocks[:,2]=blocks[:,2]-0.2
    blocks[:,3]=blocks[:,3]+0.2
    blocks[:,4]=blocks[:,4]+0.2
    blocks[:,5]=blocks[:,5]+0.2
    t0=tic()
    #discretize the map
    g,start_tuple,goal_tuple,x_coord,y_coord,z_coord=discretize_map(boundary,blocks,start,goal)
    toc(t0,'discretization')
    t0=tic()
    # astar to get the path
    Parent,closedlist,g=weighted_a_star(g,start_tuple,goal_tuple,x_coord,y_coord,z_coord,eps,blocks)
    toc(t0,'planning')
    # array path from the indices
    path=get_path(start_tuple,goal_tuple,Parent,x_coord,y_coord,z_coord) 
    boundary, blocks = load_map('./maps/'+filename+'.txt')
    collision=check_collision(path,blocks) #check if the path does not collide with objects
    success=False
    length=0
    if(collision==False):
        success=True
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
        ax.plot(path[:,0],path[:,1],path[:,2],'r-')
        np.save(filename+'.npy',path)
        length=pathlength(filename)
    return success,length

def get_obstacle(blocks):
    '''
    returns obstacle in the required format
    '''
    obs=[]
    for b in blocks:
        obs.append((b[0],b[1],b[2],b[3],b[4],b[5]))
    return np.array(obs)

def pathlength_rrt(filename):
    '''
    Gets path length
    '''
    arr=np.load('rrt_'+filename+'.npy')
    length=0
    for i in range(arr.shape[0]-1):
        length+=np.linalg.norm((arr[i+1]-arr[i]))
    return length

def rrt_runtest(start,goal,filename,r,prc):
    collision=True
    boundary, blocks = load_map('./maps/'+filename+'.txt')
    X_dimensions=np.array([(boundary[0,0],boundary[0,3]),\
                            (boundary[0,1],boundary[0,4]),(boundary[0,2],boundary[0,5])])
    x_start=(start[0],start[1],start[2])
    x_goal=(goal[0],goal[1],goal[2])
   
    obstacles=get_obstacle(blocks) #get obstacles
    X = SearchSpace(X_dimensions,obstacles) #define state space
    Q=[(0.1,6),(0.1*2**0.5,18),(0.1*3**0.5,2)] # edge lengths
    max_samples=2000000 # max samples
    rewire_count = 26  # optional, number of nearby branches to rewire
     # probability of checking for a connection to goal
    t0=tic()
    rrt = RRTStar(X, Q, x_start,x_goal, max_samples, r, prc, rewire_count) #get path
    path= rrt.rrt_star()
    toc(t0,'planning')
    arr_path=np.array(path)
    collision=check_collision(arr_path,blocks)
    length=0
    success=False
    if(collision==False):
        print('yes')
        np.save('rrt_'+filename+'.npy',arr_path)
        length=pathlength_rrt(filename)
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
        ax.plot(arr_path[:,0],arr_path[:,1],arr_path[:,2],'r-')
        success=True
    return success,length

def pathlength_bi_rrt(filename):
    '''
    Gets path length
    '''
    arr=np.load('bi_rrt_'+filename+'.npy')
    length=0
    for i in range(arr.shape[0]-1):
        length+=np.linalg.norm((arr[i+1]-arr[i]))
    return length

def bi_rrt_runtest(start,goal,filename,r,prc):
    '''
    returns pth from bi directional rrt
    r -> discretization of the connection to check if the
    '''
    collision=True
    boundary, blocks = load_map('./maps/'+filename+'.txt')
    X_dimensions=np.array([(boundary[0,0],boundary[0,3]),\
                            (boundary[0,1],boundary[0,4]),(boundary[0,2],boundary[0,5])])
    x_start=(start[0],start[1],start[2])
    x_goal=(goal[0],goal[1],goal[2])
   
    obstacles=get_obstacle(blocks)
    X = SearchSpace(X_dimensions,obstacles)
    Q=[(0.2,6),(0.2*2**0.5,18),(0.2*3**0.5,2)]
    #Q = np.array([(8, 4)]) 
    max_samples=2000000
    rewire_count = 26  # optional, number of nearby branches to rewire
     # probability of checking for a connection to goal
    t0=tic()
    rrt = RRTStarBidirectionalHeuristic(X, Q, x_start,x_goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star_bid_h()
    toc(t0,'planning')
    arr_path=np.array(path)
    collision=check_collision(arr_path,blocks)
    success=False
    if(collision==False):
        print('yes')
        np.save('bi_rrt_'+filename+'.npy',arr_path)
        length=pathlength_rrt(filename)
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)
        ax.plot(arr_path[:,0],arr_path[:,1],arr_path[:,2],'r-')
        success=True
    return success,length

def test_single_cube(choice):
    '''
    The choice variable decides whcih algorithm to use
    1: weighted A-star
    2: RRT*
    3: Bi-RRT
    '''
    print('Running single cube test...\n') 
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 5.5])
    if(choice==1):
        success,pathlength=runtest_a_star(start,goal,'single_cube',3)
    elif(choice==2):
        success,pathlength=rrt_runtest(start,goal,'single_cube',0.005,0.01)
    elif(choice==3):
        success,pathlength=bi_rrt_runtest(start,goal,'single_cube',0.005,0.01)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

def test_monza(choice):
    '''
    The choice variable decides whcih algorithm to use
    1: weighted A-star
    2: RRT*
    3: Bi-RRT
    '''
    print('Running monza test...\n') 
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    if(choice==1):
        success,pathlength=runtest_a_star(start,goal,'monza',3,0.4)
    elif(choice==2):
        success,pathlength=rrt_runtest(start,goal,'monza',0.005,0.01)
    elif(choice==3):
        success,pathlength=bi_rrt_runtest(start,goal,'monza',0.005,0.01)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

def test_room(choice):
    '''
    The choice variable decides whcih algorithm to use
    1: weighted A-star
    2: RRT*
    3: Bi-RRT
    '''
    print('Running room test...\n') 
    start = np.array([1.0, 5.0, 1.5])
    goal = np.array([9.0, 7.0, 1.5])
    if(choice==1):
        success,pathlength=runtest_a_star(start,goal,'room',3)
    elif(choice==2):
        success,pathlength=rrt_runtest(start,goal,'room',0.005,0.01)
    elif(choice==3):
        success,pathlength=bi_rrt_runtest(start,goal,'room',0.005,0.01)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

def test_flappy_bird(choice):
    '''
    The choice variable decides whcih algorithm to use
    1: weighted A-star
    2: RRT*
    3: Bi-RRT
    '''
    print('Running flappy bird test...\n') 
    start = np.array([0.5, 2.5, 5.5])
    goal = np.array([19.0, 2.5, 5.5])
    if(choice==1):
        success,pathlength=runtest_a_star(start,goal,'flappy_bird',3)
    elif(choice==2):
        success,pathlength=rrt_runtest(start,goal,'flappy_bird',0.005,0.01)
    elif(choice==3):
        success,pathlength=bi_rrt_runtest(start,goal,'flappy_bird',0.005,0.01)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

def test_tower(choice):
    '''
    The choice variable decides whcih algorithm to use
    1: weighted A-star
    2: RRT*
    3: Bi-RRT
    '''
    print('Running tower test...\n') 
    start = np.array([2.5, 4.0, 0.5])
    goal = np.array([4.0, 2.5, 19.5])
    if(choice==1):
        success,pathlength=runtest_a_star(start,goal,'tower',3)
    elif(choice==2):
        success,pathlength=rrt_runtest(start,goal,'tower',0.005,0.01)
    elif(choice==3):
        success,pathlength=bi_rrt_runtest(start,goal,'tower',0.005,0.01)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

def test_window(choice):
    '''
    The choice variable decides whcih algorithm to use
    1: weighted A-star
    2: RRT*
    3: Bi-RRT
    '''
    print('Running window test...\n') 
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    if(choice==1):
        success,pathlength=runtest_a_star(start,goal,'window',3)
    elif(choice==2):
        success,pathlength=rrt_runtest(start,goal,'window',0.005,0.01)
    elif(choice==3):
        success,pathlength=bi_rrt_runtest(start,goal,'window',0.005,0.01)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')

def test_maze(choice):
    '''
    The choice variable decides whcih algorithm to use
    1: weighted A-star
    2: RRT*
    3: Bi-RRT
    '''
    print('Running maze test...\n') 
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    if(choice==1):
        success,pathlength=runtest_a_star(start,goal,'maze',3)
    elif(choice==2):
        success,pathlength=rrt_runtest(start,goal,'maze',0.005,0.01)
    elif(choice==3):
        success,pathlength=bi_rrt_runtest(start,goal,'maze',0.005,0.01)
    print('Success: %r'%success)
    print('Path length: %d'%pathlength)
    print('\n')