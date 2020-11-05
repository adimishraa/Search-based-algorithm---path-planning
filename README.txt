The program to run for results is final.py

src folder contains the library for rrt algorithm

to run final.py,
use the functions:
run_single_maze(choice)
run_monza(choice)
etc.
The choice variable decides which algorithm to choose

choice=1 : weighted A-star
choice=2 : RRT*
choice=3: Bi-directional RRT

If the desired results are not met, then
for rrt_runtest function, decrease the value of the variable 'r'.
There are three functions used to runtest:
1) runtest_a_star(start,goal,map_name,eps)
start: staring coordinate of the agent
goal: destination vector
map_name: ['single_cube','monza','flappy_bird','tower','maze','window','room']
eps: epsilon weighting

2) runtest_rrt(start,goal,map_name,r,prc)
start: staring coordinate of the agent
goal: destination vector
map_name: ['single_cube','monza','flappy_bird','tower','maze','window','room']
r: discretization resolution of the edge
prc: probability to check a direct connection to the goal

3)bi_rrt_runtest(start,goal,'maze',0.005,0.01)
start: staring coordinate of the agent
goal: destination vector
map_name: ['single_cube','monza','flappy_bird','tower','maze','window','room']
r: discretization resolution of the edge
prc: probability to check a direct connection to the goal



