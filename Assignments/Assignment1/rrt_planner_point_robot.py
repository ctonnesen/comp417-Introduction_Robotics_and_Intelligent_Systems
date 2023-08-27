'''
Hello and welcome to the first assignment :)
Please try to enjoy implementing algorithms you learned from the course; in this regard, we also have tried
to provide a good starting point for you to develop your own code. In this script, you may find different functions
that should be defined by you to perform specific tasks. A good suggestion would be to begin with the primary
function for the RRT algo named as "rrt_search". Following this function line by line you may realize how to define
each function to make it work!
Note, you may use the following commands to run this script with the required arguments:
python3 rrt_planner_point_robot.py --arg1_name your_input1 --arg2_name your_input2 e.g.
python3 rrt_planner_point_robot.py --world="shot.png"
To see the list of arguments, take a look at utils.py
Also note:
G is a list containing two groups: the first group includes the nodes IDs (0,1,2,...), while the second holds all pairs of nodes creating an edge
Vertices is a list of nodes locations in the map; it corresponds to the nodes' IDs mentioned earlier
GOOD LUCK!
'''
import math
import random
import drawSample
import sys
import imageToRects
import utils
import os


def redraw(canvas):
    canvas.clear()
    canvas.markit(tx, ty, r=SMALLSTEP)
    drawGraph(G, canvas)
    for o in obstacles: canvas.showRect(o, outline='blue', fill='blue')
    canvas.delete("debug")


def drawGraph(G, canvas):
    global vertices, nodes, edges
    if not visualize: return
    for i in G[edges]:
        # e.g. vertices: [[10, 270], [10, 280]]
        canvas.polyline([vertices[i[0]], vertices[i[1]]])


# Use this function to generate points randomly for the RRT algo
def genPoint():
    # if args.rrt_sampling_policy == "uniform":
    #     # Uniform distribution
    #     x = random.random()*XMAX
    #     y = random.random()*YMAX
    # elif args.rrt_sampling_policy == "gaussian":
    #     # Gaussian with mean at the goal
    #     x = random.gauss(tx, sigmax_for_randgen)
    #     y = random.gauss(ty, sigmay_for_randgen)
    # else:
    #     print ("Not yet implemented")
    #     quit(1)

    bad = 1
    while bad:
        bad = 0
        if args.rrt_sampling_policy == "uniform":
            # Uniform distribution
            x = random.random() * XMAX
            y = random.random() * YMAX
        elif args.rrt_sampling_policy == "gaussian":
            # Gaussian with mean at the goal
            x = random.gauss(tx, sigmax_for_randgen)
            y = random.gauss(ty, sigmay_for_randgen)
        else:
            print("Not yet implemented")
            quit(1)
        # range check for gaussian
        if x < 0: bad = 1
        if y < 0: bad = 1
        if x > XMAX: bad = 1
        if y > YMAX: bad = 1
    return [x, y]


def returnParent(k, canvas):
    """ Return parent note for input node k. """
    for e in G[edges]:
        if e[1] == k:
            canvas.polyline([vertices[e[0]], vertices[e[1]]], style=3)
            return e[0]


def genvertex():
    vertices.append(genPoint())
    return len(vertices) - 1


def pointToVertex(p):
    vertices.append(p)
    return len(vertices) - 1


def pickvertex():
    return random.choice(range(len(vertices)))


def pointPointDistance(p1, p2):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return dist


def closestPointToPoint(G, p2):
    distance = -1
    chosen_node = -1
    for node in G[0]:
        temp_distance = pointPointDistance(vertices[node], p2)
        if distance == -1 or temp_distance < distance:
            distance = temp_distance
            chosen_node = node
    return vertices[chosen_node], chosen_node


def lineHitsRect(p1, p2, r):
    segments = []
    segments.append(((r[0], r[1]), (r[0], r[3])))
    segments.append(((r[0], r[3]), (r[2], r[3])))
    segments.append(((r[2], r[3]), (r[2], r[1])))
    segments.append(((r[2], r[1]), (r[0], r[1])))
    for segment in segments:
        will_intersect = intersection(p1, p2, segment[0], segment[1])
        if will_intersect:
            return True
    return False


def intersection(p1, p2, r1, r2):
    orien1 = orientation(p1[0], p1[1], p2[0], p2[1], r1[0], r1[1])
    orien2 = orientation(p1[0], p1[1], p2[0], p2[1], r2[0], r2[1])
    orien3 = orientation(r1[0], r1[1], r2[0], r2[1], p1[0], p1[1])
    orien4 = orientation(r1[0], r1[1], r2[0], r2[1], p2[0], p2[1])
    if orien1 != orien2 and orien3 != orien4:
        return True
    return False


def orientation(p1x, p1y, p2x, p2y, obsx, obsy):
    val = (float(p2y - p1y) * (obsx - p2x)) - (float(p2x - p1x) * (obsy - p2y))
    if (val > 0):
        # Clockwise orientation
        return 1
    elif (val < 0):
        # Counterclockwise orientation
        return 2
    return -1


def inRect(p, rect, dilation):
    """ Return 1 in p is inside rect, dilated by dilation (for edge cases). """
    if imageToRects.inRect(p, rect, dilation):
        return 1
    return 0


def addNewPoint(p1, p2, stepsize):
    vector = (p2[0] - p1[0], p2[1] - p1[1])
    vector_length = math.hypot(vector[0], vector[1])
    normalized_vector = (vector[0] / vector_length, vector[1] / vector_length)
    new_point = (p1[0] + stepsize * normalized_vector[0], p1[1] + stepsize * normalized_vector[1])
    return new_point


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def rrt_search(G, tx, ty, canvas):
    # Please carefully read the comments to get clues on where to start
    # TODO
    # Fill this function as needed to work ...
    global sigmax_for_randgen, sigmay_for_randgen
    n = 0
    counter = 0
    while 1:  # Main loop
        # This generates a point in form of [x,y] from either the normal dist or the Gaussian dist
        p = genPoint()
        if counter % 20 == 0:
            p = (tx, ty)
        counter += 1
        # This function must be defined by you to find the closest point in the existing graph to the guiding point
        cp, cn = closestPointToPoint(G, p)
        v = addNewPoint(cp, p, SMALLSTEP)
        if visualize:
            # if nsteps%500 == 0: redraw()  # erase generated points now and then or it gets too cluttered
            n = n + 1
            if n > 10:
                canvas.events()
                n = 0
        bad_point = False
        for o in obstacles:
            # The following function defined by you must handle the occlusion cases
            if lineHitsRect(v, cp, o) or inRect(v, o, 1):
                bad_point = True
                break
                # ... reject
        if bad_point:
            continue
        k = pointToVertex(v)  # is the new vertex ID
        G[nodes].append(k)
        G[edges].append((cn, k))
        if visualize:
            canvas.polyline([vertices[cn], vertices[k]])

        if pointPointDistance(v, [tx, ty]) < SMALLSTEP:
            if not mute:
                print("Target achieved.", counter, "nodes in entire tree")
            if visualize:
                t = pointToVertex([tx, ty])  # is the new vertex ID
                G[edges].append((k, t))
                if visualize:
                    canvas.polyline([p, vertices[t]], 1)
                # while 1:
                #     # backtrace and show the solution ...
                #     canvas.events()
                nsteps = 0
                totaldist = 0
                while 1:
                    oldp = vertices[k]  # remember point to compute distance
                    k = returnParent(k, canvas)  # follow links back to root.
                    canvas.events()
                    if k <= 1: break  # have we arrived?
                    nsteps = nsteps + 1  # count steps
                    totaldist = totaldist + pointPointDistance(vertices[k], oldp)  # sum lengths
                if mute:
                    print(f"{counter} {totaldist} {nsteps}")
                    quit()
                print("Path length", totaldist, "using", nsteps, "nodes.")
                global prompt_before_next
                if prompt_before_next:
                    canvas.events()
                    print("More [c,q,g,Y]>")
                    d = sys.stdin.readline().strip().lstrip()
                    print("[" + d + "]")
                    if d == "c": canvas.delete()
                    if d == "q": return 0
                    if d == "g": prompt_before_next = 0
                break


def main():
    # seed
    random.seed(args.seed)
    if visualize:
        canvas = drawSample.SelectRect(xmin=0, ymin=0, xmax=XMAX, ymax=YMAX, nrects=0,
                                       keepcontrol=0)  # , rescale=800/1800.)
        for o in obstacles: canvas.showRect(o, outline='red', fill='blue')
    while 1:
        # graph G
        redraw(canvas)
        G[edges].append((0, 1))
        G[nodes].append(1)
        if visualize: canvas.markit(tx, ty, r=SMALLSTEP)
        drawGraph(G, canvas)
        rrt_search(G, tx, ty, canvas)
    # if visualize:
    #     canvas.mainloop()


if __name__ == '__main__':
    args = utils.get_args()
    visualize = utils.get_args()
    drawInterval = 10  # 10 is good for normal real-time drawing
    mute = args.mute_building
    prompt_before_next = 1  # ask before re-running sonce solved
    SMALLSTEP = args.step_size  # what our "local planner" can handle.
    if mute:
        blockPrint()
    map_size, obstacles = imageToRects.imageToRects(args.world)
    enablePrint()
    # Note the obstacles are the two corner points of a rectangle (left-top, right-bottom)
    # Each obstacle is (x1,y1), (x2,y2), making for 4 points

    XMAX = map_size[0]
    YMAX = map_size[1]
    # The boundaries of the world are (0,0) and (XMAX,YMAX)

    G = [[0], []]  # nodes, edges
    vertices = [[args.start_pos_x, args.start_pos_y], [args.start_pos_x, args.start_pos_y + 10]]

    # goal/target
    tx = args.target_pos_x
    ty = args.target_pos_y

    # start
    sigmax_for_randgen = XMAX / 2.0
    sigmay_for_randgen = YMAX / 2.0
    nodes = 0
    edges = 1

    main()
