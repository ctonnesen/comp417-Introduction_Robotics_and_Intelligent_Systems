import time
import random
import drawSample
import math
import sys
import imageToRects
import utils
import os


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


# display = drawSample.SelectRect(imfile=im2Small,keepcontrol=0,quitLabel="")
args = utils.get_args()
visualize = utils.get_args()
drawInterval = 100  # 10 is good for normal real-time drawing
mute = args.mute_building

prompt_before_next = 1  # ask before re-running sonce solved
SMALLSTEP = args.step_size  # what our "local planner" can handle.
if mute:
    blockPrint()
map_size, obstacles = imageToRects.imageToRects(args.world)
enablePrint()
robot_size = args.robot_length  # args.robot_length
robot_angle = args.start_pos_theta
robot_angle_back = math.pi
# Note the obstacles are the two corner points of a rectangle
# Each obstacle is (x1,y1), (x2,y2), making for 4 points
XMAX = map_size[0]
YMAX = map_size[1]

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
        canvas.polyline([vertices[i[0]], vertices[i[1]]])


def genPoint():
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


def lineFromPoints(p1, p2):
    # TODO
    return None


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


def angles_rotation_locator(v, robot_front_point, previous):
    angle = math.atan2(v[1] - robot_front_point[1], v[0] - robot_front_point[0])
    if previous is None:
        robot_end_point = (robot_front_point[0] - robot_size, robot_front_point[1])
    else:
        robot_end_point = addNewPoint(robot_front_point, previous, robot_size)
    angle_to_robot_end = math.atan2(robot_end_point[1] - robot_front_point[1],
                                    robot_end_point[0] - robot_front_point[0])
    reverse_destination = addNewPoint(robot_front_point, v, -robot_size)
    angle_to_final = math.atan2(reverse_destination[1] - robot_front_point[1],
                                reverse_destination[0] - robot_front_point[0])
    angles_to_hit = angle_finders(angle_to_final, angle_to_robot_end)
    if v != [tx, ty]:
        global robot_angle
        robot_angle = angle
        global robot_angle_back
        robot_angle_back = angle_to_final
    return angles_to_hit


def angle_finders(angle_to_final, angle_to_robot_end):
    diff_over_pos = abs(angle_to_final) + abs(angle_to_robot_end)
    diff_over_neg = abs(2 * math.pi - abs(angle_to_final) - abs(angle_to_robot_end)) % math.pi
    angles_to_hit = []
    step = 0.1
    if int(math.copysign(1, angle_to_robot_end)) != int(math.copysign(1, angle_to_final)):
        swapper = 0
        limit = -1000
        if min(diff_over_neg, diff_over_pos) == diff_over_neg:
            swapper = int(math.copysign(1, angle_to_robot_end))
            limit = 314
        elif diff_over_neg == diff_over_neg:
            swapper = int(math.copysign(1, angle_to_final))
            limit = 0
        angles_to_hit = [x * 0.01 for x in
                         range(int(angle_to_robot_end * 100), (swapper * limit), int(swapper * step * 100))]
        angles_to_hit.extend(
            [x * 0.01 for x in range(swapper * -limit, int(angle_to_final * 100), int(swapper * step * 100))])
    else:
        sign = math.copysign(1, angle_to_final - angle_to_robot_end)
        angles_to_hit = [x * 0.01 for x in
                         range(int(angle_to_robot_end * 100), int(angle_to_final * 100), int(sign * step * 100))]
    return angles_to_hit


def rotation_check(robot_front_point, angles_to_hit, obstacle):
    for angle in angles_to_hit:
        x = robot_front_point[0] + robot_size * math.cos(angle)
        y = robot_front_point[1] + robot_size * math.sin(angle)
        check = lineHitsRect((x, y), robot_front_point, obstacle)
        if check:
            return True
    return False


def returnParentPoint(k):
    """ Return parent note for input node k. """
    for e in G[edges]:
        if e[1] == k:
            return vertices[e[0]]
    return None


def rrt_search(G, tx, ty, canvas):
    # TODO
    # Fill this function as needed to work ...

    global sigmax_for_randgen, sigmay_for_randgen
    n = 0
    nsteps = 0
    counter = 0
    while 1:
        p = genPoint()
        if counter % 20 == 0:
            p = (tx, ty)
        counter += 1
        cp, cn = closestPointToPoint(G, p)
        v = addNewPoint(cp, p, SMALLSTEP)
        if visualize:
            # if nsteps%500 == 0: redraw()  # erase generated points now and then or it gets too cluttered
            n = n + 1
            if n > 10:
                canvas.events()
                n = 0
        angles = angles_rotation_locator(v, cp, returnParentPoint(cn))
        bad_point = False
        for o in obstacles:
            # The following function defined by you must handle the occlusion cases
            if lineHitsRect(v, cp, o) or inRect(v, o, 1) or rotation_check(cp, angles, o):
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
            bad_point = False
            final_angles = angles_rotation_locator([tx, ty], v, cp)
            last_angle = None
            if len(final_angles) != 0:
                last_angle = final_angles[len(final_angles) - 1]
            else:
                last_angle = robot_angle_back
            to_zero = angle_finders(math.pi, last_angle)
            for o in obstacles:
                # The following function defined by you must handle the occlusion cases
                if lineHitsRect([tx, ty], v, o) or rotation_check(v, final_angles, o) or rotation_check([tx, ty],
                                                                                                        to_zero, o):
                    bad_point = True
                    break
                    # ... reject
            if bad_point:
                continue
            robot_angle = 0
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
                    if d == "q": return
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
        redraw(canvas)
        G[edges].append((0, 1))
        G[nodes].append(1)
        if visualize: canvas.markit(tx, ty, r=SMALLSTEP)

        drawGraph(G, canvas)
        rrt_search(G, tx, ty, canvas)
    if visualize:
        canvas.mainloop()


if __name__ == '__main__':
    main()
