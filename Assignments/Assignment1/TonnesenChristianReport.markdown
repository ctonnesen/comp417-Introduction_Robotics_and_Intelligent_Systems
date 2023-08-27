### Christian Tonnesen

#### ID: 260847409

## Simple Point Robot

### 2A

Implementing the RTT point bot code was relatively easy, granted there were some geometric problems and possible to
typos to deal with. `closestPointToPoint()`  was not difficult to code up, consisting of a for-loop which would cycle
through all the nodes in the config tree and select the one with the smallest distance (computed in another method) to
the `genPoint()` coordinates `P`. There was some issue with finding the new point’s location, as I needed to essentially
create a point on the line between the source and `P` that was the distance of the `smallstep` variable. To accomplish
this, I calculated the unit vectors from the config point and scaled the distance to get the corresponding coordinates
in `addNewPoint()` , resulting in the new point `V`. navigating obstacles, I reutilized the `inRect()` from
the `imageToRects` file as it was already doing what I needed.

However, the `lineHitsRect()` required a bit more finagling to get work. Contrary to my initial thoughts, finding
perpindicular line segments is quite challenging, as you either need to find orientation of 3-point segments, or
alternatively find intersection points on the full linear equations within the bounds of the rectangle. I opted for the
former, as the Python code lends itself more nicely to calculations than solving algebraic equations, granted I could
have also found the determinant. It is worth noting that the original function signatures of the obstacle methods were:
`lineHitsRect(vertices[v],p,o)` and `inRect(p,o,1)`, where `P` was the `genPoint()`. I corrected this so that the
obstacle methods were actually calculating if `V` was in the rectangle or if `V` and the closest point `CP` had an
intersecting line.

After handling the obstacles, I was able to add V to the list of vertices, add the connections between `CP` and `V`
check the distance between it and the target point `T`. When running the algorithm, there was an issue where the random
trees were not getting very close to the target location. To fix this, I followed the advice of the slides and added a
timer, which would choose `T` as `P` to help nudge the tree configurations closer. This meant that the tree would always
eventually find `T`.

### 2B

#### shot.png

|         | Uniform |          |       | Gaussian |          |       |
|---------| ------- | -------- | ----- | -------- | -------- | ----- |
| Seed    | Nodes   | Distance | Steps | Nodes    | Distance | Steps |
| 0       | 3941    | 1206     | 201   | 2961     | 1242     | 207   |
| 1       | 2261    | 1194     | 199   | 2681     | 1170     | 195   |
| 2       | 2361    | 1188     | 198   | 2668     | 1194     | 199   |
| 3       | 3841    | 1254     | 209   | 2881     | 1158     | 193   |
| 4       | 2641    | 1134     | 189   | 3601     | 1182     | 197   |
| 5       | 2341    | 1128     | 188   | 2561     | 1134     | 189   |
| 6       | 4261    | 1242     | 207   | 3061     | 1260     | 210   |
| 7       | 3381    | 1182     | 197   | 2981     | 1308     | 218   |
| 8       | 3181    | 1152     | 192   | 3892     | 1218     | 203   |
| 9       | 2641    | 1194     | 199   | 2561     | 1176     | 196   |
| Average | 3085    | 1187.4   | 197.9 | 2984.8   | 1204.2   | 200.7 |

&nbsp;
&nbsp;

#### simple.png

|      | Uniform |          |       | Gaussian |          |       |
| ---- | ------- | -------- | ----- | -------- | -------- | ----- |
| Seed | Nodes   | Distance | Steps | Nodes    | Distance | Steps |
| 0    | 761     | 1194     | 199   | 1081     | 1254     | 209   |
| 1    | 1095    | 1320     | 220   | 841      | 1230     | 205   |
| 2    | 1141    | 1182     | 197   | 981      | 1170     | 195   |
| 3    | 901     | 1224     | 204   | 921      | 1254     | 209   |
| 4    | 961     | 1242     | 207   | 781      | 1194     | 199   |
| 5    | 861     | 1260     | 210   | 741      | 1140     | 190   |
| 6    | 961     | 1104     | 184   | 901      | 1134     | 189   |
| 7    | 921     | 1128     | 188   | 1001     | 1146     | 191   |
| 8    | 961     | 1182     | 197   | 721      | 1170     | 195   |
| 9    | 1021    | 1128     | 188   | 1081     | 1284     | 214   |
|      | 958.4   | 1196.4   | 199.4 | 905      | 1197.6   | 199.6 |

The above are the results from trials using both the `shot.png` and `simple.png`. In both instances, you can see that
the averages for the uniform nodes (the amount of times the loop was run) is higher than gaussian ones, which makes
sense given that in the Gaussian
distribution, we are getting more random points `P` that are closer to the actual target

```
x = random.gauss(tx, sigmax_for_randgen)
y = random.gauss(ty, sigmay_for_randgen)
```

However, the distribution choice seemed to have a negligible effect on the distance of the path discovered or the amount
of steps (nodes to reach the target). This is also logical as the distance between the start and target points does not
change, so, unless one distribution takes a wildly different path than the other, they should both eventually follow a
similar path/distance. This is the same for the amount of nodes needed to get from the start to the target, as the steps
are more or less a function of the distance divided by the `smallstep`.

### 2C

![img_4.png](Report Images/img_4.png) ![img_3.png](Report Images/img_3.png)

&nbsp;

We see an explicit exponential decay for the iterations vs. step size, which we would expect to see given that for each
increase of the step-size, each configuration of the tree covers more of the free-space. This roughly translates into
higher chances for the RTT finding the target point on a given iteration i.e. fewer iterations required. We can see this
as the number of iterations goes from ~6000 to ~1000 over an increase of 9 units from a size of 3 to 12, but we only
see drop of ~100 difference from a size of 21 to 30. As for the path length, there is a general negative linear trend in
the beginning, but we should note that there is a change in scale from the first graph of 1000 per gridline to 10. For a
comparison, the third graph displays them with the same scale.
&nbsp;

![img_5.png](Report Images/img_5.png)

&nbsp;

We see now that there is much more drastic drop for the iterations than for the path-length. Similar to what was
outlined in 2B, the distance/path-length should not change much as a function of the step-size, as the distance between
the start and target remains constant. We wouldn't see outrageous distances until the step-size becomes larger than
the linear distance between the two points or larger than the start to the edges of the map, as that means the RTT must
then take unoptimal (but the only viable) paths to the target to not intersect the obstacles.

Based on these results, I would say that a `step-size` between 18 and the linear distance between the start and end
positions is a good choice for optimal run-time and path-finding.

## Simple Line Robot

### 3A

The implementation for this problem was very similar to 2A, with much of the challenge coming from the implementation of
a length, and as a result, the considerations of collisions when rotating it. The finding of new points remained the
same as the `simple-point-robot`, where you just needed to consider the step-size and direction of the generated point.
For each move to this new point, though, we must rotate the robot to be pointed in the correct direction for travel. For
instance, if the robot must go north and is currently west(tail)-to-east(head), we must rotate tail to south such that
the new orientation is south(tail)-to-north(head) so that the robot may travel north. In each rotation, we will
consider two points: `robot-front-point` (also known as `CP` from the Point Robot) and `robot-back-point`.

To find the angle between the `robot-front-point` and new point (`v`), we
utilize a function called `atan2(y,x)`, which treats our head as the origin and the new point as a point in a 2D plane,
giving us the radian measurement between the two. This angle measurement becomes our new `robot_angle`. For the rotation
of the tail, we will check for collisions by testing a line segment from the `robot-head-point` to each angle of the
rotation required by the tail, reusing the `lineHitsRect(p1, p2, r)` on each segment. If any cause an intersection, we
know we will hit an obstacle while rotating the robot.

We also must calculate the angle to the tail of the robot, since it will not always be exactly `robot_size` units to the
left/right of the head's `x-coordinate`. The coordinates of the tail can be found using the
same `addNewPoint(p1, p2, stepsize)` function, except with a desired direction and length (`robot_size`) towards the
point previous to `CP/robot-front-point`. To find the angle to the tail, we again use the `atan2()` function, instead
using
the coordinates of the tail as our "destination". This will yield `angle_to_robot_end`, or the angle from the head to
current position of the tail. To find how much we must rotate the tail, we require the coordinates of where
the tail will be once we start the movement towards `v`, such that the `robot-front-point` and `robot-end-point` lie on
the same line
as `v`. To do this, we simply use the same vector between the `cp` and the generated point `p` that we used to find `v`,
and instead reverse its direction and make the chosen length that of `robot_length`. This will give us the coordinates
for our `robot-back-point-destination`. We then find the angle between the `robot-head-point` and
the `robot-back-point-destination` using the `atan2()` function, which we call `angle_to_final`.

With `angle_to_robot_end` and `angle_to_final`, we
calculate the shortest distance (least amount of radians) between the two,
whether that be in the positive radian direction (counter-clockwise) or negative direction (clockwise). Once we have the
shortest path, we create an array holding all of the angles between `angle_to_robot_end` and `angle_to_final` on the
shortest path, with each angle being separated from the next by a predefined `step` (usually 10 degrees/0.1 radians). We
then execute our plan, calculating the temporary tail points using `math.cos()` and `math.sin()` and checking if the
line segment between that and the `robot-head-point` intersects some obstacle. If not, then the move from `cp` to `v` is
valid. At the end of each movement, we also store the `angle_to_final` in a global variable, similar to
the `robot_angle`.

When we are within `step-size` distance of the target coordinates, we replicate this process again, but we also check
for a final rotation from the `angle_to_final` (currently the angle required to reach `[tx,ty]`) to `math.pi`, since
that would mean that the
robot is a flat horizontal line, with a `robot_angle` of 0, as requested by the instructions.

### 3B

![img_6.png](Report Images/img_6.png)

We see an explicit exponential growth in the number of iterations as we increase the robot's size. This would make
sense, given that as the robot's size increases, it is no longer capable of making tight maneuvers that a smaller
version would, necessitating more generated points in search of finding a valid one. The starting position and obstacle
course of the `shot.png` highlighted this quite well, as there were common areas where the algorithm would get stuck. We
can see an example of both in the following:

![img_8.png](Report Images/img_8.png)

The starting position of the robot is `(100, 630, 0)`, which places it right next to the obstacle in the bottom left.
Given that the robot cannot hit the obstacle when rotating, this severely limits the movement of the robot. If we were
to get the position `(100, 700)`, or really any position with a high `y` value above the start, we immediately become
parallel to the wall of the obstacle, as that is the only way up from the current position. Once we become parallel to
the obstacle, though, as one can see at the start of the configuration, the only possible direction is further up. Any
movement in the positive `x` direction is not allowed as that would intersect with the
obstacle to the right of the robot. Conversely, any movement in the negative `x` direction would require the tail of the
robot to swing into the obstacle wall, which is not allowed. This can be demonstrated even if the robot is not directly
next to a wall that it is parallel with. For instance, if the configuration has a point that is closer to the target
than another `cp`, then on the "biased" iteration, the algorithm will choose that point and attempt to rotate. This will
be problematic however, if there is an obstacle within the distance of the size of the robot, as it will then collide on
rotation. This does not stop `cp` from getting chosen during the next "biased" iteration, though, meaning we will run
into this issue again. Thus, we are truly at the mercy of the randomness of the uniform sampling to hopefully find a
point/help generate a new `v` that is closer and more free to rotate than `cp` In all cases, the line robot is slower
than the point robot, as it cannot easily escape the walls of an obstacle like the point robot.

The bias of the `genPoint()` being the target every 1 in 20 times also proves to be
an issue, as once the robot arrives at the top left corner of the obstacle, it is able to move into the alcove to the
right, highlighted in green. Any movement to the left would still be impossible, at it would require a rotation of the
tail into the obstacle. It is worth noting the robot could move left if the tail did a full rotation from 270 degrees
back over the negative
and positive `x` axis to 0 degrees, but the algorithm for distance between angles only considers the shortest for
efficiency. However, this will prove to be the detriment of the robot, as it moves further into the
space, it is also limiting how much it can pivot/move in the opposite direction. As we see in the picture, once the
robot has entered, it is very limited in how it can move, as most generated points (when uniformly distributed) will be
somewhere to the right of the robot (the robot is only about 18% of the way to the right of the map, making 82% of
possible points create an invalid direction for the configuration to continue). Ironically, this is the problem faced by
potential field algorithms with local minima which RTT are supposed to fix. Nonetheless, the RTT will eventually leave
the alcove, but not after a large amount of iterations. The bias often has the effect of pulling the configuration
towards the obstacles, as they are between target and configuration at all points of the travel, except for when the
tree reaches the top right of the map and has a clear path.
We see parallel wall travel behaviour later as well, with the top obstacle and oblong one in the top right:

![img_11.png](Report Images/img_11.png)

This is another example where the bias towards the target brings the configuration close to the wall in its travel to
the target, resulting in the parallel wall travel. When the path between the configuration and the target is not
impeded, though, we see a relatively straight travel resulting from the bias. I suggest that the increasing iterations
can be mediated by changing the starting location to slightly higher up to avoid the alcove issue, or alternatively
moving the target to a more direct location. This is observable in the graph below, where the difference between
iterations of subsequent robot lengths is greater for the (100,630) start location than the adjusted (and higher up) (
10, 270) location. While we would expect some difference between the number of iterations for each starting location (
since the starting location
of (10, 270) is closer to the target), a difference in the trendlines' slopes/growth show that the larger iterations in
the (100,630) start location trials are not a result of just the distance between the target and start being longer.

![img_13.png](Report Images/img_13.png)