### Christian Tonnesen

#### ID: 260847409

### 2A

Implementing the PID controller was not too difficult, as the pseudocode is quite simple to follow. The `P-` control
was easy and acted as the baseline for the control system with the following code `return Kp * theta`, where `Kp` had a
set value of `1.5`. This `Kp` value produced the most stable version of the controller with minimal variance in `theta`.
Afterwards, it was fairly straightforward to implement the derivative with `return Kd*(theta - self.prev_error)`, with
a `Kd` value of `20`. Finally, the integral required a running total, which we could track through the `PID_controller`
object.

```
self.prev_integral += theta
return self.prev_integral*Ki
```

The `Ki` term was quite sensitive and had to be set low at `0.001`.

![2A.png](2A.png)

The above graph represents the full implementation of the PID controller with each aspect enabled. It can be seen that
the theta difference grows over time, as the amplitude of the absolute value increases as time does. In theory, we would
hope to see the oscillations actually become smaller as time goes on. The current performance with increasing
oscillations can be attributed to the fact that the values are not yet tuned.

I also had to fix a bug where the same PID controller object was being used between rounds, meaning
the `self.prev_integral` and `self.prev_error`
were being carried into new rounds, where they should have actually been zeroed out. This was corrected by adding
in a line that would initialize a new `PID_Controller` object after the `self.pendulum.reset_state()` call on line `210`
.

### 2B

##### P-control

To tune the `P-control`, I decided to use manual tuning as I already had the framework in place to test from
Assignment1. To simulate the oscillation, I ran a numpy loop for float values from `0.1` to `4.9`, using a step of `0.1`
and collected the average of the theta differences from the `theta_diff_list` at the end of each round. Then after 10
rounds at a chosen `Kp` value, I averaged out all the averages to get a "representative" difference for the current
float `Kp`. The idea for this
testing framework being that the best `Kp` value would provide the smallest oscillation, and thus the smallest average
theta difference. The results can be seen in the graph below:
![img_3.png](img_3.png)

As can be seen, the difference values are quite high before a `Kp` value of `1` before quickly dropping off into the
sub `0.05` range. Interestingly, we find our minimum theta value of `0.005826` with a `Kp` value of `1.6`. After `1.6`,
the theta differences results for `Kp` values above `1.6` were more or less within the realm of error, meaning we had
reached some sort of steady limit. Contrary to my initial idea, this limit does not seem to be due to the clipping of
the action value in the `InvertedPendulum.step()` function, as the value of the `action` was always between `-1` and `1`
without any clipping applied.

As expected, we see oscillation within the `theta vs time` plot to a small degree. The oscillation does not diminish
over time as the proportion is constant on the error, meaning a correction in one direction will not account for
changing error. Thus, we witness near consistent oscillation.

![img.png](2B1.png)

##### PD-control

For `PD-control`, I replicated the testing methods used `P-control`, but utilized a different step and interval to pull
from. I had initially tried a small step like `P-control`, but saw no consistent average values like the previous
version's `1.6` for `Kp`. Thus, I increased the range of values until the difference between theta error values was
comfortably smaller than 0.0001. This meant the final value (granted it could be increased for even more preciseness),
was a `Kd`
value of `49`. It should also be noted that with a `Kd` value of `48` eventually did tip the value of `action` over `1`.
Values of 20-100 were tested, which can be seen below:

![img_4.png](img_4.png)

Promisingly, we also see that on our `theta vs time` graph, we see our oscillation amplitude drop. This is because the
addition of the derivative allows us to "slow-down" our action as our error grows smaller and smaller in approaching a
theta difference of `0`

![img.png](2B2.png)

##### PI-control

Given my experience with the sensitivity of `Ki` during the initial setup, I knew that I needed to start my estimates
quite low. I started at 0.5, then proceeded down by a factor of 10 if there was large instability or the pendulum failed
to remain upright, as I knew the `Kp` value of `1.6` could keep it stable on its own. In these trials with a larger `Ki`
value, I witnessed significant windup, where once the pendulum because destabilized, it followed a death spiral where
the accumulating error resulted in increasingly forceful actions in an attempt to correct itself. An example can be seen
on the following graph, where the `Ki` value is `0.009`:

![img.png](2B3 Bad.png)

I found that I was able to decrease the `Ki` value and reduce the amplitude of the oscillations, but it never
actually removed the oscillations. I tested multiple values of `Ki` starting atr 0.1:

![img_1.png](img_1.png)

As the results show, any efforts to tune `Ki` and get the pendulum to react non-volatile way relegated to me giving `Ki`
an
infinitesimally small value. At low enough value of `Ki=0.0001`, however, the deviation was minimal enough as to not
topple the pendulum.

![img.png](2B3 Final.png)

As we can see from the `theta vs time` graph, we experience oscillations, however, they are minor enough to not disturb
the original `Kp` value. In fact, the graph is near identical to the one produced in the `P-control` tuning, as we had
shrunk the integral addition so as to be near-indistinguishable from actually making it zero.

I also attempted to find the `Ki` value through the Ziegler-Nichols method to verify my own manual results. For
reference, if one uses our `Kp` value of `1.6` as `Ku`, then for `PI-Control`, we should use a `Kp` value of `0.45Ku`
or `0.72`. For the `Tu`, the period of the oscillations came out to `273 units`, so the `Ki` value would
be `((0.54*Ku)/Tu)` or `0.003165`. However, using these values did not produce a finely-tuned result either:

![img.png](2B3 ZN.png)

It should be noted that for the `PD-Control`, though, the ZN method gave a `Kd` value of `43.68`, quite close to our
chosen value of `49` and still worthwhile if we were to adjust our `Kp` according to ZN rules.

After much searching and testing, I decided to implement an integral error limit to "soft reset" when the value of the
cumulative errors got too high. I was initially concerned that this would replicate the issue of negating any addition
by the integral term similar as to when we made the `Ki` variable set very low, but I was intrigued to see that over the
course of the `1000` timestep, the code only "reset" the integral about a 20th of the time, meaning it still made a
useful contribution.

```
        if 0.05 < self.prev_integral + theta or self.prev_integral + theta < -0.05:
            self.prev_integral = 0
            return self.prev_integral
        self.prev_integral += theta
        return self.prev_integral * ki
```

Subsequent tests with this rate-limiter used the original `Kp` value of `1.6` and a `Ki` value of `0.0001`, which
produced the following:

![img.png](2B3Limit.png)

Nonetheless, even with a rate-limiter, we were unable to produce any settling behavior with the `PI-Control`. This could
possibly be due to the fact that the integrand value will always be lagging, and is prone to wind-up, whereas the
derivative term is self-regulated. However, there does seem to be some effort by the controller to bring the model to a
steady state thanks to the presence of the `Kp` term, as evidenced by the oscillations.

I will also briefly mention I did try to use negative `Ki` values, but found they were unable to strongly decrease (only
slightly decrease) the amplitude of the oscillations over time at any value I tried.

`Ki = -0.1`

![img.png](2B31Neg.png)

`Ki = -0.001`

![img.png](2B301Neg.png)

`Ki = -0.0001`

![img.png](2B3001Neg.png)

##### DI-control

For `DI` control, I attempted to utilize the manual control methods again, this time with better results than
the `PI-Control`. I tried to preliminary `Kd` values, noticing that with the `Kp` term zeroed out, the model required a
higher value than last time to stabilize the pendulum. As such, I ran tests from `0` to `10000`:

![img_5.png](img_5.png)

From the testing, I found the ideal `Kd` value of `3000`, which, on its own, produced the following graph:

![img_5.png](2B4MicroKd.png)

It seems that there is a series of micro oscillations between `0` and `150`, which then either disappear or become so
small as to look linear on the graph. This is because the derivative function attempts to predict the future error, and
does not tend towards a steady state without the presence of a `Pk` or `Pi` term. I then attempted to find a `Ki`
pairing, starting small given the experience with
the `PI-Control`. I also kept the rate-limiter in place, as I felt it regulated the windup well. However, I found that
as I increased the `Ki` value, the model actually became more stable. Thus, I
ran some manual tuning tests:

![img_6.png](img_6.png)

Having discovered that a value of `29` provided the best theta difference, I chose it for our `Ki` parameter. Combining
the new values, the controller acted as follows:

![img_6.png](2B4DI.png)

In the model, we can see that there is a period of increasing oscillation, which then promptly drop off
around `900 units`, similar to a settling time with a very high period. We can therefore see the effect of pairing
the `Kd` with integral control, granting us a model that does seem to trend towards some steady state.

### 2C

For gravity and pole mass testing, I utilized the full `PID` control type, since that had proven the most stable of the
configurations so far without fully dampening the error. In the case of the gravity, I tested both increases (first
picture) and decreases (second picture) to observe their settling behaviour.

`Gravity at 12 m/s^2 (instead of 9.81)`

![img.png](2C12Gravity.png)

`Gravity at 4.905 m/s^2 (instead of 9.81)`

![img.png](2CHalfGravity.png)

In the case where we increased the gravity, we observe an increase in settling time, with a correlated increase in the
oscillation period from our standard `273 units` measured in the `PI-Control` testing to somewhere around `~400 units`.
Conversely, when we decreased the force of gravity, we saw a decrease in settling time and oscillation period down
to `~190` at its lowest. These both make sense, as an increase in gravitational force would pull the pendulum head
closer to the ground at the apex of its oscillations (increasing the period/theta difference/acceleration), while a
smaller
gravitational force would result in less of a downward pull (and smaller period/theta difference/acceleration). We also
see similar results when one increases or lowers the mass of the pendulum:

`Pole Mass at 0.2 (instead of 0.1)`

![img.png](2C2xPole.png)

`Pole Mass at 0.01 (instead of 0.1)`

![img.png](2CHalfPole.png)

The higher the mass of the pendulum, the higher the net torque due to gravity, similarly increasing the period of the
oscillations. Decreasing the mass of the pendulum results in a lower net torque, and thus decreases the period/settling
time.

### 2D

To implement the disturbance, I utilized a `numpy.random.randomint(0, 1000)` in order to obtain two timesteps. Whenever
either of these timesteps was reached, then the action returned by the controller would be some random number
between `-1` and `1` (`return np.random.uniform(-1, 1)`). In the below example, we can see the timesteps are `87`
and `174`, where the theta difference spikes from a random action. However, we can see that the system is able to reduce
the oscillations and settle again, granted at a much later time (`900` vs the original `700`). Thus, we can say that the
controller is competent in rejecting disturbances.

![img.png](2D Disturbance.png)

