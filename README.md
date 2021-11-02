# Perspective 3D engine
A 3D engine with quaternion rotation, simulating gravity and classical billiards physics, with collisions between discrete frame intervals.

## Controls
You can play with WASD motion and mouse panning like Minecraft, but the keys affect velocity (which doesn't decay in space mode) and if you look around in a clockwise circle you will end up rotated anticlockwise in the roll axis. You can also use R and F to move up and down, Q and E for roll and the arrow keys for yaw and pitch.

## Raytracing
Press the t key at any time to stop the program and render the current scene with raytracing (in 960\*720, it takes a while but goes row-by-row in real time). Hold the p key on startup for the real-time raytracing mode (with larger pixels but the same number of reflections). The pixels are large enough that it uses a hexagonal lattice instead of a square one, allowing an 8% reduction in rays cast for the same human-perceptible "resolution" (in terms of average distance of a random point on the screen from the closest pixel). If this goes wrong, hold the k key also for the square lattice mode. 

Rays use the same functions to detect collisions with balls as balls do with each other, but without the one-frame time constraint.

Rays reflect up to 8 times by default (additional ones don't seem to have visible effect at current opacities). Balls' reflectivenesses are controlled by variables set for each on their creation, but the edges of lower ones are still reflective regardless, reflectiveness only controls the how much a ray collision's absolute change in velocity shifts the ray's colour towards that of the ball.

The speed of light is infinite by default (when the 'light speed input to the raytracing function is 0) so the balls' positions don't change. Adjusting the speed of light makes rays' velocity changes from collisions account for the ball's velocity also.

Enabling 'Minkowski mode' at finite light speed makes them calculate interceptions with moving spheres instead of stationary ones (though they only proceed them along their current trajectories, not accounting for collisions between balls or gravity acting upon them).

None of these take additional computing power, and they can be set on the real-time mode's raytracing function call to experience for yourself (at the bottom of the green flag loop).

The gravity input, however, does take more computing power, and simulates rays' acceleration due to gravity at each step in their trajectory. The gravity input itself (when not set to 0) describes the granularity of this sampling (setting it higher makes it take longer steps and experience more gravity during each), and it makes the reflections input describe the number of steps.  

## Play in browser
The reasonably up-to-date version (where you control a ball in space with n-body gravity) is [here](https://turbowarp.org/574159176/fullscreen), the first version (with only the rendering, no physics) is [here](https://turbowarp.org/575040192/fullscreen).

I don't update the web version with each commit, however. For the latest one, [install TurboWarp desktop](https://turbowarp.org/desktop) and download and run the .sb3 file from this repository.
