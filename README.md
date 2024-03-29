# Perspective 3D engine
A 3D engine with quaternion rotation, simulating gravity and classical billiards physics, with collisions between discrete frame intervals, and raytracing with rays going backwards in time.

## Controls
You can play with WASD motion and mouse panning like Minecraft, but the keys affect velocity (which doesn't decay in space mode) and if you look around in a clockwise circle you will end up rotated anticlockwise in the roll axis. You can also use R and F to move up and down, Q and E for roll and the arrow keys for yaw and pitch.

## Projections
The 3D space is projected to the 2D surface of a sphere, which is projected to the 2D rectangle of your screen. The sphere-to-rectangle conversion is also a problem in map projection, this program implements some projection algorithms, you can choose which one by pressing the corresponding number key.
1. Azimuthal equidistant (each location on the sphere's direction and distance across the sphere's surface from the front is converted to direction and distance from the screen's centre (image is conformal at centre but is stretched laterally towards edges))
2. Lambert azimuthal equi-area (like equidistant except with distance through the sphere instead of across its surface (is compressed radially, counteracting how equidistant is stretched laterally, so all objects' apparent sizes are preserved but becomes very distorted)) (default mode)
3. Stereographic (each point is projected onto an infinite plane tangent to the sphere by following the line through itself from the point on the sphere opposite the plane, is locally conformal but doesn't preserve area and size must be infinite for it to work) (arbitrarily large circles on the sphere's surface become circles on the plane, so non-raytracing mode calculates balls' apparent positions and sizes based on their images' closest and furthest points from the plane's origin, it works very well)

If you like interesting ones that make you sick:

0. Unnamed (the first one I made before deciding to implement map projections, is like azimuthal equidistant but calculates X and Y separately based on the dot product of the point's position on the sphere to its basis vector (I had thought of azimuthal equidistant, but it instead makes a strange square where things deviate towards edges (if you increase the field of view to 360º, you can see that there's another square connected continuously at these vertices)))

## Raytracing
Press the t key at any time to stop the program and render the current scene with raytracing (in 960\*720, it takes a while but goes row-by-row in real time). Hold the p key on startup to use the real-time raytracing mode (with larger pixels but the same maximum number of reflections per pixel). Real-time mode's pixels are large enough that its use of a hexagonal lattice instead of a square one allows an 8% reduction in rays cast for the same "resolution" as perceived by human eyes (in terms of average distance of a random point on the screen from the closest pixel), though you can hold the k key also for the square lattice mode if you want (if it breaks or if you want to test whether it's actually better).)

Rays use the same functions to detect and calculate collisions with balls as balls do with each other, but without the one-frame time constraint.

Rays reflect up to 8 times by default (additional ones don't seem to have visible effect at current opacities). Balls each have a reflectiveness characteristic, when it's 1 they reflect colours perfectly, but as it reduces towards 0 they impart increasing colour changes, though the change to the ray's colour is an average weighted by the absolute change in velocity.

In raytracing mode, the 64-times-larger grey ball in regular mode is made of refractive glass (with Snell's law). It looks like its refracted images' edges are stretched but it's only because the observer and observed ball are too close to it, it looks more like how you'd expect if you go far away and zoom in. While reflective balls' images are perfectly black at their edges (where approaching rays are reflected less and less, with ones at tangents not reflecting at all), refractive ones' images are most opaque at their edges and transparent in their centres.

The speed of light is infinite by default (when the 'light speed' input to the raytracing function is 0) so the balls are in the same position from each ray's perspective. In a collision, a ray reflect off a ball like a mirror without caring about the ball's velocity, but setting light speed to a finite amount makes them behave like a ball bouncing off a moving car (causing the darkest band (where the change to the ray's velocity is negligible) to shift away from the edges of the ball's image, on one side of it they reflect a wider 'field of view' to the camera than on the other).

Enabling 'Minkowski mode' at finite light speeds makes rays detect interceptions with moving spheres instead of stationary ones, so the front of a ball moving rightwards will be further right than the back, it seems to make them look like ellipsoids (though balls only proceed backwards along their current trajectories, so they can appear to intersect each other, and collisions in real space make their images in Minkowski space appear to teleport).

Neither finite light speed nor Minkowski mode take additional computing power, and they can be set on the real-time mode's raytracing function call to experience for yourself (at the bottom of the main loop).

The gravity mode (enabled by changing the 'gravity' input parameter) simulates rays' acceleration due to gravity at each step in their trajectory. The gravity input itself (when not set to 0) describes the granularity of this sampling (setting it higher makes it take longer steps and experience more gravity during each). When enabled, the reflections input describe the number of steps instead of operations. Gravity takes many times more computing power, but rays can reflect off the same object multiple times consecutively and space appears stretched around massive objects and clusters (though it isn't actually curved [like Interstellar](https://arxiv.org/abs/1502.03809)), use it only in high-quality non-real-time mode (unless you're from the future and have more processing power).

## Play in browser
The reasonably up-to-date web version (where you control a ball in space with n-body gravity) is [here](https://turbowarp.org/575040192/fullscreen).

I don't update it with each commit, however, it will be outdated. For the latest one, [install TurboWarp desktop](https://turbowarp.org/desktop) and download and run [the .sb3 file in this repository](https://github.com/DroneBetter/Perspective3Dengine/blob/main/Perspective%203D%20engine.sb3).

## Credits
[@boraini](https://scratch.mit.edu/users/boraini) on Scratch for [the quaternion functions](https://turbowarp.org/454897467). See [my modification](https://turbowarp.org/574159176/fullscreen), originally made as a fix for mouse panning working in the wrong axes laterally after dragging a long way, but then I added the spheres and perspective rendering and movement controls. I recommend using this repository's program, however. 

The raytracing function implements the methods and equations described in [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html), the first book in [a series of three](https://raytracing.github.io/), they're freely licensed online, read them. This program deviates from theirs in aspects like light (which is imparted upon rays based on the magnitude of their reflections/refractions in this, instead of their eventual angle towards the sun) and has the real-time and Minkowski modes. Without the book, this program's collisions would still use quaternions instead of dot products for collisions (which were computationally expensive and had regions where balls passed through each other), and there would be no glass balls.
