# AMR Visualisations
Simple python animated examples for the planning and dynamics segments of the Autonomous Mobile Robots lecture slides. Primarily for students of the [course](https://asl.ethz.ch/education/lectures/autonomous_mobile_robots.html) but open to anyone interested.


## Basic installation and requirements
Clone the repo, install dependencies
```
git clone https://github.com/ethz-asl/amr_visualisations.git
cd amr_visualisations
pip install -r requirements.txt
``` 

## Configuration space examples

These are some basic examples of generating configuration spaces and illustrating them.

First, enter the directory `cd config_space`

### 3D (x,y,&theta;) robot configuration space
This simple script uses a basic polygon collision checking library and sampling to show the configuration space for a 2D robot.
Obstacles are generated by sampling random points and creating a convex hull.

A basic example can be generated with:
```
python config_space_plot.py
```
![config_space_slices](https://user-images.githubusercontent.com/10678827/81062882-16cf0400-8ed7-11ea-9d36-697450a56593.png)


Help for the flags etc. are available with:
```
python config_space_plot.py -h
```

Some things you might like to change include:
 - The robot footprint is defined with a csv file ([example](config_space/robots/bar_robot.csv)) and set with the `--robot-footprint` flag
 - You can change the sampling density with the `-nx` flag
 - Generate the 3D rotation animation with the `--animation` flag 

![config_space_rotation](https://user-images.githubusercontent.com/10678827/81062839-0585f780-8ed7-11ea-8619-5da014477b18.gif)

### Robot arm configuration space
This script shows the configuration space for a basic multi-jointed robot arm.
A basic example can be generated with:
```
python arm_config_space.py
```

Again, basic help can be found with
```
python arm_config_space.py -h
```
The robot parameters and workspace obstacles are defined with a basic yaml file. 
The default configuration can be found in the `config/block_world.yaml` file.
To set a new workspace, create a copy of the `block_world.yaml` file, edit the robot and/or obstacles, and use the `-w` flag to specify your new world file:
```
python arm_config_space.py -w config/my_new_world.yaml
``` 
Currently, only a basic jointed robot arm and polygonal obstacles are implemented.

![arm_config_space_video](https://user-images.githubusercontent.com/10678827/81062807-eedfa080-8ed6-11ea-8d94-a39898cf47cd.gif)

### Rapidly-exploring Random Tree (RRT)
Basic example of an RRT search showing the Voronoi regions to demonstrate how an RRT demonstrates a 'space-filling' effect.
A basic example can be generated with:
```
python rrt_simple.py
```
The workspace definition can be changed to another file using the `--world` argument, number of iterations with `-i` and other options (try `python rrt_simple.py -h` for more help).

![rrt_video](https://user-images.githubusercontent.com/10678827/127008176-1b9e58d1-330a-45af-a9b2-ebdbfe8938de.gif)

### Potential field example
```
python potential_field.py
```
![rotating_potentialfieldobs](https://user-images.githubusercontent.com/10678827/81062961-3e25d100-8ed7-11ea-917e-fd2a75dafec1.gif)

## Poincaré example
```
cd poincare_example
python poincare_1d.py
```
![poincare_example_ffm](https://user-images.githubusercontent.com/10678827/81069466-2ef85080-8ee2-11ea-9317-88f10991f86b.gif)

## Feedback
Comments, feedback and PRs for changes are welcome (nicholas.lawrance AT mavt.ethz.ch).
