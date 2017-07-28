# Shape-Sorting

### Files

**game.py**: Main source file, including definition for the shape sorting environment and functions for running it for human, and random play.

**initializers.py** Functions specifying rules for how to initialize blocks and holes on the screen.

**shape_zoo.py** Shape classes used for both blocks and holes. Each has an associated render and rotate method.

**game_settings.py** Configuration file containing lists and dictionaries defining general settings (e.g., what types of blocks to use, what size) for the shape sorting envrionment.

**util.py** Wrapper allowing "registration" of the environment. Saver object that uses h5py.

### Installing

```
git clone https://github.com/akuefler/shape-sorting.git
```

### Requirements
Gym

shapely >= 1.5.13

pygame >= 1.9.2b8

h5py >= 2.6.0

