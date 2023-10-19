from functools import wraps
import time

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numba import jit, prange


# Constants
# ========
FPS = 30
T_IN_SEC = 1
GRID_W = 200
GRID_H = 200
ZEROS = np.zeros((GRID_W, GRID_H))
INIT_POP = 0.01
R_PROP = 3  # radius of dispersal for cell propagation
R_AB = 0  # radius of dispersal for antibiotics
R_DEG = 0  # radius of dispersal for antibiotics degrader
R_SURV = np.maximum(R_AB, R_DEG)
# ========


def timefn(fn):
    """wrapper to time the enclosed function"""

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: {} took {} seconds".format(fn.__name__, t2 - t1))
        return result

    return measure_time


def initialize_grid():
    """
    Generate start grid, with species 1, 2, and 3 populating 1% of the space
    Return this grid to main function
    """
    grid = ZEROS.copy()
    if R_PROP <= 20:
        for x in range(GRID_W):
            for y in range(GRID_H):
                state = np.random.random()
                if state <= INIT_POP:
                    grid[y][x] = np.random.randint(1, 4)
    else:
        for x in range(GRID_W):
            for y in range(GRID_H):
                _x = x / GRID_W
                _y = y / GRID_H
                if _x + _y >= 1.0 and _x - _y <= 0.0:
                    grid[y][x] = 1
                if _x + _y <= 1.0 and _x <= 0.5:
                    grid[y][x] = 2
                if _x - _y >= 0.0 and _x >= 0.5:
                    grid[y][x] = 3
    return grid


# @jit(nopython=True, parallel=False)
def update_grid(grid):
    """Calculate the next iteration of the grid using the previous grid
    :param grid: 2D grid of dead/alive cells
    :returns: 2D grid of dead/alive cells
    """

    for y in prange(GRID_W):  # iterate through grid
        for x in range(GRID_H):
            # accumulator for cells w/in antibiotic radius
            total_ab = np.zeros(3)
            # accumulator for cells w/in antibiotic degrader radius
            total_deg = np.zeros(3)

            # cycle through antibiotic/degrader radius
            for j in range(-R_SURV, (R_SURV + 1)):
                for i in range(-R_SURV, (R_SURV + 1)):
                    # take cell tally for antibiotic producers
                    if (i * i + j * j) < (R_AB + 0.5) ** 2:
                        if grid[(y + j) % GRID_H][(x + i) % GRID_W] == 1:
                            total_ab[0] += 1
                        elif grid[(y + j) % GRID_H][(x + i) % GRID_W] == 2:
                            total_ab[1] += 1
                        elif grid[(y + j) % GRID_H][(x + i) % GRID_W] == 3:
                            total_ab[2] += 1
                    # take cell tally for antibiotic degradors
                    if (i * i + j * j) < (R_DEG + 0.5) ** 2:
                        if grid[(y + j) % GRID_H][(x + i) % GRID_W] == 1:
                            total_deg[0] += 1
                        elif grid[(y + j) % GRID_H][(x + i) % GRID_W] == 2:
                            total_deg[1] += 1
                        elif grid[(y + j) % GRID_H][(x + i) % GRID_W] == 3:
                            total_deg[2] += 1

            # Create a lookup table for this
            # determine outcomes for cells
            if grid[y][x] == 1 and (total_deg[1] < 1 <= total_ab[2]):
                grid[y][x] = 0
            if grid[y][x] == 2 and (total_deg[2] < 1 <= total_ab[0]):
                grid[y][x] = 0
            if grid[y][x] == 3 and (total_deg[0] < 1 <= total_ab[1]):
                grid[y][x] = 0

    grid_new = grid.copy()
    # iterate through list of empty spots and determine what cell type grows
    for y in prange(GRID_W):  # iterate through grid
        for x in range(GRID_H):
            if grid[y][x]:
                continue
            sp_present = np.zeros(3)
            for j in range(-R_PROP, (R_PROP + 1)):
                for i in range(-R_PROP, (R_PROP + 1)):
                    if (i * i + j * j) <= (R_PROP + 0.5) ** 2:
                        if grid[(y + j) % GRID_H][(x + i) % GRID_W] == 1:
                            sp_present[0] = 1
                        elif grid[(y + j) % GRID_H][(x + i) % GRID_W] == 2:
                            sp_present[1] = 1
                        elif grid[(y + j) % GRID_H][(x + i) % GRID_W] == 3:
                            sp_present[2] = 1

            if sp_present.any():
                grid_new[y][x] = np.random.choice(np.nonzero(sp_present)[0]) + 1

    return grid_new


# Function for generating population counts; called by draw_N()
def value_count(grid, normalize=False, remove_blank=True):
    count = np.array([(grid == i).sum() for i in range(4)])
    if normalize:
        count = count / (GRID_W * GRID_H)
    if remove_blank:
        count = count[1:]
    return count


def to_rgb(grid):
    colored_grid = np.zeros((grid.shape[0], grid.shape[1], 3))
    colored_grid[:, :, 0] = grid == 1
    colored_grid[:, :, 1] = grid == 2
    colored_grid[:, :, 2] = grid == 3
    return colored_grid


def update(frame, im, c_plot, grid, counts):
    """function which is called each tick of the animation
    :param frame: The current frame index
    :type frame: int
    :param im: The image being updated
    :type im: matplotlib imshow
    :param grid: 2D grid of dead/alive cells
    :type grid: np.array
    :returns: updated image
    """
    im.set_array(to_rgb(grid))
    t_count = np.array(counts)
    for i in range(3):
        c_plot[i].set_data(np.arange(t_count.shape[0]), t_count[:, i])
    new_grid = update_grid(grid)
    counts.append(value_count(new_grid, normalize=True))
    grid[:] = new_grid[:]
    return (
        im,
        c_plot,
    )


# this just runs the grid update without rendering an image; useful for optimization
@timefn
def run_N(N):
    grid = initialize_grid()
    cell_count = []
    for i in range(N):
        cell_count.append(value_count(grid, normalize=True))
        grid[:] = update_grid(grid)[:]

    return cell_count, grid


def animate():
    colors = {0: "black", 1: "red", 2: "green", 3: "blue"}
    grid = initialize_grid()
    counts = [value_count(grid, normalize=True)]
    fig, ax = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={"height_ratios": [4, 1]})
    im = ax[0].imshow(
        to_rgb(grid),
        aspect="equal",
        interpolation="none",  # this sets grid resolution at screen resolution
    )
    c_plot = ax[1].plot(counts)
    [ax[1].lines[i].set_color(color) for i, color in enumerate("rgb")]
    ax[1].set_xlim(0, FPS * T_IN_SEC)
    ax[1].set_ylim(0, 1)

    ani = FuncAnimation(
        fig,
        update,
        fargs=(im, c_plot, grid, counts),
        frames=FPS * T_IN_SEC,
        interval=1000 / FPS,
        repeat=False,  # stop rendering once you've hit max number of frames
        blit=True,  # prevents redrawing pixels that haven't changed
    )

    return ani


if __name__ == "__main__":
    result = run_N(5)
    print(result)
