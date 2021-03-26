"""
Plot planes from joint analysis files.


Usage:
    plot_slices.py <files>... [--output=<dir>]


Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from dedalus.extras import plot_tools

def plot_z_mid(filename, start, count, output):
    """plots middle z slice from 3D data cube"""

    # Plot settings
    tasks = ['c', 'η']

    scale = 2
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'zslice_{:06}.png'.format(write)

    # Layout
    nrows, ncols = 1, 2
    image = plot_tools.Box(2,2)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)    

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call plotting helper (dset axes: [t, x, y, z])
                dset = file['tasks'][task]
                z_slice_index = dset.shape[1]//2 - 1
                print("z_sl = {}".format(z_slice_index))
                image_axes = (3,2)
                data_slices = (index, z_slice_index, slice(None), slice(None))
                plot_tools.plot_bot(dset, image_axes, data_slices, axes=axes, title=task, even_scale=True)
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.48, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], plot_z_mid, output=output_path)
