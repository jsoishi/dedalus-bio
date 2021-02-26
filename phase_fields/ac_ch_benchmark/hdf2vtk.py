"""hdf2vtk: convert Dedalus fields stored in HDF5 files to vtk format for 3D visualization

Usage:
    hdf2vtk [--fields=<fields> --nt=<nt>] <input_file> [<output_file>]

Options:
    --fields=<fields>           comma separated list of fields to extract from the hdf5 file [default: None]
    --nt=<nt>                   time index [default: -1]

"""
from dedalus.extras import plot_tools
from pathlib import Path
from docopt import docopt
from pyevtk.hl import gridToVTK
import h5py
import numpy as np
H5_FIELD_PATH = 'tasks/'
H5_SCALE_PATH = 'scales/'
H5_DIM_LABEL = 'DIMENSION_LABELS'
H5_STR_DECODE = 'UTF-8'

if __name__ == "__main__":
    args = docopt(__doc__ )

    nt = int(args['--nt'])
    
    fields = args['--fields']
    if fields == 'None':
        raise ValueError("Must specify fields to copy.")
    fields = fields.split(',')
    print("fields = {}".format(fields))

    infile = Path(args['<input_file>'])
    if args['<output_file>']:
        outfile = args['<output_file>']
    else:
        outfile = infile.stem

    print("outfile = {}".format(outfile))

    datafile = h5py.File(infile,"r")

    field_names = [H5_FIELD_PATH+f for f in fields]
    dim_labels = datafile[field_names[0]].attrs[H5_DIM_LABEL][1:]
    if len(dim_labels) != 3:
        raise NotImplementedError("hdf2vtk only supports 3D data.")

    # currently cartesian only
    scale_names = [H5_SCALE_PATH+d.decode(H5_STR_DECODE) for d in dim_labels]
    # just get first scale you find...
    grid_scale = list(datafile[scale_names[0]].keys())[0]
    scale_names = [sn+'/'+grid_scale for sn in scale_names]
    x = plot_tools.get_1d_vertices(datafile[scale_names[0]][:])
    y = plot_tools.get_1d_vertices(datafile[scale_names[1]][:])
    z = plot_tools.get_1d_vertices(datafile[scale_names[2]][:])

    cellData = {}
    for i, f in enumerate(fields):
        #cellData[f] = np.asfortranarray(datafile[field_names[i]][nt])
        cellData[f] = datafile[field_names[i]][nt]

    gridToVTK(outfile, x, y, z, cellData = cellData)
