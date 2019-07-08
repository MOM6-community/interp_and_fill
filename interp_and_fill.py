#!/usr/bin/env python

import argparse
import netCDF4
import numpy
import scipy.sparse.linalg
import sys
import time

def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  """

  # Arguments
  parser = argparse.ArgumentParser(description=
      """
      regrid_runoff.py regrids runoff data from a regular/uniform latitude-longitude grid to a curvilinear ocean grid
      """,
      epilog='Written by A.Adcroft, 2018.')
  parser.add_argument('hgrid_file', type=str,
      help="""Filename for ocean horizontal grid (super-grid format).""")
  parser.add_argument('mask_file', type=str,
      help="""Filename for ocean mask.""")
  parser.add_argument('orig_file', type=str,
      help="""Filename for gridded source data on a regular spherical (lat-lon) grid.""")
  parser.add_argument('variable', type=str,
      help="""Name of variable in data file.""")
  parser.add_argument('out_file', type=str,
      help="""Filename for data interpolated to the ocean model grid.""")
  parser.add_argument('-m','--mask_var', type=str, default='mask',
      help="""Name of mask variable in mask file.""")
  parser.add_argument('-c','--closest', action='store_true',
      help="""Use closest neighbor when bilinear interpolation fails for missing data.""")
  parser.add_argument('--fms', action='store_true',
      help="""Add non-standard attributes for FMS!""")
  parser.add_argument('-p','--progress', action='store_true',
      help="""Report progress.""")
  parser.add_argument('-q','--quiet', action='store_true',
      help="""Disable informational messages.""")

  return parser.parse_args()

def info(msg):
  """Prints an informational message with trailing ... and no newline"""
  print(msg + ' ...', end='')
  sys.stdout.flush()
  return time.time()

def end_info(tic):
  """Closes the informational line"""
  print(' done in %.3fs.'%(time.time() - tic) )

def main(args):
  """
  Does everything.
  """

  start_time = time.time()
  pickle_file = 'pickle.regrid_runoff_A'

  # Open ocean grid
  if args.progress: tic = info('Reading ocean grid')
  ocn_qlon = netCDF4.Dataset(args.hgrid_file).variables['x'][::2,::2]   # Mesh longitudes (cell corners)
  ocn_qlat = netCDF4.Dataset(args.hgrid_file).variables['y'][::2,::2] # Mesh latitudes (cell corners)
  ocn_lon = netCDF4.Dataset(args.hgrid_file).variables['x'][1::2,1::2]    # Cell-center longitudes (cell centers)
  ocn_lat = netCDF4.Dataset(args.hgrid_file).variables['y'][1::2,1::2]  # Cell-center latitudes (cell centers)
  ocn_area = netCDF4.Dataset(args.hgrid_file).variables['area'][:]      # Super-grid cell areas
  ocn_area = ( ocn_area[::2,::2] + ocn_area[1::2,1::2] ) + ( ocn_area[1::2,::2] + ocn_area[::2,1::2] ) # Ocean-grid cell areas
  if args.progress: end_info(tic)

  if args.progress: tic = info('Reading ocean mask')
  ocn_mask = netCDF4.Dataset(args.mask_file).variables[args.mask_var][:] # 1=ocean, 0=land
  ocn_nj, ocn_ni = ocn_mask.shape
  if ocn_qlon.shape != (ocn_nj+1, ocn_ni+1): raise Exception('mask in mask_file has incompatible shape')
  if args.progress: end_info(tic)

  if not args.quiet: print('Ocean grid shape is %i x %i.'%(ocn_nj, ocn_ni))

  if args.progress: tic = info('Reading source grid')
  src_nc = netCDF4.Dataset(args.orig_file)
  src_data = src_nc.variables[args.variable]
  src_nj, src_ni = src_data.shape[-2], src_data.shape[-1]
  src_lon = src_nc.variables[src_data.dimensions[-1]]
  src_lat = src_nc.variables[src_data.dimensions[-2]]
  src_lat = ((numpy.arange(src_nj)+0.5)/src_nj - 0.5)*180. # Recompute as doubles
  src_x0 = int( ( src_lon[0] + src_lon[-1] )/2 + 0.5) - 180.
  src_lon = ((numpy.arange(src_ni)+0.5)/src_ni)*360.+src_x0 # Recompute as doubles
  src_qlat = ((numpy.arange(src_nj+1))/src_nj - 0.5)*180. # For plotting
  src_qlon = ((numpy.arange(src_ni+1))/src_ni)*360.+src_x0 # For plotting
  if args.progress: end_info(tic)

  if not args.quiet: print('Source data shape = %i x %i.'%(src_nj,src_ni))

  if args.progress: tic = info('Creating supersampled grid')
  spr_lat, spr_lon = super_sample_grid(ocn_qlat, ocn_qlon, ocn_mask, src_nj, src_ni)
  if args.progress: end_info(tic)

  if not args.quiet: print('Super grid shape = %i x %i x %i x %i.'%(spr_lat.shape))

  if args.progress: tic = info('Creating file')
  new_file = netCDF4.Dataset(args.out_file, 'w', 'clobber', format="NETCDF3_64BIT_OFFSET")

  # Axes
  new_file.createDimension('i',ocn_ni)
  new_file.createDimension('j',ocn_nj)
  new_file.createDimension('IQ',ocn_ni+1)
  new_file.createDimension('JQ',ocn_nj+1)
  extra_dim = None
  if len(src_data.shape)==3:
    extra_dim = src_data.dimensions[0]
    if src_nc.dimensions[extra_dim].isunlimited():
      new_file.createDimension(extra_dim, None)
    else:
      new_file.createDimension(extra_dim, len(src_nc.dimensions[extra_dim]))

  # 1d variables
  i = new_file.createVariable('i', 'f4', ('i',))
  i.long_name = 'Grid position along first dimension'
  if args.fms: i.cartesian_axis = 'X'
  j = new_file.createVariable('j', 'f4', ('j',))
  j.long_name = 'Grid position along second dimension'
  if args.fms: j.cartesian_axis = 'Y'
  I = new_file.createVariable('IQ', 'f4', ('IQ',))
  I.long_name = 'Grid position along first dimension'
  J = new_file.createVariable('JQ', 'f4', ('JQ',))
  J.long_name = 'Grid position along second dimension'
  if extra_dim is not None:
    extra_dim = src_nc.variables[extra_dim]
    t = new_file.createVariable(extra_dim.name, src_nc.variables[extra_dim.name].dtype, (extra_dim.name,))
    for a in src_nc.variables[extra_dim.name].ncattrs():
      t.setncattr(a, src_nc.variables[extra_dim.name].getncattr(a))
    if src_nc.dimensions[extra_dim.name].isunlimited():
      if args.fms:
        t.cartesian_axis = 'T'
        t.modulo = ' '

  # 2d variables
  lon = new_file.createVariable('lon', 'f4', ('j','i',))
  lon.long_name = 'Longitude of cell centers'
  lon.standard_name = 'longitude'
  lon.units = 'degrees_east'
  lat = new_file.createVariable('lat', 'f4', ('j','i',))
  lat.long_name = 'Latitude of cell centers'
  lat.standard_name = 'latitude'
  lat.units = 'degrees_north'
  lonq = new_file.createVariable('lon_crnr', 'f4', ('JQ','IQ',))
  lonq.long_name = 'Longitude of mesh nodes'
  lonq.standard_name = 'longitude'
  lonq.units = 'degrees_east'
  latq = new_file.createVariable('lat_crnr', 'f4', ('JQ','IQ',))
  latq.long_name = 'Latitude of mesh nodes'
  latq.standard_name = 'latitude'
  latq.units = 'degrees_north'
  area = new_file.createVariable('area', 'd', ('j','i',))
  area.long_name = 'Cell area'
  area.standard_name = 'cell_area'
  area.units = 'm2'
  area.coordinates = 'lon lat'
  area.mesh_coordinates = 'lon_crnr lat_crnr'
  if extra_dim is not None:
    new_var = new_file.createVariable(args.variable, 'f4', (extra_dim.name, 'j','i',))
  else:
    new_var = new_file.createVariable(args.variable, 'f4', ('j','i',))
  new_var.coordinates = 'lon lat'
  new_var.mesh_coordinates = 'lon_crnr lat_crnr'

  # variable attributes
  for a in src_data.ncattrs():
    new_var.setncattr(a, src_data.getncattr(a))
    if a == '_FillValue':
      if args.fms:
        new_var.missing_value = src_data.getncattr(a)

  # Write static data
  i[:] = numpy.arange(ocn_ni)+0.5
  j[:] = numpy.arange(ocn_nj)+0.5
  I[:] = numpy.arange(ocn_ni+1)
  J[:] = numpy.arange(ocn_nj+1)
  lon[:,:] = ocn_lon
  lat[:,:] = ocn_lat
  lonq[:,:] = ocn_qlon
  latq[:,:] = ocn_qlat
  area[:,:] = ocn_area

  if extra_dim is None:
    q_int = super_interp(src_lat, src_lon, src_data[:], spr_lat, spr_lon) * ocn_mask
    if args.closest:
      q_nrst = super_closest(src_lat, src_lon, src_data[:], spr_lat, spr_lon)
      q[ (ocn_mask>0) & q.mask ] = q_nrst[ (ocn_mask>0) & q.mask ]
    data = fill_missing_data(q_int, ocn_mask)
    new_var[:] = data[:]
  else:
    for n in range(src_data.shape[0]):
      if args.progress or not args.quiet:
        print(' %i'%(n), end='')
        sys.stdout.flush()
      q_int = super_interp(src_lat, src_lon, src_data[n], spr_lat, spr_lon)
      q_int = q_int.swapaxes(1,2).reshape((ocn_nj,ocn_ni,q_int.shape[3]*q_int.shape[-1])).mean(axis=-1)
      if type(q_int) is numpy.ndarray:
        q = numpy.ma.array( q_int, mask=( (ocn_mask==0) ) )
      elif type(q_int) is numpy.ma.core.MaskedArray:
        q = numpy.ma.array( q_int.filled(-1.e9), mask=(q_int.mask | (ocn_mask==0)) )
      else: raise Exception('Unknown type for variable q_int!')
      if args.closest:
        q_nrst = super_closest(src_lat, src_lon, src_data[n], spr_lat, spr_lon)
        q_nrst = q_nrst.swapaxes(1,2).reshape((ocn_nj,ocn_ni,q_nrst.shape[3]*q_nrst.shape[-1])).mean(axis=-1)
        q[ (ocn_mask>0) & q.mask ] = q_nrst[ (ocn_mask>0) & q.mask ]
      data = fill_missing_data(q, ocn_mask)
      new_var[n] = data[:]
      t[n] = src_nc.variables[extra_dim.name][n]
    if not args.progress and not args.quiet: print()

  new_file.close()
  if args.progress: end_info(tic)

def super_sample_grid(ocn_qlat, ocn_qlon, ocn_mask, src_nj, src_ni):
  nj, ni = ocn_mask.shape
  fac = 1
  while fac*nj<src_nj and fac*ni<src_ni:
    fac += 1
  lon = numpy.zeros( (nj,fac,ni,fac) )
  lat = numpy.zeros( (nj,fac,ni,fac) )
  mask = numpy.zeros( (nj,fac,ni,fac) )
  for j in range(fac):
    ya = ( 2*j+1 ) / ( 2*fac )
    yb = 1. - ya
    for i in range(fac):
      xa = ( 2*i+1 ) / ( 2*fac )
      xb = 1. - xa
      lon[:,j,:,i] = (  yb * ( xb * ocn_qlon[:-1,:-1] + xa * ocn_qlon[:-1,1:] )
                      + ya * ( xb * ocn_qlon[1:,:-1] + xa * ocn_qlon[1:,1:] ) )
      lat[:,j,:,i] = (  yb * ( xb * ocn_qlat[:-1,:-1] + xa * ocn_qlat[:-1,1:] )
                      + ya * ( xb * ocn_qlat[1:,:-1] + xa * ocn_qlat[1:,1:] ) )
  return lat, lon

def latlon2ji(src_lat, src_lon, lat, lon):
  nj, ni = len(src_lat), len(src_lon)
  src_x0 = int( ( src_lon[0] + src_lon[-1] )/2 + 0.5) - 180.
  j = numpy.maximum(0, numpy.floor( ( ( lat + 90. ) / 180. ) * nj - 0.5 ).astype(int))
  i = numpy.mod( numpy.floor( ( ( lon - src_x0 ) / 360. ) * ni - 0.5 ), ni ).astype(int)
  jp1 = numpy.minimum(nj-1, j+1)
  ip1 = numpy.mod(i+1, ni)
  return j,i,jp1,ip1

def super_interp(src_lat, src_lon, data, spr_lat, spr_lon):
  nj, ni = data.shape
  dy, dx = 180./nj, 360./ni
  j0, i0, j1, i1 = latlon2ji(src_lat, src_lon, spr_lat, spr_lon)
  def ydist(lat0, lat1):
    return numpy.abs( lat1-lat0 )
  def xdist(lon0, lon1):
    return numpy.abs( numpy.mod((lon1-lon0)+180, 360) - 180 )
  w_e = xdist( src_lon[i0], spr_lon) / dx
  w_w = 1. - w_e
  w_n = ydist( src_lat[j0], spr_lat) / dy
  w_s = 1. - w_n
  return ( w_s*w_w * data[j0,i0] + w_n*w_e * data[j1,i1] ) + ( w_n*w_w * data[j1,i0] + w_s*w_e * data[j0,i1] )

#def super_closest(src_lat, src_lon, data, spr_lat, spr_lon):
#  nj, ni = data.shape
#  src_x0 = int( ( src_lon[0] + src_lon[-1] )/2 + 0.5) - 180.
#  j = numpy.maximum(0, numpy.floor( ( ( spr_lat + 90. ) / 180. ) * nj ).astype(int))
#  i = numpy.mod( numpy.floor( ( ( spr_lon - src_x0 ) / 360. ) * ni ), ni ).astype(int)
#  return data[j,i]

def super_closest(src_lat, src_lon, data, spr_lat, spr_lon):
  """Uses piecewise constant reconstruction to map src onto a grid twice as fine and then interpolate.
     This avoids excessive extrapolation that occurs with closest neighbor near land."""
  nj, ni = data.shape
  nj, ni = 2*nj, 2*ni
  data2 = numpy.ma.zeros((nj,ni))
  data2[::2,::2] = data
  data2[1::2,::2] = data
  data2[::2,1::2] = data
  data2[1::2,1::2] = data
  src_lon2 = numpy.zeros( ni )
  src_lon2[0] = src_lon[0] - ( src_lon[1] - src_lon[0] )/4
  src_lon2[-1] = src_lon[-1] - ( src_lon[-2] - src_lon[-1] )/4
  src_lon2[1:-2:2] = ( 3 * src_lon[:-1] + src_lon[1:] ) / 4
  src_lon2[2:-1:2] = ( src_lon[:-1] + 3 * src_lon[1:] ) / 4
  src_lat2 = numpy.zeros( nj )
  src_lat2[0] = src_lat[0] - ( src_lat[1] - src_lat[0] )/4
  src_lat2[-1] = src_lat[-1] - ( src_lat[-2] - src_lat[-1] )/4
  src_lat2[1:-2:2] = ( 3 * src_lat[:-1] + src_lat[1:] ) / 4
  src_lat2[2:-1:2] = ( src_lat[:-1] + 3 * src_lat[1:] ) / 4
  return super_interp(src_lat2, src_lon2, data2, spr_lat, spr_lon)

def fill_missing_data(idata, mask, verbose=False, maxiter=0, debug=False, stabilizer=1.e-14):
  """
  Returns data with masked values objectively interpolated except where mask==0.
  
  Arguments:
  data - numpy.ma.array with mask==True where there is missing data or land.
  mask - numpy.array of 0 or 1, 0 for land, 1 for ocean.
  
  Returns a numpy.ma.array.
  """
  nj,ni = idata.shape
  fdata = idata.filled(0.) # Working with an ndarray is faster than working with a masked array
  missing_j, missing_i = numpy.where( idata.mask & (mask>0) )
  n_missing = missing_i.size
  if verbose:
      print('Data shape: %i x %i = %i with %i missing values'%(nj, ni, nj*ni, numpy.count_nonzero(idata.mask)))
      print('Mask shape: %i x %i = %i with %i land cells'%(mask.shape[0], mask.shape[1],
                                                               numpy.prod(mask.shape), numpy.count_nonzero(1-mask)))
      print('Data has %i missing values in ocean'%(n_missing))
      print('Data range: %g .. %g '%(idata.min(),idata.max()))
  # ind contains column of matrix/row of vector corresponding to point [j,i]
  ind = numpy.zeros( fdata.shape, dtype=int ) - int(1e6)
  ind[missing_j,missing_i] = numpy.arange( n_missing )
  if verbose: print('Building matrix')
  A = scipy.sparse.lil_matrix( (n_missing, n_missing) )
  b = numpy.zeros( (n_missing) )
  ld = numpy.zeros( (n_missing) )
  A[range(n_missing),range(n_missing)] = 0.
  if verbose: print('Looping over cells')
  for n in range(n_missing):
    j,i = missing_j[n],missing_i[n]
    im1 = ( i + ni - 1 ) % ni
    ip1 = ( i + 1 ) % ni
    jm1 = max( j-1, 0)
    jp1 = min( j+1, nj-1)
    if j>0 and mask[jm1,i]>0:
      ld[n] -= 1.
      ij = ind[jm1,i]
      if ij>=0:
        A[n,ij] = 1.
      else:
        b[n] -= fdata[jm1,i]
    if mask[j,im1]>0:
      ld[n] -= 1.
      ij = ind[j,im1]
      if ij>=0:
        A[n,ij] = 1.
      else:
        b[n] -= fdata[j,im1]
    if mask[j,ip1]>0:
      ld[n] -= 1.
      ij = ind[j,ip1]
      if ij>=0:
        A[n,ij] = 1.
      else:
        b[n] -= fdata[j,ip1]
    if j<nj-1 and mask[jp1,i]>0:
      ld[n] -= 1.
      ij = ind[jp1,i]
      if ij>=0:
        A[n,ij] = 1.
      else:
        b[n] -= fdata[jp1,i]
    if j==nj-1 and mask[j,ni-1-i]>0: # Tri-polar fold
      ld[n] -= 1.
      ij = ind[j,ni-1-i]
      if ij>=0:
        A[n,ij] = 1.
      else:
        b[n] -= fdata[j,ni-1-i]
  # Set leading diagonal
  b[ld>=0] = 0.
  A[range(n_missing),range(n_missing)] = ld - stabilizer
  if verbose: print('Matrix constructed')
  A = scipy.sparse.csr_matrix(A)
  if verbose: print('Matrix converted')
  new_data = numpy.ma.array( fdata, mask=(mask==0))
  if maxiter is None:
    x,info = scipy.sparse.linalg.bicg(A, b)
  elif maxiter==0:
    x = scipy.sparse.linalg.spsolve(A, b)
  else:
    x,info = scipy.sparse.linalg.bicg(A, b, maxiter=maxiter)
  if verbose: print('Matrix inverted')
  new_data[missing_j,missing_i] = x
  return new_data

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__':
  args = parseCommandLine()
  main(args)
