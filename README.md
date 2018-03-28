# interp_and_fill

Non-conservatively interpolate from a latitude-longitude grid to a curvilinear (tri-polar) ocean grid with
a given land/sea mask and fill in where needed.

## Usage

```
./interp_and_fill.py -h
./interp_and_fill.py ocean_hgrid.nc ocean_mask.nc source_data.nc new_file.nc
```

## Algorithm

1. Super-sample ocean grid by binary division until the number of degrees of freedom in the i-direction are finer than those of the source data.
2. Linearly interpolate source data to super-sampled grid.
   - No extrapolation or nearest neighbor interpolation.
3. Average super-sampled data to model cells.
4. Objectively fill-in missing data using an elliptic solver.
