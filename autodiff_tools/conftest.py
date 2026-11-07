import jax

# Finite-difference checks in this package need float64. Set it before any
# test module builds an array. Diffusion code pins its own arrays to the
# data dtype, so this does not force those scans onto mixed precision.
jax.config.update("jax_enable_x64", True)
