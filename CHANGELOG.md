# Changelog

# v0.28.5
 - Fixes some issues with threading.

# v0.28.4
 - Improves speed of SMC^2 when we require an increase in state particles by checking acceptance rate threshold in kernel instead.

# v0.28.3
 - Check average acceptance rate for MH kernel instead.

# v0.28.2
 - Bug fix for matrix shaped parameters.

# v0.28.1
 - Use `torch.Tensor.copy_` instead of `fill_`.

# v0.28.0
 - Uses stoch-proc version 0.3.0, which means that whenever the context is updated in an inference algorithm we rebuild the model.
 - Speed improvement for SISR.

# v0.27.0
 - Major change in backend by interchaning batch and sample shape in inference algorithms. This enables much more natural handling of parameters.
 - Adds support for finding mode of distribution by means of `functorch

# v0.26.1
 - Adds the Gaussian Particle Filter
 - Reworks correct/predict logic by enforcing that correct only takes a prediction as input
 - Simplifies proposal logic
 - Uses latest version of stoch-proc

# v0.26.0
 - Reworks how we define parameters by initializing all parameters to their "correct" shapes from the get-go by introducing a shape object stored on `InferenceContext`.

# v0.25.1
 - Reworks `ParameterContext` logic by removing the requirements for the context being on the stack in order to register parameters.
 - Adds fixed-lag smoothing.
 - Renames `ParameterContext` to `InferenceContext`

# v0.24.11
 - Adds support for using QMC points in `SMC2`

# v0.24.10
 - Utilizes `int` instead of `long`

# v0.24.5
 - Adds a proposal for locally linearized observation dynamics.

# v0.24.4
 - Adds support for performing variational inference using particle filter.

# v0.24.3
 - Improves plotting functionality by handling multi-dimensional parameters.

# v0.24.2
 - Adds support for applying function on context and return a copy of that context.
