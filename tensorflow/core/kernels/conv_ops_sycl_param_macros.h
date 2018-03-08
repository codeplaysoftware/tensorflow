#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_PARAM_MACROS_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_PARAM_MACROS_H_
/**
 * A collection of macros to help add and reference parameters in SYCL kernels.
 *
 * SNN_INJECT_CONV_PARAMS will add a declaration for all the variables in a
 * SYCLConv2DParams struct, but separated into separate variables. These
 * variables should be initialised in the kernel constructor using the
 * SNN_CONSTRUCT_CONV_PARAMS macro which takes the name of the SYCLConv2DParams
 * object to copy the values from.
 *
 * Inside the kernel the variables can be accessed using the SNN_PARAM(var_name)
 * macro. The var_name to pass in to the macro is the variable name in
 * SYCLConv2DParams that is required.
 *
 * If the kernel uses template parameters to provide static parameters, the
 * SNN_STATIC_PARAM macro should be used to use the static version if available,
 * or the runtime version otherwise.
 */
#define SNN_PARAM_NAME(x) param_##x
#define SNN_PARAM_ARG(x) const Index SNN_PARAM_NAME(x)
#define SNN_INJECT_CONV_PARAMS   \
  SNN_PARAM_ARG(channels_);      \
  SNN_PARAM_ARG(features_);      \
  SNN_PARAM_ARG(batch_);         \
  SNN_PARAM_ARG(in_rows_);       \
  SNN_PARAM_ARG(in_cols_);       \
  SNN_PARAM_ARG(window_rows_);   \
  SNN_PARAM_ARG(window_cols_);   \
  SNN_PARAM_ARG(stride_rows_);   \
  SNN_PARAM_ARG(stride_cols_);   \
  SNN_PARAM_ARG(out_rows_);      \
  SNN_PARAM_ARG(out_cols_);      \
  SNN_PARAM_ARG(pad_rows_);      \
  SNN_PARAM_ARG(pad_cols_);      \
  SNN_PARAM_ARG(dilation_rows_); \
  SNN_PARAM_ARG(dilation_cols_);

#define SNN_PARAM_CONSTRUCT(x, params) \
  SNN_PARAM_NAME(x) { params.x }

#define SNN_CONSTRUCT_CONV_PARAMS(params)          \
  SNN_PARAM_CONSTRUCT(channels_, params)           \
  , SNN_PARAM_CONSTRUCT(features_, params),        \
      SNN_PARAM_CONSTRUCT(batch_, params),         \
      SNN_PARAM_CONSTRUCT(in_rows_, params),       \
      SNN_PARAM_CONSTRUCT(in_cols_, params),       \
      SNN_PARAM_CONSTRUCT(window_rows_, params),   \
      SNN_PARAM_CONSTRUCT(window_cols_, params),   \
      SNN_PARAM_CONSTRUCT(stride_rows_, params),   \
      SNN_PARAM_CONSTRUCT(stride_cols_, params),   \
      SNN_PARAM_CONSTRUCT(out_rows_, params),      \
      SNN_PARAM_CONSTRUCT(out_cols_, params),      \
      SNN_PARAM_CONSTRUCT(pad_rows_, params),      \
      SNN_PARAM_CONSTRUCT(pad_cols_, params),      \
      SNN_PARAM_CONSTRUCT(dilation_rows_, params), \
      SNN_PARAM_CONSTRUCT(dilation_cols_, params)

#define SNN_PARAM(x) SNN_PARAM_NAME(x)
#define SNN_STATIC_PARAM(name, qual) \
  (static_##name > 0 ? static_##name : SNN_PARAM(name##_##qual))
#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_PARAM_MACROS_H_
