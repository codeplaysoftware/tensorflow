/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_MIRROR_PAD_OP_CPU_IMPL_H_
#define TENSORFLOW_CORE_MIRROR_PAD_OP_CPU_IMPL_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/mirror_pad_op.h"

namespace tensorflow {

using CpuDevice = Eigen::ThreadPoolDevice;
#ifdef TENSORFLOW_USE_SYCL
using SYCLDevice = Eigen::SyclDevice;
#endif  // TENSORFLOW_USE_SYCL

#define DEFINE_CPU_SPECS(T)                                                    \
  template struct functor::MirrorPad<CpuDevice, T, int32, CPU_PROVIDED_IXDIM>; \
  template struct functor::MirrorPad<CpuDevice, T, int64, CPU_PROVIDED_IXDIM>;
TF_CALL_POD_TYPES(DEFINE_CPU_SPECS);
TF_CALL_string(DEFINE_CPU_SPECS);
#undef DEFINE_CPU_SPECS

#define DEFINE_CPU_SPECS(T)                                   \
  template struct functor::MirrorPadGrad<CpuDevice, T, int32, \
                                         CPU_PROVIDED_IXDIM>; \
  template struct functor::MirrorPadGrad<CpuDevice, T, int64, \
                                         CPU_PROVIDED_IXDIM>;
TF_CALL_NUMBER_TYPES(DEFINE_CPU_SPECS);
#undef DEFINE_CPU_SPECS

#ifdef TENSORFLOW_USE_SYCL
#define DEFINE_SYCL_SPECS(T)                                                    \
  template struct functor::MirrorPad<SYCLDevice, T, int32, CPU_PROVIDED_IXDIM>; \
  template struct functor::MirrorPad<SYCLDevice, T, int64, CPU_PROVIDED_IXDIM>;
TF_CALL_SYCL_NUMBER_TYPES(DEFINE_SYCL_SPECS);
#undef DEFINE_SYCL_SPECS

#define DEFINE_SYCL_SPECS(T)                                   \
  template struct functor::MirrorPadGrad<SYCLDevice, T, int32, \
                                         CPU_PROVIDED_IXDIM>;  \
  template struct functor::MirrorPadGrad<SYCLDevice, T, int64, \
                                         CPU_PROVIDED_IXDIM>;
TF_CALL_SYCL_NUMBER_TYPES(DEFINE_SYCL_SPECS);
#undef DEFINE_SYCL_SPECS
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_MIRROR_PAD_OP_CPU_IMPL_H_
