/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/avgpooling_op.h"

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/maxpooling_op_gpu.h"
#include "tensorflow/core/kernels/pooling_ops_common_gpu.h"
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"

#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/operators.h"
#include "sycldnn/backend/eigen_backend.h"
#endif  // TENSORFLOW_USE_SYCL

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename T>
class AvgPoolingOp : public UnaryOp<T> {
 public:
  explicit AvgPoolingOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default AvgPoolingOp only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }
    OP_REQUIRES(context, params.depth_window == 1,
                errors::Unimplemented("Non-spatial pooling is not "
                                      "yet supported. Volunteers? :)"));

    // For avgpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));

    SpatialAvgPool<Device, T>(context, output, tensor_in, params, padding_);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(
    Name("AvgPool").Device(DEVICE_CPU).TypeConstraint<double>("T"),
    AvgPoolingOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(
    Name("AvgPool").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    AvgPoolingOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("AvgPool").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("T"),
    AvgPoolingOp<CPUDevice, Eigen::half>);

#if GOOGLE_CUDA
template <typename T>
class AvgPoolingOp<GPUDevice, T> : public UnaryOp<T> {
 public:
  typedef GPUDevice Device;
  explicit AvgPoolingOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    if (!context->status().ok()) {
      return;
    }
    OP_REQUIRES(context, params.depth_window == 1,
                errors::Unimplemented("Non-spatial pooling is not "
                                      "yet supported. Volunteers? :)"));

    // For avgpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 4-dimensional"));

    TensorShape output_shape = params.forward_output_shape();

    if (data_format_ == FORMAT_NCHW) {
      DnnPoolingOp<T>::Compute(context, se::dnn::PoolingMode::kAverage, ksize_,
                               stride_, padding_, data_format_, tensor_in,
                               output_shape,
                               /*propagate_nans=*/false);
    } else {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
      Eigen::PaddingType pt = BrainPadding2EigenPadding(padding_);
      functor::SpatialAvgPooling<Device, T>()(
          context->eigen_device<Device>(), output->tensor<T, 4>(),
          tensor_in.tensor<T, 4>(), params.window_rows, params.window_cols,
          params.row_stride, params.col_stride, pt);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                      \
  template <>                                                    \
  void SpatialAvgPooling<GPUDevice, T>::operator()(              \
      const GPUDevice& d, typename TTypes<T, 4>::Tensor output,  \
      typename TTypes<T, 4>::ConstTensor input, int window_rows, \
      int window_cols, int row_stride, int col_stride,           \
      const Eigen::PaddingType& padding);                        \
  extern template struct SpatialAvgPooling<GPUDevice, T>;

DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNEL_BUILDER(
    Name("AvgPool").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    AvgPoolingOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("AvgPool").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    AvgPoolingOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("AvgPool").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    AvgPoolingOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
// AvgPool2D SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output tensor.
//
// For each output value find the corresponding input window, and run through
// the window accumulating the values to form an average. We divide each value
// before accumulating to prevent the accumulator from becoming significantly
// bigger than the values we are adding and so decrease any errors.
template <typename T>
class AvgPool2DSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;
  public:
  AvgPool2DSYCL(const int depth, const int batch, const int in_rows,
                const int in_cols, const int out_rows, const int out_cols,
                const std::array<int64, 2>& window,
                const std::array<int64, 2>& stride,
                const std::array<int64, 2>& padding,
                const read_accessor input_accessor,
                write_accessor output_accessor)
      : p_(depth, batch, in_rows, in_cols, out_rows, out_cols, window, stride,
           padding),
        input_accessor_(input_accessor),
        output_accessor_(output_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_data = ConvertToActualTypeSycl(T, input_accessor_);
    T* output_data = ConvertToActualTypeSycl(T, output_accessor_);
     int index = item.get_linear_id();
    int n = index;
    int d = n % p_.depth_;
    n /= p_.depth_;
    int cstart = (n % p_.out_cols_) * p_.stride_cols_ - p_.pad_cols_;
    int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
    cstart = std::max(cstart, 0);
    n /= p_.out_cols_;
    int rstart = (n % p_.out_rows_) * p_.stride_rows_ - p_.pad_rows_;
    int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
    rstart = std::max(rstart, 0);
    n /= p_.out_rows_;
    T accum = T(0);
    T count = static_cast<T>((rend - rstart) * (cend - cstart));
    const T* input_data_n =
        input_data + n * p_.in_cols_ * p_.in_rows_ * p_.depth_;
    for (int r = rstart; r < rend; ++r) {
      for (int c = cstart; c < cend; ++c) {
        int idx = (r * p_.in_cols_ + c) * p_.depth_ + d;
        accum += input_data_n[idx] / count;
      }
    }
    output_data[index] = accum;
  }
  private:
  const SYCL2DPoolParams p_;
  const read_accessor input_accessor_;
  write_accessor output_accessor_;
};
 template <typename T>
struct LaunchAvgPoolingOpSYCL {
  static void launch(OpKernelContext* context, const Tensor& tensor_in,
                     const std::array<int64, 2>& window,
                     const std::array<int64, 2>& stride,
                     const std::array<int64, 2>& padding,
                     TensorFormat data_format, Padding padding_type,
                     Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int out_rows = GetTensorDim(*output, data_format, '0');
    const int out_cols = GetTensorDim(*output, data_format, '1');
    const int batch = GetTensorDim(tensor_in, data_format, 'N');
    const int in_rows = GetTensorDim(tensor_in, data_format, '0');
    const int in_cols = GetTensorDim(tensor_in, data_format, '1');
    const int depth = GetTensorDim(tensor_in, data_format, 'C');
     const int num_threads = output->NumElements();
     auto input_buffer =
        device.get_sycl_buffer(tensor_in.template flat<T>().data());
    auto output_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());
     device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_access =
          input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
      AvgPool2DSYCL<T> avg_pool(depth, batch, in_rows, in_cols, out_rows,
                                out_cols, window, stride, padding, input_access,
                                output_access);
       cgh.parallel_for(cl::sycl::range<1>(num_threads), avg_pool);
    });
  }
};

template <typename T>
class AvgPoolingOp<SYCLDevice, T> : public UnaryOp<T> {
 public:
  explicit AvgPoolingOp(OpKernelConstruction* context) : UnaryOp<T>(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default AvgPoolingOp only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window stride field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in = context->input(0);
    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, tensor_in.shape()};
    namespace sd = sycldnn::pooling;
    sd::PoolingParams sd_params = get_sd_params(params);
    if (!context->status().ok()) {
      return;
    }
    OP_REQUIRES(context, params.depth_window == 1,
                errors::Unimplemented("Non-spatial pooling is not "
                                      "yet supported."));

    OP_REQUIRES(context, tensor_in.dims() == 4,
                errors::InvalidArgument("tensor_in must be 1-dimensional and 4 "
                                        "elements"));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, params.forward_output_shape(), &output));

    // This is not an error in TensorFlow, the context expect an empty output
    // in this case.
    if (sd_params.batch == 0)
      return;

    if (!is_snn_enabled()) {
      const int64 depth = GetTensorDim(tensor_in, data_format_, 'C');
      const int64 in_batch = GetTensorDim(tensor_in, data_format_, 'N');
       // Dimension order for these arrays is x, y.
      std::array<int64, 2> input_size{
          {GetTensorDim(tensor_in, data_format_, '1'),
           GetTensorDim(tensor_in, data_format_, '0')}};
      std::array<int64, 2> window{{GetTensorDim(ksize_, data_format_, '1'),
                                   GetTensorDim(ksize_, data_format_, '0')}};
      std::array<int64, 2> stride{{GetTensorDim(stride_, data_format_, '1'),
                                   GetTensorDim(stride_, data_format_, '0')}};
      std::array<int64, 2> out, padding;
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(input_size[0], window[0], stride[0],
                                           padding_, &out[0], &padding[0]));
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(input_size[1], window[1], stride[1],
                                           padding_, &out[1], &padding[1]));
      LaunchAvgPoolingOpSYCL<T>::launch(context, tensor_in, window, stride,
                                        padding, data_format_, padding_, output);
    }
    else {
      auto device = context->eigen_device<SYCLDevice>();
      sycldnn::backend::EigenBackend backend(device);
      auto in_ptr = tensor_in.template flat<T>().data();
      auto out_ptr = output->template flat<T>().data();

      auto status = sd::launch<T, sd::Average, sd::Forward>(in_ptr, out_ptr,
          sd_params, backend);
      if (status.status != sycldnn::StatusCode::OK) {
        context->SetStatus(get_sd_err_msg(status));
        return;
      }
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

#define REGISTER_AVGPOOL_SYCL(type)                                       \
REGISTER_KERNEL_BUILDER(Name("AvgPool")                                   \
                            .Device(DEVICE_SYCL)                          \
                            .TypeConstraint<type>("T"),                   \
                        AvgPoolingOp<SYCLDevice, type>);
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_AVGPOOL_SYCL);
#undef REGISTER_AVGPOOL_SYCL
#endif  // TENSORFLOW_USE_SYCL

// The operation to compute AvgPool gradients.
// It takes two inputs:
//   - The original input tensor shape
//   - Backprop tensor for output
// It produces one output: backprop tensor for input.
template <typename Device, class T>
class AvgPoolingGradOp : public OpKernel {
 public:
  explicit AvgPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default AvgPoolingGradOp only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
        errors::InvalidArgument("out_backprop must be 1-dimensional and 4 "
                                "elements"));
    // For avgpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));
    const int64 out_backprop_batch = out_backprop.dim_size(0);
    const int64 out_backprop_rows = out_backprop.dim_size(1);
    const int64 out_backprop_cols = out_backprop.dim_size(2);
    const int64 out_backprop_depth = out_backprop.dim_size(3);

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }
    const int64 in_rows = output_shape.dim_size(1);
    const int64 in_cols = output_shape.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    output->flat<T>().setZero();

    const int window_rows = ksize_[1];
    const int window_cols = ksize_[2];
    const int depth_window = ksize_[3];

    const int row_stride = stride_[1];
    const int col_stride = stride_[2];

    // We (will) use different code for spatial pooling and
    // non-spatial pooling.
    //
    // Spatial pooling is when depth_window = 1
    OP_REQUIRES(context, depth_window == 1,
                errors::Unimplemented("Non-spatial pooling is not "
                                      "yet supported. Volunteers? :)"));

    int64 out_height, out_width, pad_rows, pad_cols;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_rows, window_rows, row_stride,
                                         padding_, &out_height, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(in_cols, window_cols, col_stride,
                                         padding_, &out_width, &pad_cols));

    const T* out_backprop_ptr = out_backprop.flat<T>().data();
    T* input_backprop_ptr = output->flat<T>().data();

    auto shard = [context, out_backprop_ptr, input_backprop_ptr,
                  out_backprop_rows, out_backprop_cols, out_backprop_depth,
                  in_rows, in_cols, window_rows, window_cols, row_stride,
                  col_stride, pad_rows, pad_cols](int64 start, int64 limit) {
      for (int64 b = start; b < limit; ++b) {
        for (int64 r = 0; r < out_backprop_rows; ++r) {
          // Calculates row broadcast size.  For SAME padding, current
          // index could be in the padding area, and r*row_stride +
          // window_rows could be beyond the input tensor's boundary. In
          // such cases, change the starting index and reduce the
          // broadcast size.
          int rindex, rsize;
          OP_REQUIRES_OK(context,
                         GetBroadcastSize(r, in_rows, window_rows, row_stride,
                                          pad_rows, &rindex, &rsize));
          for (int64 c = 0; c < out_backprop_cols; ++c) {
            // Calculates col broadcast size.  For SAME padding, current
            // index could be in the padding area, and c*col_stride +
            // window_cols could be beyond the input tensor's boundary. In
            // such cases, change the starting index and reduce the
            // broadcast size.
            int cindex, csize;
            OP_REQUIRES_OK(context,
                           GetBroadcastSize(c, in_cols, window_cols, col_stride,
                                            pad_cols, &cindex, &csize));

            T divide_coeff(1.0 / (rsize * csize));
            int64 output_index =
                (b * out_backprop_rows + r) * out_backprop_cols + c;
            for (int64 r_dst = rindex; r_dst < rindex + rsize; ++r_dst) {
              for (int64 c_dst = cindex; c_dst < cindex + csize; ++c_dst) {
                int64 input_index = (b * in_rows + r_dst) * in_cols + c_dst;
                const T* output_offset =
                    out_backprop_ptr + output_index * out_backprop_depth;
                T* input_offset =
                    input_backprop_ptr + input_index * out_backprop_depth;
                for (int64 d = 0; d < out_backprop_depth; ++d) {
                  *input_offset += *output_offset * divide_coeff;
                  ++output_offset;
                  ++input_offset;
                }
              }
            }
          }
        }
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost =
        window_rows * window_cols * depth_window * in_rows * in_rows * in_cols;
    Shard(worker_threads.num_threads, worker_threads.workers,
          out_backprop_batch, shard_cost, shard);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

#define REGISTER_CPU_KERNEL(T)                                 \
  REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")                  \
                              .Device(DEVICE_CPU)              \
                              .TypeConstraint<T>("T")          \
                              .HostMemory("orig_input_shape"), \
                          AvgPoolingGradOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
TF_CALL_half(REGISTER_CPU_KERNEL);

#if GOOGLE_CUDA

// A CUDNN based AvgPoolingGrad implementation. It includes the padding as the
// candidates for the pooling operation.
template <class T>
class AvgPoolingGradOp<GPUDevice, T> : public OpKernel {
 public:
  typedef GPUDevice Device;

  explicit AvgPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
        errors::InvalidArgument("out_backprop must be 1-dimensional and 4 "
                                "elements"));
    // For avgpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    DnnPoolingGradOp<T>::Compute(context, se::dnn::PoolingMode::kAverage,
                                 ksize_, stride_, padding_, data_format_,
                                 nullptr, nullptr, out_backprop, output_shape,
                                 /*propagate_nans=*/false);
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("orig_input_shape")
                            .Label("cudnn"),
                        AvgPoolingGradOp<GPUDevice, double>);
REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("orig_input_shape")
                            .Label("cudnn"),
                        AvgPoolingGradOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("orig_input_shape")
                            .Label("cudnn"),
                        AvgPoolingGradOp<GPUDevice, Eigen::half>);

// A custom GPU kernel based AvgPoolingGrad implementation. It includes the
// padding as the candidates for the pooling operation.
template <class T>
class AvgPoolingGradOpCustomGPUKernel : public OpKernel {
 public:
  typedef GPUDevice Device;

  explicit AvgPoolingGradOpCustomGPUKernel(OpKernelConstruction* context)
      : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    const int32 ksize_n = GetTensorDim(ksize_, data_format_, 'N');
    const int32 stride_n = GetTensorDim(stride_, data_format_, 'N');
    OP_REQUIRES(context, ksize_n == 1 && stride_n == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
        errors::InvalidArgument("out_backprop must be 1-dimensional and 4 "
                                "elements"));
    // For avgpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));
    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    if (data_format_ == FORMAT_NHWC) {
      const int64 out_backprop_batch = out_backprop.dim_size(0);
      const int64 out_backprop_rows = out_backprop.dim_size(1);
      const int64 out_backprop_cols = out_backprop.dim_size(2);
      const int64 out_backprop_depth = out_backprop.dim_size(3);

      const int64 in_rows = output_shape.dim_size(1);
      const int64 in_cols = output_shape.dim_size(2);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));

      const int window_rows = ksize_[1];
      const int window_cols = ksize_[2];
      const int depth_window = ksize_[3];

      const int row_stride = stride_[1];
      const int col_stride = stride_[2];

      // We (will) use different code for spatial pooling and
      // non-spatial pooling.
      //
      // Spatial pooling is when depth_window = 1
      OP_REQUIRES(context, depth_window == 1,
                  errors::Unimplemented("Non-spatial pooling is not "
                                        "yet supported. Volunteers? :)"));

      int64 out_height, out_width, pad_rows, pad_cols;
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(in_rows, window_rows, row_stride,
                                           padding_, &out_height, &pad_rows));
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(in_cols, window_cols, col_stride,
                                           padding_, &out_width, &pad_cols));

      RunAvePoolBackwardNHWC<T>(out_backprop.flat<T>().data(),  // top_diff
                                out_backprop_batch,             // num
                                in_rows,                        // height
                                in_cols,                        // width
                                out_backprop_depth,             // channels
                                out_backprop_rows,              // pooled_height
                                out_backprop_cols,              // pooled_width
                                window_rows,                    // kernel_h
                                window_cols,                    // kernel_w
                                row_stride,                     // stride_h
                                col_stride,                     // stride_w
                                pad_rows,                       // pad_t
                                pad_cols,                       // pad_l
                                output->flat<T>().data(),       // bottom_diff
                                context->eigen_gpu_device());   // d
    } else {
      DnnPoolingGradOp<T>::Compute(context, se::dnn::PoolingMode::kAverage,
                                   ksize_, stride_, padding_, data_format_,
                                   nullptr, nullptr, out_backprop, output_shape,
                                   /*propagate_nans=*/false);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("orig_input_shape"),
                        AvgPoolingGradOpCustomGPUKernel<float>);
REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("orig_input_shape"),
                        AvgPoolingGradOpCustomGPUKernel<double>);
REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .HostMemory("orig_input_shape"),
                        AvgPoolingGradOpCustomGPUKernel<Eigen::half>);

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
// AvgPoolGrad SYCL kernel. Expects the number of threads to be equal to the
// number of elements in the output backprop tensor, i.e. the number of
// elements in the input tensor.
//
// For each output backprop index find a window in the input backprop tensor
// which corresponds to all the values of the output which were affected by the
// input value at this index. Then for each gradient in this window, compute
// the size of the input window which was averaged to give this output, and use
// this size to scale the gradient accordingly. Add this scaled gradient to the
// output backprop value.
template <typename T>
class AvgPoolGradSYCL {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;
  public:
  AvgPoolGradSYCL(const int depth, const int batch, const int in_rows,
                  const int in_cols, const std::array<int64, 2>& out_shape,
                  const std::array<int64, 2>& window,
                  const std::array<int64, 2>& stride,
                  const std::array<int64, 2>& padding,
                  const read_accessor input_backprop_accessor,
                  write_accessor output_backprop_accessor)
      : p_(depth, batch, in_rows, in_cols, out_shape, window, stride, padding),
        input_backprop_accessor_(input_backprop_accessor),
        output_backprop_accessor_(output_backprop_accessor) {}
  void operator()(cl::sycl::item<1> item) {
    T* input_backprop = ConvertToActualTypeSycl(T, input_backprop_accessor_);
    T* output_backprop = ConvertToActualTypeSycl(T, output_backprop_accessor_);
     const int index = item.get_linear_id();
    int n = index;
    const int d = n % p_.depth_;
    n /= p_.depth_;
    const int c = (n % p_.in_cols_) + p_.pad_cols_;
    const int poolcstart =
        (c < p_.window_cols_) ? 0 : (c - p_.window_cols_) / p_.stride_cols_ + 1;
    const int poolcend = std::min(c / p_.stride_cols_ + 1, p_.out_cols_);
    n /= p_.in_cols_;
    const int r = (n % p_.in_rows_) + p_.pad_rows_;
    const int poolrstart =
        (r < p_.window_rows_) ? 0 : (r - p_.window_rows_) / p_.stride_rows_ + 1;
    const int poolrend = std::min(r / p_.stride_rows_ + 1, p_.out_rows_);
    n /= p_.in_rows_;
     T gradient = T(0);
    const T* input_backprop_n =
        input_backprop + n * p_.out_cols_ * p_.out_rows_ * p_.depth_;
    for (int poolr = poolrstart; poolr < poolrend; ++poolr) {
      int rstart = poolr * p_.stride_rows_ - p_.pad_rows_;
      const int rend = std::min(rstart + p_.window_rows_, p_.in_rows_);
      rstart = std::max(rstart, 0);
      const int row_window_size = rend - rstart;
      for (int poolc = poolcstart; poolc < poolcend; ++poolc) {
        const int idx = (poolr * p_.out_cols_ + poolc) * p_.depth_ + d;
        int cstart = poolc * p_.stride_cols_ - p_.pad_cols_;
        const int cend = std::min(cstart + p_.window_cols_, p_.in_cols_);
        cstart = std::max(cstart, 0);
        const int col_window_size = cend - cstart;
        const int window_size = row_window_size * col_window_size;
        gradient += input_backprop_n[idx] / static_cast<T>(window_size);
      }
    }
    output_backprop[index] = gradient;
  }
  private:
  const SYCL2DPoolParams p_;
  const read_accessor input_backprop_accessor_;
  write_accessor output_backprop_accessor_;
};
template <typename T>
struct LaunchAvgPoolingGradOpSYCL {
  static void launch(OpKernelContext* context,
                     const TensorShape& tensor_in_shape,
                     const Tensor& out_backprop,
                     const std::array<int64, 2>& window,
                     const std::array<int64, 2>& stride,
                     const std::array<int64, 2>& output_shape,
                     const std::array<int64, 2>& padding,
                     TensorFormat data_format, Tensor* output) {
    const SYCLDevice& device = context->eigen_device<SYCLDevice>();
    const int batch = GetTensorDim(tensor_in_shape, data_format, 'N');
    const int in_rows = GetTensorDim(tensor_in_shape, data_format, '0');
    const int in_cols = GetTensorDim(tensor_in_shape, data_format, '1');
    const int depth = GetTensorDim(tensor_in_shape, data_format, 'C');
     const int num_threads = output->NumElements();
     auto input_backprop_buffer =
        device.get_sycl_buffer(out_backprop.template flat<T>().data());
    auto output_backprop_buffer =
        device.get_sycl_buffer(output->template flat<T>().data());
     device.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto input_backprop_access =
          input_backprop_buffer
              .template get_access<cl::sycl::access::mode::read>(cgh);
      auto output_backprop_access =
          output_backprop_buffer
              .template get_access<cl::sycl::access::mode::write>(cgh);
      AvgPoolGradSYCL<T> avgpoolgrad(
          depth, batch, in_rows, in_cols, output_shape, window, stride, padding,
          input_backprop_access, output_backprop_access);
       cgh.parallel_for(cl::sycl::range<1>(num_threads), avgpoolgrad);
    });
  }
};

template <class T>
class AvgPoolingGradOp<SYCLDevice, T> : public OpKernel {
 public:
  explicit AvgPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Default AvgPoolingGradOp only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(context, ksize_.size() == 4,
                errors::InvalidArgument("Sliding window ksize field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES(context, ksize_[0] == 1 && stride_[0] == 1,
                errors::Unimplemented(
                    "Pooling is not yet supported on the batch dimension."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& tensor_in_shape = context->input(0);
    const Tensor& out_backprop = context->input(1);
    // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
    OP_REQUIRES(
        context,
        tensor_in_shape.dims() == 1 && tensor_in_shape.NumElements() == 4,
        errors::InvalidArgument("out_backprop must be 1-dimensional and 4 "
                                "elements"));
    // For avgpooling, out_backprop should have 4 dimensions.
    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    TensorShape output_shape;
    auto shape_vec = tensor_in_shape.vec<int32>();
    for (int64 i = 0; i < tensor_in_shape.NumElements(); ++i) {
      output_shape.AddDim(shape_vec(i));
    }

    PoolParameters params{context,  ksize_,       stride_,
                          padding_, data_format_, output_shape};
    namespace sd = sycldnn::pooling;
    sd::PoolingParams sd_params = get_sd_params(params);
    if (!context->status().ok()) {
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // This is not an error in TensorFlow, the context expect an empty output
    // in this case.
    if (sd_params.batch == 0)
      return;

    if (!is_snn_enabled()) {
       // Dimension order for these arrays is x, y, z.
      std::array<int64, 2> input_size{
          {GetTensorDim(output_shape, data_format_, '1'),
           GetTensorDim(output_shape, data_format_, '0')}};
      std::array<int64, 2> window{{GetTensorDim(ksize_, data_format_, '1'),
                                   GetTensorDim(ksize_, data_format_, '0')}};
      std::array<int64, 2> stride{{GetTensorDim(stride_, data_format_, '1'),
                                   GetTensorDim(stride_, data_format_, '0')}};
      std::array<int64, 2> out, padding;
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(input_size[0], window[0], stride[0],
                                           padding_, &out[0], &padding[0]));
      OP_REQUIRES_OK(context,
                     GetWindowedOutputSize(input_size[1], window[1], stride[1],
                                           padding_, &out[1], &padding[1]));
      LaunchAvgPoolingGradOpSYCL<T>::launch(context, output_shape, out_backprop,
                                            window, stride, out, padding,
                                            data_format_, output);
    }
    else {
      auto device = context->eigen_device<SYCLDevice>();
      sycldnn::backend::EigenBackend backend(device);
      auto in_ptr = out_backprop.template flat<T>().data();
      auto out_ptr = output->template flat<T>().data();
      auto status = sd::launch<T, sd::Average, sd::Backpropagate>(in_ptr, out_ptr,
          sd_params, backend);
      if (status.status != sycldnn::StatusCode::OK) {
        context->SetStatus(get_sd_err_msg(status));
        return;
      }
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
};

#define REGISTER_AVGPOOLGRAD_SYCL(type)                                   \
REGISTER_KERNEL_BUILDER(Name("AvgPoolGrad")                               \
                            .Device(DEVICE_SYCL)                          \
                            .TypeConstraint<type>("T")                    \
                            .HostMemory("orig_input_shape"),              \
                        AvgPoolingGradOp<SYCLDevice, type>);
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_AVGPOOLGRAD_SYCL);
#undef REGISTER_AVGPOOLGRAD_SYCL
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
