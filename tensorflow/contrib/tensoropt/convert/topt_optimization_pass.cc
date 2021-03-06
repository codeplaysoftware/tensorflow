/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
   Copyright (C) Codeplay Software Limited

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

#include "tensorflow/contrib/tensoropt/convert/topt_optimization_pass.h"
#include "tensorflow/contrib/tensoropt/convert/convert_graph.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
namespace tensorflow {
namespace tensoropt {
namespace convert {
using tensorflow::str_util::Uppercase;
using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;

tensorflow::Status TOPTOptimizationPass::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  VLOG(1) << "Called INIT for " << name_ << " with config = " << config;
  if (config == nullptr) {
    maximum_workspace_size_ = 2 << 30;
    return tensorflow::Status::OK();
  }
  const auto params = config->parameter_map();
  if (params.count("minimum_segment_size")) {
    minimum_segment_size_ = params.at("minimum_segment_size").i();
  }
  if (params.count("max_batch_size")) {
    maximum_batch_size_ = params.at("max_batch_size").i();
  }
  if (params.count("max_workspace_size_bytes"))
    maximum_workspace_size_ = params.at("max_workspace_size_bytes").i();
  if (params.count("precision_mode")) {
    string pm = Uppercase(params.at("precision_mode").s());
    if (pm == "FP32") {
      precision_mode_ = 0;
    } else if (pm == "FP16") {
      precision_mode_ = 1;
    } else if (pm == "INT8") {
      precision_mode_ = 2;
    } else {
      LOG(ERROR) << "Unknown precision mode '" << pm << "'";
      return tensorflow::errors::InvalidArgument(
          "Unknown precision mode argument" + pm +
          " Valid values are FP32, FP16, INT8");
    }
  }
  return tensorflow::Status::OK();
}

void TOPTOptimizationPass::PrintDebugInfo(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item) {
  VLOG(1) << "Cluster = " << cluster;
  string offset("  ");
  string offset2 = StrCat(offset, offset);
  string offset3 = StrCat(offset2, offset);
  string offset4 = StrCat(offset2, offset2);
  if (cluster) {
    VLOG(1) << offset << "type             = " << cluster->type();
    VLOG(1) << offset << "num warmup steps = " << cluster->NumWarmupSteps();
    const auto dev_names = cluster->GetDeviceNames();
    if (dev_names.size()) {
      VLOG(1) << offset << " Device names:";
      for (const auto s : dev_names) {
        VLOG(1) << offset2 << s;
      }
    }
    std::unordered_map<string, uint64> peak_mem;
    auto status = cluster->GetPeakMemoryUsage(&peak_mem);
    if (status == tensorflow::Status::OK()) {
      VLOG(1) << offset << "Peak Memory Usage :";
      for (auto s : peak_mem) {
        VLOG(1) << offset2 << s.first << " = " << s.second;
      }
    }

    const auto dev_props = cluster->GetDevices();
    if (dev_props.size()) {
      VLOG(1) << offset << "Device properties:";
      for (auto k : dev_props) {
        VLOG(1) << offset2 << k.first;
        const auto& dt = k.second;
        VLOG(1) << offset3 << "type          = " << dt.type();
        VLOG(1) << offset3 << "vendor        = " << dt.vendor();
        VLOG(1) << offset3 << "model         = " << dt.model();
        VLOG(1) << offset3 << "frequency     = " << dt.frequency();
        VLOG(1) << offset3 << "num cores     = " << dt.num_cores();
        VLOG(1) << offset3 << "num registers = " << dt.num_registers();
        VLOG(1) << offset3 << "L1 cache size = " << dt.l1_cache_size();
        VLOG(1) << offset3 << "L2 cache size = " << dt.l2_cache_size();
        VLOG(1) << offset3 << "L3 cache size = " << dt.l3_cache_size();
        VLOG(1) << offset3 << "SHMem per SMP = "
                << dt.shared_memory_size_per_multiprocessor();
        VLOG(1) << offset3 << "memory size   = " << dt.memory_size();
        VLOG(1) << offset3 << "bandwidth     = " << dt.bandwidth();
        if (dt.environment_size()) {
          VLOG(1) << offset3 << "environment   :";
          for (const auto e : dt.environment()) {
            VLOG(1) << offset4 << e.first << " = " << e.second;
          }
        }
      }
    }
  }
  VLOG(1) << "item: " << item.id;
  if (item.feed.size()) {
    VLOG(1) << offset << "Feeds  :";
    for (const auto& f : item.feed) {
      const auto& shape = f.second.shape();
      VLOG(1) << offset2 << f.first << " = shaped " << shape.DebugString();
    }
  } else {
    VLOG(1) << offset << "No Feeds";
  }
  if (item.fetch.size()) {
    VLOG(1) << offset << "Fetches  :";
    for (const auto& f : item.fetch) {
      VLOG(1) << offset2 << f;
    }
  } else {
    VLOG(1) << offset << "No Fetches";
  }

  if (item.init_ops.size()) {
    VLOG(1) << offset << "init ops  :";
    for (const auto& f : item.init_ops) {
      VLOG(1) << offset2 << f;
    }
  } else {
    VLOG(1) << offset << "No init ops";
  }
  VLOG(1) << "Save Op = " << item.save_op;
  VLOG(1) << "Restore Op = " << item.restore_op;
  VLOG(1) << "save_restore_loc_tensor = " << item.save_restore_loc_tensor;
  if (item.keep_ops.size()) {
    VLOG(1) << offset << "keep ops  :";
    for (const auto& f : item.keep_ops) {
      VLOG(1) << offset2 << f;
    }
  } else {
    VLOG(1) << offset << "No keep ops";
  }
  VLOG(3) << item.graph.DebugString();
  for (const auto dev : cluster->GetDeviceSet()->devices()) {
    const auto& pname = dev->parsed_name();
    VLOG(1) << "Device name= " << dev->name()
            << " parsedname job= " << pname.job << " id= " << pname.id
            << " has_id: " << pname.has_id << " has_job: " << pname.has_job
            << "has_type: " << pname.has_type << " type =" << pname.type;
  }
}

tensorflow::Status TOPTOptimizationPass::Optimize(
    tensorflow::grappler::Cluster* cluster,
    const tensorflow::grappler::GrapplerItem& item, GraphDef* optimized_graph) {
  VLOG(1) << "Called TOPTOptimization Pass " << name_;
  if (VLOG_IS_ON(1)) {
    PrintDebugInfo(cluster, item);
  }
  auto status = tensorflow::tensoropt::convert::ConvertGraphDefToTensorOpt(
      item.graph, item.fetch, optimized_graph, minimum_segment_size_, cluster);
  VLOG(2) << optimized_graph->DebugString();
  return status;
}

void TOPTOptimizationPass::Feedback(tensorflow::grappler::Cluster*,
                                    const tensorflow::grappler::GrapplerItem&,
                                    const GraphDef&, double) {}

}  // namespace convert
}  // namespace tensoropt
}  // namespace tensorflow

class VerboseCustomGraphOptimizerRegistrar
    : public tensorflow::grappler::CustomGraphOptimizerRegistrar {
 public:
  VerboseCustomGraphOptimizerRegistrar(
      const tensorflow::grappler::CustomGraphOptimizerRegistry::Creator& cr,
      const tensorflow::string& name)
      : tensorflow::grappler::CustomGraphOptimizerRegistrar(cr, name) {
    VLOG(1) << "Constructing a CustomOptimizationPass registration object for "
            << name;
  }
};

static VerboseCustomGraphOptimizerRegistrar TOPTOptimizationPass_Registrar(
    []() {
      VLOG(1) << "Instantiating CustomOptimizationPass object TOPTOptimizer";
      return new tensorflow::tensoropt::convert::TOPTOptimizationPass(
          "TOPTOptimizer");
    },
    ("TOPTOptimizer"));

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
