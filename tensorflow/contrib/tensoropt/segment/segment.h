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

#ifndef TENSORFLOW_CONTRIB_TENSOROPT_SEGMENT_SEGMENT_H_
#define TENSORFLOW_CONTRIB_TENSOROPT_SEGMENT_SEGMENT_H_

#include <set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensoropt {
namespace segment {

// vector of segments, each entry contains a device name and a set of nodes in
// segment
using SegmentNodesVector = std::vector<std::pair<std::set<string>, string>>;

// Check if a node has a property
using NodeTagFunction = std::function<bool(const tensorflow::Node*)>;

struct SegmentOptions {
  // Segment must contain at least this many nodes.
  int minimum_segment_size = 2;
  std::set<string> exclude_node_list;
};

// Get the subgraphs of a graph that can be handled by TensorOpt.
//
// @param gdef The GraphDef describing the network
// @param candidate_fn A function that returns true for a NodeDef if
// that node can be handled by TensorOpt.
// @param segments Returns the TensorOpt segments/subgraphs. Each entry
// in the vector describes a subgraph by giving a set of the names of
// all the NodeDefs in that subgraph.
// @return the status.
tensorflow::Status SegmentGraph(
    const tensorflow::GraphDef& gdef, const NodeTagFunction& is_candidate_fn,
    const SegmentOptions& options, SegmentNodesVector* segments,
    const NodeTagFunction& is_mandatory_fn =
        [](const tensorflow::Node*) { return false; },
    const NodeTagFunction& is_weak_fn =
        [](const tensorflow::Node*) { return false; });

// Get the subgraphs of a graph that can be handled by TensorOpt.
//
// @param graph tensorflow::Graph of the network
// @param candidate_fn A function that returns true for a Node* if
// that node can be handled by TensorOpt.
// @param segments Returns the TensorOpt segments/subgraphs. Each entry
// in the vector describes a subgraph by giving a set of the names of
// all the NodeDefs in that subgraph.
// @return the status.
tensorflow::Status SegmentGraph(
    tensorflow::Graph* tf_graph,
    const std::function<bool(const tensorflow::Node*)>& candidate_fn,
    const SegmentOptions& options, SegmentNodesVector* segments,
    const NodeTagFunction& is_mandatory_fn =
        [](const tensorflow::Node*) { return false; },
    const NodeTagFunction& is_weak_fn =
        [](const tensorflow::Node*) { return false; });

}  // namespace segment
}  // namespace tensoropt
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOROPT_SEGMENT_SEGMENT_H_
