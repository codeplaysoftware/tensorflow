/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_METADATA_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_METADATA_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace xla {

// Cannot include hlo_instruction.h as this file is included from there.
class HloInstruction;

// The DomainMetadata represents the base class for metadata which can be
// attached to kDomain HLO instructions.
class DomainMetadata {
 public:
  // A Domain data structure captures all the information about a kDomain
  // bounded instruction set.
  struct Domain {
    // The set of instructions which are reachable from each other via
    // operand/user pathways, without crossing a kDomain instruction of a given
    // kind. The reach_set can contain kDomain instructions of other kinds, if
    // two domains of different kind intersect each other.
    tensorflow::gtl::FlatSet<HloInstruction*> reach_set;

    // The same instructions in reach_set, but purged from kDomain instructions.
    std::vector<HloInstruction*> instructions;

    // If we consider a graph edge as an arrow oriented from the operand to the
    // user, the enter_domains will contain the set of kDomain instructions
    // whose dataflow enters the reach set (domain), while the exit_domains
    // contains the set of kDomain instructions whose dataflow exit the reach
    // set.
    tensorflow::gtl::FlatSet<HloInstruction*> enter_domains;
    tensorflow::gtl::FlatSet<HloInstruction*> exit_domains;
  };

  virtual ~DomainMetadata() = default;

  // Clones the metadata object.
  virtual std::unique_ptr<DomainMetadata> Clone() const = 0;

  // Returns the metadata type. A unique identifier which describes the real
  // metadata type.
  virtual tensorflow::StringPiece Kind() const = 0;

  // Compares the metadata object with another one and returns true if the
  // two matches.
  virtual bool Matches(const DomainMetadata& other) const = 0;

  // Returns a string representation of the metadata.
  virtual string ToString() const = 0;

  // Given a reachable set (the set of instructions which are reachable from
  // each other via user/operand pathways, without crossing a kDomain
  // instruciton), makes sure that all of them have metadata attributes which
  // are coherent with this metadata object.
  virtual Status NormalizeInstructions(const Domain& domain) const = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_DOMAIN_METADATA_H_
