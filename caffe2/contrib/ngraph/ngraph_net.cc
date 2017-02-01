/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "caffe2/core/net.h"
#include "caffe2/contrib/ngraph/ngraph_net.h"
#include "caffe2/core/operator.h"

#include <iostream>
#include <fstream>

#undef VLOG
#define VLOG(X) \
std::cout<<std::endl

namespace caffe2 {

  // TBD: mostly copy-pased from SimpleNet

NgraphNet::NgraphNet(const NetDef& net_def, Workspace* ws)
    : NetBase(net_def, ws) {
  VLOG(1) << "Constructing NgraphNet " << net_def.name();
  bool net_def_has_device_option = net_def.has_device_option();
  // Initialize the operators
  for (const OperatorDef& operator_def : net_def.op()) {
    VLOG(1) << "Creating operator " << operator_def.name()
            << ":" << operator_def.type();
    if (!operator_def.has_device_option() && net_def_has_device_option) {
      // In the case that the operator def does not specify a device option but
      // the net def has a default option, we copy the device option over to the
      // operator def.
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def.device_option());
      operators_.emplace_back(CreateOperator(temp_def, ws));
    } else {
      operators_.emplace_back(CreateOperator(operator_def, ws));
    }
  }
}

bool NgraphNet::Run() {
  VLOG(1) << "Running net through ngraph" << name_;

  std::ofstream file;
  file.open("net_def.prototxt");

  for (auto& op : operators_) {
    // Note: Instead of running the net, write the graph to a file
     file << "op { " << ProtoDebugString(op->def()) << " }" << std::endl;

    // Don't actually run it here
    /*
    if (!op->Run()) {
      LOG(ERROR) << "Operator failed: "
                      << ProtoDebugString(op->def());
      return false;
    }
    */
  }
  file.close();
  return true;
}

bool NgraphNet::RunAsync() {
  // TODO
  return true;
}

vector<float> NgraphNet::TEST_Benchmark(
    const int warmup_runs,
    const int main_runs,
    const bool run_individual) {

  // TODO: NgraphNet::TEST_Benchmark()

}


  REGISTER_NET(ngraph, NgraphNet);

} // namespace caffe2
