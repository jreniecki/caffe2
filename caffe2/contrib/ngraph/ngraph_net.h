/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ngraph_net.h
 * Author: jrenieck
 *
 * Created on February 1, 2017, 11:39 AM
 */

#ifndef NGRAPH_NET_H
#define NGRAPH_NET_H

#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"

namespace caffe2 {
    
class NgraphNet final : public NetBase {
 public:
  NgraphNet(const NetDef& net_def, Workspace* ws);
  bool Run() override;
  bool RunAsync() override;
  vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) override;

 protected:
  vector<unique_ptr<OperatorBase> > operators_;

  DISABLE_COPY_AND_ASSIGN(NgraphNet);
};


}  // namespace caffe2

#endif /* NGRAPH_NET_H */

