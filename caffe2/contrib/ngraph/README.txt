

New type of network was added and registered - named "ngraph" represented by class NgraphNet.
This change is quite non-intrusive - the rest of the caffe2 was not modified.
We can declare that we want to use this type by adding one line using python-level interface:

    net.Proto().type = 'ngraph'

^This will cause creating NgraphNet object instead of default SimpleNet or alternative DAGNet.

Caffe2 will call NgraphNet::Run() method to execute the graph, but we can use this method to hand-off the graph to ngraph instead of actually executing the computations.

Here in this prototype I simply output the network to a prototxt file (using ProtoDebugString for convenience).
This file could be parsed by existing (currently developed) high-graph front-end.

The final API between caffe2 and ngraph would also require providing pointers to buffers from Caffe2 to ngraph (weights, data, other parameters).

There does not seem to be much difference between high-graph and low-graph approach.
Actually, retrieving all the network parameters with both "train_net" and "init_net" and possibly other Caffe2 workspace parameters could be more troublesome in the low-graph approach.

Caffe2 doesn't seem to be doing much optimization if any at all. There is only (as far as I noticed) parsing of the network in DAGNet that splits the network into "chains" that can run independently in parallel.

