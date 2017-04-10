# Monadic Deep Learning

---

## Abstract

### Computational graph approach

Most of deep learning frameworks([1], [2]) think neural networks as computational graphs. A node in the graph represents a mathematical operation, a function, or even a control flow operation.

However, neural networks built from those frameworks interoperate badly between their hosting language. For example, a neural network written in Python is not able to use any native Python functions, data structures, nor control flows, during the network running.

### Automatic differentiation approach

Recent studies reveal that neural networks and programs are isomorphism([3]). Some other libraries treat neural networks as ordinary programs with ability of automatic differentiation([4], [5]). Ordinary programming control flow operations are allowed in neural networks.

Unfortunately, those libraries have bad performance. They cannot perform multiple calculations parallelly, nor enqueue multiple commands to CUDA streams. Because the programs are directly written by users, those libraries have no room to optimize the computation process.

### Our approach

In DeepLearning.scala, we introduce a new approach that treats neural networks as `Monad`s. Users create neural networks in a way almost the same as ordinary programs, and all Scala language features are available in neural networks. At the mean time, the DeepLearning.scala runtime is still able to schedule computation onto GPU and CPU parallelly.

In addition, our monads manage resource automatically, without depending on garbage collection. As a result, unlike other Lua or JVM frameworks, our framework never leaks memory.

[1]: http://tensorflow.org/
[2]: http://deeplearning.net/software/theano/
[3]: https://colah.github.io/posts/2015-09-NN-Types-FP/
[4]: https://github.com/HIPS/autograd
[5]: https://github.com/twitter/torch-autograd
