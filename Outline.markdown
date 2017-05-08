1. Introducing neural networks and backpropagation.
2. Introducing the computational graph representation for neural networks, and showing some examples written in other deep learning framework.
3. Showing the interoperability issue in other frameworks's computational graph approach, including: 
   * Impossible to invoke hosting language's functions from neural networks
   * Impossible to use hosting language's control flow expressions from neural networks
   * Impossible to custom special layer in hosting language (requires hard-coded CUDA/C++ code)
4. Introducing automatic differentiation based on dual number
5. Explaining why dual number approach has better hosting language interoperability than computational graph.
6. Showing the performance issue in naive dual number approach, including: 
   * Does not support GPU
   * Impossible to run in parallel
7. Introducing our approach: monad-based automatic differentiation, and showing some examples written in DeepLearning.scala.
8. The technology stack of DeepLearning.scala
9. Explain the basis of monad.
10. Explain the detail of our approach:
   * Unlike TensorFlow's static computational graph, our monad represents a process that dynamically creates computational graph nodes.
   * All Scala features, including any Scala functions and Scala expressions, can be used in the process. (Address TensorFlow's issue 1 and 2)
   * `Monad`s are composable, a complex layer can be built from atomic operators. (Address TensorFlow's issue 3)
   * These computation can be performed on arbitrary Scala types, including native Double, Float, custom ADT, and nd4j's GPU-accelerated N-dimensional arrays. (Address autograd's issue 1)
   * Along with Monad, we also provide an `Applicative` type class, which calculate multiple operations in parallel. (Address autograd's issue 2)
   * In addition, our monads manage resource automatically, without depending on garbage collection. As a result, unlike other Lua or JVM frameworks, our framework never leaks memory.
   * We provide a DSL to generate the monadic code. The users still write normal Scala code, our DSL will automatically convert the code to monadic style. All the above features are enabled by default.
11. Run benchmark and show results. DeepLearning.scala should run faster than naive dual number approach.
