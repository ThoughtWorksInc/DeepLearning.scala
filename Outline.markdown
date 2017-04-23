1. Introducing neural networks and backpropagation.
1. Introducing the computational graph representation for neural networks, and showing some examples written in TensorFlow.
1. Showing the interoperability issue in TensorFlow's computational graph approach, including: 
   * Impossible to invoking Python functions from TensorFlow
   * Impossible to use Python control flow expressions from TensorFlow
   * Impossible to custom special layer in Python (requires hard-coded CUDA/C++ code)
1. Introducing automatic differentiation, and some examples written in [autograd](https://github.com/HIPS/autograd)
1. Explaining why autograd has better interoperability than TensorFlow.
1. Showing the performance issue in autograd's automatic differentiation approach, including: 
   * Does not support GPU
   * Impossible to run in parallel
1. Introducing our approach: monad-based automatic differentiation, and showing some examples written in DeepLearning.scala.
1. The technology stack of DeepLearning.scala:<br> 
   * [Scala](http://scala-lang.org)
   * [Scalaz](https://github.com/scalaz/scalaz)
   * [ThoughtWorks Each](https://github.com/ThoughtWorksInc/each)
   * [Shapeless](https://github.com/milessabin/shapeless/)
   * [nd4j](https://github.com/deeplearning4j/nd4s)
1. Explain the basis of monad Scalaz.
1. Explain the detail of our approach:<br> 
   * Unlike TensorFlow's static computational graph, our monad represents a process that dynamically creates computational graph nodes.
   * All Scala features, including any Scala functions and Scala expressions, can be used in the process. (Address TensorFlow's issue 1 and 2)
   * `Monad`s are composable, a complex layer can be built from atomic operators. (Address TensorFlow's issue 3)
   * These computation can be performed on arbitrary Scala types, including native Double, Float, custom ADT, and nd4j's GPU-accelerated N-dimensional arrays. (Address autograd's issue 1)
   * Along with Monad, we also provide an `Applicative` type class, which calculate multiple operations in parallel. (Address autograd's issue 2)
   * In addition, our monads manage resource automatically, without depending on garbage collection. As a result, unlike other Lua or JVM frameworks, our framework never leaks memory.
   * We provide a DSL to generate the monadic code. The users still write normal Scala code, our DSL will automatically convert the code to monadic style. All the above features are enabled by default.
1. Run benchmark and show results. DeepLearning.scala should run faster than autograd.
