Most of current deep learning frameworks are static. The structure of neural networks in those frameworks must be determined before running. In this talk, we will present the design and implementation tips of the dynamic neural network feature in [DeepLearning.scala](https://github.com/ThoughtWorksInc/DeepLearning.scala/) 2.0.

DeepLearning.scala is a simple Domain Specific Language(DSL) for creating complex neural networks, which has the following advantages in comparison to TensorFlow:
 1. DeepLearning.scala's DSL represents a process that dynamically creates computational graph nodes, instead of a static computational graph.
 2. Our neural networks are programs. All Scala features, including any Scala functions and Scala expressions are available in the DSL.
 3. The DSL is based on `Monad`s, which are composable, thus a complex layer can be built from atomic operators.
 4. Along with Monad, we also provide an `Applicative` type class, which calculate multiple operations in parallel.

In brief, in DeepLearning.scala 2.0, you can create neural networks the same as ordinary Scala code, and the computation in the networks still get scheduled onto GPU and CPU in parallel.
