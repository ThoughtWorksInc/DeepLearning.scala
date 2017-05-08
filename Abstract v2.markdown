Most of current deep learning frameworks are static. The structure of neural networks in those frameworks must be determined before running. In this talk, we will present the design and implementation tips of the dynamic neural network feature in [DeepLearning.scala](https://github.com/ThoughtWorksInc/DeepLearning.scala/) 2.0.

DeepLearning.scala is a simple Domain Specific Language(DSL) for creating complex neural networks, which has the following advantages in comparison to TensorFlow:
 1. DeepLearning.scala's DSL represents the process that dynamically creates computational graph nodes, instead of static computational graphs.
 2. Our neural networks are programs. All Scala features, including functions and expressions, are available in the DSL.
 3. The DSL is based on `Monad`s, which are composable, thus a complex layer can be built from atomic operators.
 4. Along with the Monad, we provide an `Applicative` type class, to perform multiple calculations in parallel.

In brief, in DeepLearning.scala 2.0, you can create neural networks in the same way as ordinary Scala code, and the computation in the networks still get scheduled onto GPU and CPU in parallel.
