package com.thoughtworks.deeplearning
//import org.nd4j.linalg.api.ndarray.INDArray
//import org.nd4s.Implicits._
import org.scalatest.{FreeSpec, Matchers}

final class CNNTest extends FreeSpec with Matchers {
  "fix view error" in {
/*    def assertClear(layer: Any): Unit = {
      layer match {
        case cached: BufferedLayer =>
          assert(cached.cache.isEmpty)
        case _ =>
      }
      layer match {
        case parent: Product =>
          for (upstreamLayer <- parent.productIterator) {
            assertClear(upstreamLayer)
          }
        case _ =>
      }
    }

    def convolutionThenRelu(implicit input: From[INDArray]##T): To[INDArray]##T = {
      -input
    }

    def convolutionThenRelu2(implicit input: From[INDArray]##T): To[INDArray]##T = {

      val imageCount = input.shape(0)

      input.reshape(imageCount + 0, 1.toLayer)
    }

    def hiddenLayer(implicit input: From[INDArray]##T): To[INDArray]##T = {
      val layer1 = convolutionThenRelu2.compose(convolutionThenRelu)

      layer1
    }

    val predictor = hiddenLayer

    print(predictor)

    assertClear(predictor)
    predictor.predict((1 to 1).toNDArray.reshape(1, 1))
    assertClear(predictor)*/
  }
}
