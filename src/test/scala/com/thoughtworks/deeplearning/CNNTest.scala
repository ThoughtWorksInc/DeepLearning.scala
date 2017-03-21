package com.thoughtworks.deeplearning
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.enableMembersIf
import org.scalatest.{FreeSpec, Matchers}
import shapeless._
import com.thoughtworks.deeplearning.Poly._
import com.thoughtworks.deeplearning.Poly.MathMethods.{*, /}
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Symbolic.Layers._

@enableMembersIf(!scala.util.Properties.versionNumberString.startsWith("2.12."))
final class CNNTest extends FreeSpec with Matchers {

  import org.nd4j.linalg.api.ndarray.INDArray
  import org.nd4s.Implicits._
  import org.nd4j.linalg.api.ndarray.INDArray
  import com.thoughtworks.deeplearning.DifferentiableHList._
  import com.thoughtworks.deeplearning.DifferentiableDouble._
  import com.thoughtworks.deeplearning.DifferentiableINDArray._
  import com.thoughtworks.deeplearning.DifferentiableAny._
  import com.thoughtworks.deeplearning.DifferentiableInt._
  import com.thoughtworks.deeplearning.DifferentiableSeq._
  import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._

  "fix view error" in {
    def assertClear(layer: Any): Unit = {
      layer match {
        case cached: CumulativeLayer =>
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

    def convolutionThenRelu(implicit input: INDArray @Symbolic): INDArray @Symbolic = {
      -input
    }

    def convolutionThenRelu2(implicit input: INDArray @Symbolic): INDArray @Symbolic = {

      val imageCount = input.shape(0)

      input.reshape(imageCount + 0, 1.toLayer)
    }

    def hiddenLayer(implicit input: INDArray @Symbolic): INDArray @Symbolic = {
      val layer1 = convolutionThenRelu2.compose(convolutionThenRelu)

      layer1
    }

    val predictor = hiddenLayer

    print(predictor)

    assertClear(predictor)
    predictor.predict((1 to 1).toNDArray.reshape(1, 1))
    assertClear(predictor)
  }
}
