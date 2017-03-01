package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.BufferedLayer
import com.thoughtworks.deeplearning.Lift.{From, To}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FreeSpec, Matchers}
import shapeless._

import com.thoughtworks.deeplearning
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableInt._
import com.thoughtworks.deeplearning.DifferentiableSeq._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._
import com.thoughtworks.deeplearning._
import com.thoughtworks.deeplearning.Layer.Batch
import com.thoughtworks.deeplearning.Lift.Layers.Identity
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly.MathMethods.{*, /}
import com.thoughtworks.deeplearning.Poly.MathOps
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.PadMode
import org.nd4j.linalg.factory.Nd4j.PadMode.EDGE
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._
import shapeless._
import shapeless._
import shapeless.OpticDefns.compose

import scala.annotation.tailrec
import scala.collection.immutable.IndexedSeq
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableINDArray._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers._
import com.thoughtworks.deeplearning.Lift._
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.Poly.MathOps
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import shapeless._

final class CNNTest extends FreeSpec with Matchers {
  "fix view error" in {

    def assertClear(layer: Any): Unit = {
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
    assertClear(predictor)
  }
}
