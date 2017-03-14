package com.thoughtworks
package deeplearning

import org.scalatest._
import shapeless._
import cats._
import cats.implicits._
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Layer.Tape._
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableNothing._
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._
import com.thoughtworks.deeplearning.DifferentiableBoolean._
import com.thoughtworks.deeplearning.DifferentiableSeq._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableCoproduct._
import com.thoughtworks.deeplearning.DifferentiableAny._

import language.implicitConversions
import language.existentials
import Predef.{any2stringadd => _, _}
import util.Random

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
@enableMembersIf(!scala.util.Properties.versionNumberString.startsWith("2.12."))
final class XorSpec extends FreeSpec with Matchers {

  import XorSpec._
  import com.thoughtworks.deeplearning.DifferentiableINDArray._
  import org.nd4s.Implicits._
  import org.nd4j.linalg.api.ndarray.INDArray
  import org.nd4j.linalg.factory.Nd4j
  import org.nd4j.linalg.ops.transforms.Transforms
  import com.thoughtworks.deeplearning.DifferentiableINDArray.Optimizers.LearningRate
  import org.nd4j.linalg.factory.Nd4j
  import org.nd4s.Implicits._

  implicit val optimizer = new DifferentiableINDArray.Optimizers.L2Regularization
  with LearningRate{
    override protected def currentLearningRate() = 0.006

    override protected def l2Regularization = 0.01
  }

  def fullyConnectedThenRelu(inputSize: Int, outputSize: Int)(implicit row: INDArrayPlaceholder) = {
    val w = (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
    //    val b = (Nd4j.randn(1, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    max((row dot w) + b, 0.0)
  }

  def sigmoid(implicit input: INDArrayPlaceholder) = {
    1.0 / (exp(-input) + 1.0)
  }

  def fullyConnectedThenSigmoid(inputSize: Int, outputSize: Int)(implicit row: INDArrayPlaceholder) = {
    val w = (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight
    //    val b = (Nd4j.randn(1, outputSize) / math.sqrt(outputSize)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    sigmoid.compose((row dot w) + b)
  }

  val ArrayToArray = FromTo[INDArray, INDArray]

  def hiddenLayers(implicit encodedInput: From[INDArray]##`@`): ArrayToArray.`@` = {
    fullyConnectedThenSigmoid(50, 3).compose(
      fullyConnectedThenRelu(50, 50).compose(
        fullyConnectedThenRelu(50, 50).compose(fullyConnectedThenRelu(6, 50).compose(encodedInput))))
  }

  val hiddenLayersNetwork = hiddenLayers

  def encode(implicit input: From[XorSpec.InputData]##`@`): To[INDArray]##`@` = {
    val field0 = input.head
    val rest0 = input.tail
    val field1 = rest0.head
    val rest1 = rest0.tail
    val field2 = rest1.head

    val encoded0 = field0.choice { _ =>
      1.0
    } { _ =>
      0.0
    }
    val encoded1 = field0.choice { _ =>
      0.0
    } {
      _.choice { value =>
        value
      } { _ =>
        `throw`(new IllegalArgumentException)
      }
    }
    val encoded2 = field1.choice { _ =>
      1.0
    } { _ =>
      0.0
    }
    val encoded3 = field1.choice { _ =>
      0.0
    } {
      _.choice { value =>
        value
      } { _ =>
        `throw`(new IllegalArgumentException)
      }
    }
    val encoded4 = field2.choice { _ =>
      1.0
    } { _ =>
      0.0
    }
    val encoded5 = field2.choice { _ =>
      0.0
    } {
      _.choice { value =>
        value
      } { _ =>
        `throw`(new IllegalArgumentException)
      }
    }
    Vector(Vector(encoded0, encoded1, encoded2, encoded3, encoded4, encoded5)).toINDArray
  }

  val encodeNetwork = encode

  def decode(implicit row: From[INDArray]##`@`): To[XorSpec.OutputData]##`@` = {
    val rowSeq = row.toSeq
    rowSeq(0)(0) :: rowSeq(0)(1) :: rowSeq(0)(2) :: shapeless.HNil.toLayer
  }

  val decodeNetwork = decode

  def predict(implicit input: From[XorSpec.InputData]##`@`): To[XorSpec.OutputData]##`@` = {
    decodeNetwork.compose(hiddenLayersNetwork.compose(encodeNetwork.compose(input)))
  }

  val predictNetwork = predict

  def loss(implicit pair: From[ExpectedLabelData :: INDArray :: HNil]##`@`): To[Double]##`@` = {

    val expectedLabel = pair.head
    val expectedField0 = expectedLabel.head
    val expectedRestField0 = expectedLabel.tail
    val expectedField1 = expectedRestField0.head
    val expectedRestField1 = expectedRestField0.tail
    val expectedField2 = expectedRestField1.head
    val expectedRestField2 = expectedRestField1.tail

    val predictionResult = pair.tail.head.toSeq

    val loss0 = expectedField0.choice { _ =>
      0.0
    } {
      _.choice { expectedValue =>
        val value = predictionResult(0)(0)
        -expectedValue * log(value) - (1.0 - expectedValue) * log(1.0 - value)
      } { _ =>
        `throw`(new IllegalArgumentException)
      }
    }

    val loss1 = expectedField1.choice { _ =>
      0.0
    } {
      _.choice { expectedValue =>
        val value = predictionResult(0)(1)
        -expectedValue * log(value) - (1.0 - expectedValue) * log(1.0 - value)
      } { _ =>
        `throw`(new IllegalArgumentException)
      }
    }

    val loss2 = expectedField2.choice { _ =>
      0.0
    } {
      _.choice { expectedValue =>
        val value = predictionResult(0)(2)
        -expectedValue * log(value) - (1.0 - expectedValue) * log(1.0 - value)
      } { _ =>
        `throw`(new IllegalArgumentException)
      }
    }

    loss0 + loss1 + loss2
  }

  def train(implicit pair: From[ExpectedLabelData :: InputData :: HNil]##`@`): To[Double]##`@` = {
    val expectedLabel = pair.head
    val input = pair.tail.head
    loss.compose(expectedLabel :: hiddenLayersNetwork.compose(encodeNetwork.compose(input)) :: shapeless.HNil.toLayer)
  }

  val trainNetwork = train

  def makeTrainingData(): ExpectedLabelData :: InputData :: HNil = {
    import shapeless._
    val field0 = Random.nextBoolean()
    val field1 = Random.nextBoolean()
    val field2 = field0 ^ field1
    val Seq(dropout0, dropout1, dropout2) = Seq.fill(3)(false).updated(Random.nextInt(3), true)
    def input(isDropout: Boolean, value: Boolean) = {
      if (isDropout) {
        Inl(HNil)
      } else {
        Inr(Inl(if (value) {
          1.0
        } else {
          0.0
        }))
      }
    }
    def expectedLabel(isDropout: Boolean, value: Boolean) = {
      if (isDropout) {
        Inr(Inl(if (value) {
          1.0
        } else {
          0.0
        }))
      } else {
        Inl(HNil)
      }
    }
    (expectedLabel(dropout0, field0) :: expectedLabel(dropout1, field1) :: expectedLabel(dropout2, field2) :: HNil) :: (input(
      dropout0,
      field0) :: input(dropout1, field1) :: input(dropout2, field2) :: HNil) :: HNil
  }

  def predictAndPrint() = {
    import shapeless._
    val (left :: result :: right :: HNil) =
      predictNetwork.predict(Inr(Inl(1.0)) :: Inl(HNil) :: Inr(Inl(0.0)) :: HNil)
    val loss =
      trainNetwork.predict((Inl(HNil) :: Inr(Inl(1.0)) :: Inl(HNil) :: HNil) :: (Inr(Inl(1.0)) :: Inl(HNil) :: Inr(
        Inl(0.0)) :: HNil) :: HNil)
    println(raw"""${left.value}^${result.value}=${right.value}
loss: ${loss.value}
""")

    val (left10 :: right10 :: result10 :: HNil) =
      predictNetwork.predict(Inr(Inl(1.0)) :: Inr(Inl(0.0)) :: Inl(HNil) :: HNil)
    val (left01 :: right01 :: result01 :: HNil) =
      predictNetwork.predict(Inr(Inl(0.0)) :: Inr(Inl(1.0)) :: Inl(HNil) :: HNil)
    val (left00 :: right00 :: result00 :: HNil) =
      predictNetwork.predict(Inr(Inl(0.0)) :: Inr(Inl(0.0)) :: Inl(HNil) :: HNil)
    val (left11 :: right11 :: result11 :: HNil) =
      predictNetwork.predict(Inr(Inl(1.0)) :: Inr(Inl(1.0)) :: Inl(HNil) :: HNil)

    println(raw"""${left00.value}^${right00.value}=${result00.value}
${left01.value}^${right01.value}=${result01.value}
${left10.value}^${right10.value}=${result10.value}
${left11.value}^${right11.value}=${result11.value}
""")
  }

  "xor" in {
    for (i <- 0 until 1000) {
//      predictAndPrint()
      trainNetwork.train(makeTrainingData())
    }
//    predictAndPrint();
    {
      import shapeless._
      val (left00 :: right00 :: result00 :: HNil) =
        predictNetwork.predict(Inr(Inl(0.0)) :: Inr(Inl(0.0)) :: Inl(HNil) :: HNil)
      result00 should be < 0.5
      val (left01 :: right01 :: result01 :: HNil) =
        predictNetwork.predict(Inr(Inl(0.0)) :: Inr(Inl(1.0)) :: Inl(HNil) :: HNil)
      result01 should be > 0.5
      val (left10 :: right10 :: result10 :: HNil) =
        predictNetwork.predict(Inr(Inl(1.0)) :: Inr(Inl(0.0)) :: Inl(HNil) :: HNil)
      result10 should be > 0.5
      val (left11 :: right11 :: result11 :: HNil) =
        predictNetwork.predict(Inr(Inl(1.0)) :: Inr(Inl(1.0)) :: Inl(HNil) :: HNil)
      result11 should be < 0.5

      val (_ :: result0x0 :: _ :: HNil) =
        predictNetwork.predict(Inr(Inl(0.0)) :: Inl(HNil) :: Inr(Inl(0.0)) :: HNil)
      result0x0 should be < 0.5

      val (resultx11 :: _ :: _ :: HNil) =
        predictNetwork.predict(Inl(HNil) :: Inr(Inl(1.0)) :: Inr(Inl(1.0)) :: HNil)
      resultx11 should be < 0.5

      val (_ :: result1x0 :: _ :: HNil) =
        predictNetwork.predict(Inr(Inl(1.0)) :: Inl(HNil) :: Inr(Inl(0.0)) :: HNil)
      result1x0 should be > 0.5
    }
  }

}

@enableMembersIf(!scala.util.Properties.versionNumberString.startsWith("2.12."))
object XorSpec {
  type OptionalDoubleData = HNil :+: Double :+: CNil
  type ExpectedLabelData = OptionalDoubleData :: OptionalDoubleData :: OptionalDoubleData :: HNil
  type InputData = OptionalDoubleData :: OptionalDoubleData :: OptionalDoubleData :: HNil
  type OutputData = Double :: Double :: Double :: HNil
}
