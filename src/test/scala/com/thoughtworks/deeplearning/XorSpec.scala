package com.thoughtworks.deeplearning

import cats._
import cats.implicits._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.scalatest.{FreeSpec, Matchers}
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Batch._
import com.thoughtworks.deeplearning.hlist._
import com.thoughtworks.deeplearning.boolean._
import com.thoughtworks.deeplearning.seq._
import com.thoughtworks.deeplearning.double._
import com.thoughtworks.deeplearning.array2D._
import com.thoughtworks.deeplearning.dsl._
import com.thoughtworks.deeplearning.dsl.layers.{Identity, Literal}
import com.thoughtworks.deeplearning.array2D.optimizers.LearningRate
import com.thoughtworks.deeplearning.coproduct._
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest._

import scala.language.implicitConversions
import scala.language.existentials
import Predef.{any2stringadd => _, _}
import scala.util.Random
import com.thoughtworks.{deeplearning => dl}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class XorSpec extends FreeSpec with Matchers {

  import XorSpec._

  implicit val optimizer = new array2D.optimizers.L2Regularization with double.optimizers.L2Regularization {
    override protected def currentLearningRate() = 0.006

    override protected def l2Regularization = 0.01
  }

  def fullyConnectedThenRelu(inputSize: Int, outputSize: Int)(implicit row: Array2D) = {
    val w = (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
    //    val b = (Nd4j.randn(1, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    max((row dot w) + b, 0.0)
  }

  def sigmoid(implicit input: Array2D) = {
    1.0 / (exp(-input) + 1.0)
  }

  def fullyConnectedThenSigmoid(inputSize: Int, outputSize: Int)(implicit row: Array2D) = {
    val w = (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize)).toWeight
    //    val b = (Nd4j.randn(1, outputSize) / math.sqrt(outputSize)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    sigmoid.compose((row dot w) + b)
  }

  def hiddenLayers(implicit encodedInput: Array2D): encodedInput.To[Array2D] = {
    fullyConnectedThenSigmoid(50, 3).compose(
      fullyConnectedThenRelu(50, 50).compose(
        fullyConnectedThenRelu(50, 50).compose(fullyConnectedThenRelu(6, 50).compose(encodedInput))))
  }

  val hiddenLayersNetwork = hiddenLayers

  def encode(implicit input: XorSpec.Input): input.To[Array2D] = {
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
    Vector(Vector(encoded0, encoded1, encoded2, encoded3, encoded4, encoded5)).toArray2D
  }

  val encodeNetwork = encode

  def decode(implicit row: Array2D): row.To[XorSpec.Output] = {
    val rowSeq = row.toSeq
    rowSeq(0)(0) :: rowSeq(0)(1) :: rowSeq(0)(2) :: HNil
  }

  val decodeNetwork = decode

  def predict(implicit input: XorSpec.Input): input.To[XorSpec.Output] = {
    decodeNetwork.compose(hiddenLayersNetwork.compose(encodeNetwork.compose(input)))
  }

  val predictNetwork = predict

  def loss(implicit pair: ExpectedLabel :: Array2D :: HNil): pair.To[Double] = {

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

  def train(implicit pair: ExpectedLabel :: Input :: HNil): pair.To[Double] = {
    val expectedLabel = pair.head
    val input = pair.tail.head
    loss.compose(expectedLabel :: hiddenLayersNetwork.compose(encodeNetwork.compose(input)) :: HNil)
  }

  val trainNetwork = train

  def makeTrainingData(): (ExpectedLabel :: Input :: HNil)#Data = {
    import shapeless._
    val field0 = Random.nextBoolean()
    val field1 = Random.nextBoolean()
    val field2 = field0 ^ field1
    val scala.Seq(dropout0, dropout1, dropout2) = scala.Seq.fill(3)(false).updated(Random.nextInt(3), true)
    def input(isDropout: scala.Boolean, value: scala.Boolean) = {
      if (isDropout) {
        Inl(HNil)
      } else {
        Inr(Inl(Eval.now(if (value) {
          1.0
        } else {
          0.0
        })))
      }
    }
    def expectedLabel(isDropout: scala.Boolean, value: scala.Boolean) = {
      if (isDropout) {
        Inr(Inl(Eval.now(if (value) {
          1.0
        } else {
          0.0
        })))
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
      predictNetwork.predict(Inr(Inl(Eval.now(1.0))) :: Inl(HNil) :: Inr(Inl(Eval.now(0.0))) :: HNil)
    val loss = trainNetwork.predict(
      (Inl(HNil) :: Inr(Inl(Eval.now(1.0))) :: Inl(HNil) :: HNil) :: (Inr(Inl(Eval.now(1.0))) :: Inl(HNil) :: Inr(
        Inl(Eval.now(0.0))) :: HNil) :: HNil)
    println(raw"""${left.value}^${result.value}=${right.value}
loss: ${loss.value}
""")

    val (left10 :: right10 :: result10 :: HNil) =
      predictNetwork.predict(Inr(Inl(Eval.now(1.0))) :: Inr(Inl(Eval.now(0.0))) :: Inl(HNil) :: HNil)
    val (left01 :: right01 :: result01 :: HNil) =
      predictNetwork.predict(Inr(Inl(Eval.now(0.0))) :: Inr(Inl(Eval.now(1.0))) :: Inl(HNil) :: HNil)
    val (left00 :: right00 :: result00 :: HNil) =
      predictNetwork.predict(Inr(Inl(Eval.now(0.0))) :: Inr(Inl(Eval.now(0.0))) :: Inl(HNil) :: HNil)
    val (left11 :: right11 :: result11 :: HNil) =
      predictNetwork.predict(Inr(Inl(Eval.now(1.0))) :: Inr(Inl(Eval.now(1.0))) :: Inl(HNil) :: HNil)

    println(raw"""${left00.value}^${right00.value}=${result00.value}
${left01.value}^${right01.value}=${result01.value}
${left10.value}^${right10.value}=${result10.value}
${left11.value}^${right11.value}=${result11.value}
""")
  }

  "xor" in {
    for (i <- 0 until 500) {
//      predictAndPrint()
      trainNetwork.train(makeTrainingData())
    }
//    predictAndPrint();
    {
      import shapeless._
      val (left00 :: right00 :: result00 :: HNil) =
        predictNetwork.predict(Inr(Inl(Eval.now(0.0))) :: Inr(Inl(Eval.now(0.0))) :: Inl(HNil) :: HNil)
      result00.value should be < 0.5
      val (left01 :: right01 :: result01 :: HNil) =
        predictNetwork.predict(Inr(Inl(Eval.now(0.0))) :: Inr(Inl(Eval.now(1.0))) :: Inl(HNil) :: HNil)
      result01.value should be > 0.5
      val (left10 :: right10 :: result10 :: HNil) =
        predictNetwork.predict(Inr(Inl(Eval.now(1.0))) :: Inr(Inl(Eval.now(0.0))) :: Inl(HNil) :: HNil)
      result10.value should be > 0.5
      val (left11 :: right11 :: result11 :: HNil) =
        predictNetwork.predict(Inr(Inl(Eval.now(1.0))) :: Inr(Inl(Eval.now(1.0))) :: Inl(HNil) :: HNil)
      result11.value should be < 0.5

      val (_ :: result0x0 :: _ :: HNil) =
        predictNetwork.predict(Inr(Inl(Eval.now(0.0))) :: Inl(HNil) :: Inr(Inl(Eval.now(0.0))) :: HNil)
      result0x0.value should be < 0.5

      val (resultx11 :: _ :: _ :: HNil) =
        predictNetwork.predict(Inl(HNil) :: Inr(Inl(Eval.now(1.0))) :: Inr(Inl(Eval.now(1.0))) :: HNil)
      resultx11.value should be < 0.5

      val (_ :: result1x0 :: _ :: HNil) =
        predictNetwork.predict(Inr(Inl(Eval.now(1.0))) :: Inl(HNil) :: Inr(Inl(Eval.now(0.0))) :: HNil)
      result1x0.value should be > 0.5
    }
  }

}

object XorSpec {
  type OptionalDouble = HNil :+: Double :+: CNil
  type Input = OptionalDouble :: OptionalDouble :: OptionalDouble :: HNil
  type ExpectedLabel = OptionalDouble :: OptionalDouble :: OptionalDouble :: HNil
  type Output = Double :: Double :: Double :: HNil
}
