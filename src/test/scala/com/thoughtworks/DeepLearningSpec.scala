package com.thoughtworks

import com.thoughtworks.DeepLearning.{Array2D, DifferentiableINDArray}

import scala.language.existentials
import scala.language.higherKinds
import Predef.{any2stringadd => _, _}
import DeepLearning.ops._
import DeepLearning._
import cats.{Applicative, Eval}
import com.thoughtworks.Differentiable.{DifferentiableHNil, _}
import com.thoughtworks.Pointfree.Ski
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.scalatest.{FreeSpec, Inside, Matchers}
import org.nd4s.Implicits._
import shapeless._
import DeepLearning.DifferentiableINDArray.INDArrayToStrong
import DeepLearning.DifferentiableDouble.DoubleToStrong

private object DeepLearningSpec {

  object Networks {

    import DeepLearning.ops._

    def inference[F[_]](input: F[Array2D])(implicit deepLearning: DeepLearning[F]): F[Array2D] = {
      import deepLearning._
      val scores = input.fullyConnectedThenRelu(2, 2)
      exp(scores) / sum(1)(exp(scores))
    }

    def loss[F[_]](likelihood: F[Array2D], expectedLabels: F[Array2D])(implicit deepLearning: DeepLearning[F]): F[Double] = {
      import deepLearning._
      -reduceSum(log(likelihood) * expectedLabels)
    }

    def inferenceAndTrain[F[_]](inputAndLabels: F[Array2D :: Array2D :: HNil])(implicit deepLearning: DeepLearning[F]): F[Array2D :: Double :: HNil] = {
      import deepLearning._
      val likelihood: F[Array2D] = inference(inputAndLabels.head)
      likelihood :: loss(likelihood, inputAndLabels.tail.head) :: hnil
    }

    def train[F[_]](inputAndLabels: F[Array2D :: Array2D :: HNil])(implicit deepLearning: DeepLearning[F]): F[Double] = {
      import deepLearning._
      loss(inference(inputAndLabels.head), inputAndLabels.tail.head)
    }
  }

}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers with Inside {

  import DeepLearningSpec._

  def test1x1Binary(left: Double, right: Double, expected: Double, operator: BinaryOperator) = {
    val network = operator.hlistFunction.toStrong
    import shapeless.syntax.std.tuple._
    val forwardPass = network forward (Eval.now(Nd4j.valueArrayOf(1, 1, left)), Eval.now(Nd4j.valueArrayOf(1, 1, right))).productElements
    forwardPass.output.value.shape should be(Array(1, 1))
    forwardPass.output.value(0, 0) should be(expected +- 0.01)
  }

  type UnaryOperatorFunction[Weight, DeltaWeight] =
  StrongOps[
    Weight, DeltaWeight,
    DifferentiableFunction[
      Weight, DeltaWeight,
      Eval[INDArray], Eval[Option[INDArray]], DifferentiableINDArray.type,
      Eval[INDArray], Eval[Option[INDArray]], DifferentiableINDArray.type
      ]
    ]

  trait UnaryOperator {
    def apply[F[_] : DeepLearning](input: F[Array2D]): F[Array2D]

    final def function[F[_] : DeepLearning]: F[Array2D => Array2D] = {
      apply[Lambda[X => F[Array2D => X]]](DeepLearning[F].id)
    }

    def test1x1(input: Double, expected: Double) = {
      def test[Weight, DeltaWeight](network: UnaryOperatorFunction[Weight, DeltaWeight]): Unit = {
        val forwardPass = network.forward(Eval.now(Nd4j.valueArrayOf(1, 1, input)))
        forwardPass.output.value.shape should be(Array(1, 1))
        forwardPass.output.value(0, 0) should be(expected +- 0.01)
      }
      test(function.asInstanceOf[UnaryOperatorFunction[_, _]])

    }

  }

  trait BinaryOperator {
    def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D]

    final def fromHlist[F[_] : DeepLearning](input: F[Array2D :: Array2D :: HNil]): F[Array2D] = {
      apply(input.head, input.tail.head)
    }

    final def hlistFunction[F[_] : DeepLearning]: F[Array2D :: Array2D :: HNil => Array2D] = {
      fromHlist[Lambda[X => F[Array2D :: Array2D :: HNil => X]]](DeepLearning[F].id)
    }
  }

  "neg" in {
    val op = new UnaryOperator {
      override def apply[F[_] : DeepLearning](input: F[Array2D]): F[Array2D] = {
        -input
      }
    }
    op.test1x1(1.0, -1.0)
    op.test1x1(1.5, -1.5)
    op.test1x1(-1.0, 1.0)
  }

  "exp" in {
    val op = new UnaryOperator {
      override def apply[F[_] : DeepLearning](input: F[Array2D]): F[Array2D] = DeepLearning[F].exp(input)
    }
    op.test1x1(0.0, 1.0)
    op.test1x1(1.0, math.E)
    op.test1x1(-1.0, 1 / math.E)
  }

  "dot 1x1" in {
    val addOperator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = DeepLearning[F].dot(left, right)
    }
    test1x1Binary(0.5, 0.45, 0.225, addOperator)
  }
  "+" in {
    val addOperator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = left + right
    }
    test1x1Binary(0.5, 0.45, 0.95, addOperator)
  }
  "-" in {
    val addOperator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = left - right
    }
    test1x1Binary(0.5, 0.45, 0.05, addOperator)
  }
  "*" in {
    val addOperator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = left * right
    }
    test1x1Binary(0.5, 0.45, 0.225, addOperator)
  }
  "/" in {
    val addOperator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = left / right
    }
    test1x1Binary(0.5, 0.45, 1.1111111111, addOperator)
  }

  "dot" in {
    val input = Nd4j.arange(6).reshape(2, 3)
    val weight = Nd4j.arange(15).reshape(3, 5)
    val expectedOutput = Nd4j.arange(10).reshape(2, 5)
    val forwardPass = DeepLearning.DeepLearningInstances.dot.forward(Eval.now(input) :: Eval.now(weight) :: HNil)
    val output = forwardPass.output
    output.value.shape should be(Array(2, 5))
    val delta = output.map[Option[INDArray]] { outputValue =>
      Some(expectedOutput - outputValue)
    }
    val backwardPass = forwardPass.backward(delta)
    backwardPass.weightDelta should be(HNil)

    inside(backwardPass.inputDelta(0).value) {
      case Some(inputDeltaValue) =>
        inputDeltaValue.shape should be(Array(2, 3))
    }
    inside(backwardPass.inputDelta(1).value) {
      case Some(weightDeltaValue) =>
        weightDeltaValue.shape should be(Array(3, 5))
    }
  }

  "XOR" - {

    val inputTypeClass = DifferentiableINDArray :: DifferentiableINDArray :: DifferentiableHNil

    val inputTypeClassWiden = Widen[inputTypeClass.type]

    val inputAndLabel = Eval.now(
      Array(
        Array(0.0, 0.0),
        Array(0.0, 1.0),
        Array(1.0, 0.0),
        Array(1.0, 1.0)
      ).toNDArray
    ) :: Eval.now(
      Array(
        Array(1.0, 0.0),
        Array(0.0, 1.0),
        Array(0.0, 1.0),
        Array(1.0, 0.0)
      ).toNDArray
    ) :: HNil

    "train" in {


      type TrainFunction[Weight, DeltaWeight] =
      StrongOps[
        Weight, DeltaWeight,
        DifferentiableFunction[
          Weight, DeltaWeight,
          Eval[INDArray] :: Eval[INDArray] :: HNil,
          Eval[Option[INDArray]] :: Eval[Option[INDArray]] :: HNil, inputTypeClassWiden.Out,
          Eval[Double], Eval[Double], DifferentiableDouble.type
          ]
        ]

      val weakNetwork = Networks.train[Lambda[A => WeakOps[Array2D :: Array2D :: HNil => A]]](DeepLearning[WeakOps].id)

      val strongNetwork: TrainFunction[_, _] = weakNetwork.asInstanceOf[TrainFunction[_, _]]

      def train[Weight, DeltaWeight]
      (network: TrainFunction[Weight, DeltaWeight]): TrainFunction[Weight, DeltaWeight] = {
        val forwardPass = network.forward(inputAndLabel)
        val loss = forwardPass.output
        val backwardPass = forwardPass.backward(loss)
        val learningRate = 0.0003
        new WeakOps[Array2D :: Array2D :: HNil => Double] with Differentiable.AllOps[Weight] {
          override val typeClassInstance = network.typeClassInstance
          override val self = network.applyPatch(backwardPass.weightDelta, learningRate)
        }
      }

      def currentLoss[Weight, DeltaWeight](network: TrainFunction[Weight, DeltaWeight]) = {
        val forwardPass = network.forward(inputAndLabel)
        forwardPass.output.value
      }

      val initialLoss = currentLoss(strongNetwork)
      val finalNetwork = (0 until 5).foldLeft(strongNetwork) { (network, currentIteration) =>
        train(network)
      }
      val finalLoss = currentLoss(finalNetwork)
      initialLoss should be > finalLoss
    }

    "predict" in {
      val scoresAndLossTypeClass = DifferentiableINDArray :: DifferentiableDouble :: DifferentiableHNil
      val scoresAndLossTypeClassWiden = Widen[scoresAndLossTypeClass.type]
      type InferenceAndTrainFunction[Weight, DeltaWeight] =
      StrongOps[
        Weight, DeltaWeight,
        DifferentiableFunction[
          Weight, DeltaWeight,

          // input forward
          Eval[INDArray] :: Eval[INDArray] :: HNil,

          // input backward
          Eval[Option[INDArray]] :: Eval[Option[INDArray]] :: HNil,

          inputTypeClassWiden.Out,

          // output forward
          Eval[INDArray] :: Eval[Double] :: HNil,

          // output backward
          Eval[Option[INDArray]] :: Eval[Double] :: HNil,

          scoresAndLossTypeClassWiden.Out
          ]
        ]
      val weakNetwork = Networks.inferenceAndTrain[Lambda[A => WeakOps[Array2D :: Array2D :: HNil => A]]](DeepLearning[WeakOps].id)
      val strongNetwork: InferenceAndTrainFunction[_, _] = weakNetwork.asInstanceOf[InferenceAndTrainFunction[_, _]]

      def train[Weight, DeltaWeight]
      (network: InferenceAndTrainFunction[Weight, DeltaWeight]): InferenceAndTrainFunction[Weight, DeltaWeight] = {
        val forwardPass = network.forward(inputAndLabel)
        val predicted :: loss :: HNil = forwardPass.output
        //        println(inputAndLabel.tail.head.value)
        //        println(predicted.value)
        println(loss.value)
        println()
        val backwardPass = forwardPass.backward(Eval.now[Option[INDArray]](None) :: loss :: HNil)
        val learningRate = 0.00001
        new WeakOps[Array2D :: Array2D :: HNil => Double] with Differentiable.AllOps[Weight] {
          override val typeClassInstance = network.typeClassInstance
          override val self = network.applyPatch(backwardPass.weightDelta, learningRate)
        }
      }
      val finalNetwork = (0 until 100).foldLeft(strongNetwork) { (network, currentIteration) =>
        train(network)
      }

      def predict[Weight, DeltaWeight](network: InferenceAndTrainFunction[Weight, DeltaWeight]) = {

        val forwardPass = network.forward(inputAndLabel)
        val prediction = forwardPass.output.head.value
        println(prediction)
        prediction(0, 0) should be > 0.5
        prediction(0, 1) should be < 0.5

      }

      //      println(finalNetwork.toFastring)

      predict(finalNetwork)
    }

  }
}