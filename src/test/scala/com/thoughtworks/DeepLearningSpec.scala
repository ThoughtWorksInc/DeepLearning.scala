package com.thoughtworks

import com.thoughtworks.DeepLearning.{Array2D, DifferentiableINDArray}
import shapeless.syntax.std.tuple._

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
import shapeless.{::, _}
import DeepLearning.DifferentiableINDArray.INDArrayToStrong
import DeepLearning.DifferentiableDouble.DoubleToStrong
import com.thoughtworks.Differentiable.DifferentiableFunction.ForwardPass
import com.thoughtworks.Differentiable.WeakOps._

private object DeepLearningSpec {

  object Networks {

    import DeepLearning.ops._

    /*assumption: minibatch. could try removing it*/
    def inference[F[_]](input: F[Array2D])(implicit deepLearning: DeepLearning[F]): F[Array2D] = {
      import deepLearning._
      val scores =/* show[Array2D]*/(input.fullyConnectedThenRelu(2, 50).fullyConnectedThenRelu(50, 2))
      show[Array2D](      exp(scores) / sum(1)(exp(scores)) )
    }

    def loss[F[_]](likelihood: F[Array2D], expectedLabels: F[Array2D])(implicit deepLearning: DeepLearning[F]): F[Double] = {
      import deepLearning._
      show[Double](-reduceSum(/*show[Array2D]*/(log(/*show[Array2D]*/(likelihood))) * expectedLabels))
    }

    type MinibatchInput = Array2D
    type ExpectedLabel = Array2D
    type Likelihood = Array2D
    type Loss = Double

    def inferenceAndTrain[F[_]](inputAndLabels: F[MinibatchInput :: ExpectedLabel :: HNil])(implicit deepLearning: DeepLearning[F]): F[Likelihood :: Loss :: HNil] = {
      import deepLearning._
      val likelihood = inference(inputAndLabels.head)
      likelihood :: loss(likelihood, inputAndLabels.tail.head) :: hnil
    }

    def train[F[_]](inputAndLabels: F[MinibatchInput :: ExpectedLabel :: HNil])(implicit deepLearning: DeepLearning[F]): F[Loss] = {
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
    val forwardPass = network.forward((Eval.now(Nd4j.valueArrayOf(1, 1, left)), Eval.now(Nd4j.valueArrayOf(1, 1, right))).productElements)
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
    val operator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = DeepLearning[F].dot(left, right)
    }
    test1x1Binary(0.5, 0.58, 0.29, operator)
  }
  "+" in {
    val operator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = left + right
    }
    test1x1Binary(0.5, 0.45, 0.95, operator)
  }
  "-" in {
    val operator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = left - right
    }
    test1x1Binary(0.5, 0.45, 0.05, operator)
  }
  "*" in {
    val operator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = left * right
    }
    val forwardPass = operator.hlistFunction.toStrong.forward(Eval.now(Array(Array(0.5)).toNDArray) :: Eval.now(Array(Array(0.58)).toNDArray) :: HNil)
    forwardPass.output.value(0, 0) should be(0.29 +- 0.01)
    val backwardPass = forwardPass.backward(Eval.now(Some(Array(Array(2.0)).toNDArray)))
    val leftDelta :: rightDelta :: HNil = backwardPass.inputDelta
    inside(leftDelta.value) {
      case Some(leftDeltaValue) =>
        leftDeltaValue(0, 0) should be((0.58 * 2.0) +- 0.01)
    }
    inside(rightDelta.value) {
      case Some(rightDeltaValue) =>
        rightDeltaValue(0, 0) should be((0.5 * 2.0) +- 0.01)
    }
  }
  "/" in {
    val operator = new BinaryOperator {
      override def apply[F[_] : DeepLearning](left: F[Array2D], right: F[Array2D]): F[Array2D] = left / right
    }
    test1x1Binary(0.5, 0.45, 1.1111111111, operator)
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
        println(inputDeltaValue)
        inputDeltaValue.shape should be(Array(2, 3))
    }
    inside(backwardPass.inputDelta(1).value) {
      case Some(weightDeltaValue) =>
        println(weightDeltaValue)
        weightDeltaValue.shape should be(Array(3, 5))
    }
  }

  "XOR" - {

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

      val weakNetwork = Networks.train[Lambda[A => WeakOps[Array2D :: Array2D :: HNil => A]]](DeepLearning[WeakOps].id)
      val toStrong = ToStrong[Array2D :: Array2D :: HNil => Double]
      val network = weakNetwork.toStrong(toStrong)
      type TrainFunction = toStrong.Out

      def train(network: TrainFunction): TrainFunction = {
        val forwardPass = ForwardOps(network).forward(inputAndLabel)
        val backwardPass = forwardPass.backward(Eval.now(1.0))
        val learningRate = 0.0003
        network.applyPatch(backwardPass.weightDelta /*.asInstanceOf[network.typeClassInstance.Delta]*/ , learningRate)
      }

      def currentLoss[Weight, DeltaWeight](network: TrainFunction) = {
        val forwardPass = network.forward(inputAndLabel)
        forwardPass.output.value
      }

      val initialLoss = currentLoss(network)
      val finalNetwork = (0 until 5).foldLeft[TrainFunction](network) { (network, currentIteration) =>
        train(network)
      }
      val finalLoss = currentLoss(finalNetwork)
      initialLoss should be > finalLoss
    }

    "predict" in {

      val weakNetwork = Networks.inferenceAndTrain[Lambda[A => WeakOps[Array2D :: Array2D :: HNil => A]]](DeepLearning[WeakOps].id)
      val toStrong = ToStrong[(Array2D :: Array2D :: HNil) => (Array2D :: Double :: HNil)]
      val network = weakNetwork.toStrong(toStrong)
      type TrainFunction = toStrong.Out

      def train(network: TrainFunction): TrainFunction = {
        val forwardPass = ForwardOps(network).forward(inputAndLabel)
        val predicted :: loss :: HNil = forwardPass.output
        predicted.value
//        println(loss.value)
        //        println(predicted.value)
        //        println(loss.value)
        //        println()
        val backwardPass = forwardPass.backward(Eval.now[Option[INDArray]](None) :: Eval.now(1.0) :: HNil)
        val learningRate = 0.003
        network.applyPatch(backwardPass.weightDelta /*.asInstanceOf[network.typeClassInstance.Delta]*/ , learningRate)
      }
      val (_ :: initialLoss :: HNil) = network.forward(inputAndLabel).output
      val finalNetwork = (0 until 10).foldLeft[TrainFunction](network) { (network, currentIteration) =>
        train(network)
      }

      val finalPredicted :: finalLoss :: HNil = finalNetwork.forward(inputAndLabel).output
      println(finalPredicted.value)
      finalPredicted.value(0, 0) should be > 0.5
      finalPredicted.value(0, 1) should be < 0.5
      initialLoss.value should be > finalLoss.value

    }
  }

  "id should forward and backward" in {
    import DeepLearning.DeepLearningInstances._
    val weakOps = id[Double]
    val strongOps = weakOps.toStrong
    @volatile var inputReadCount = 0
    val forwardPass = strongOps.forward(Eval.always {
      inputReadCount += 1
      4.5
    })
    inputReadCount should be(0)
    forwardPass.output.value should be(4.5)
    inputReadCount should be(1)
    @volatile var outputDeltaReadCount = 0
    val backwardPass = forwardPass.backward(Eval.always {
      outputDeltaReadCount += 1
      2.4
    })
    backwardPass.weightDelta should be(HNil)
    outputDeltaReadCount should be(0)
    backwardPass.inputDelta.value should be(2.4)
    outputDeltaReadCount should be(1)
  }


  "compose(id, id) should forward and backward" in {
    import DeepLearning.DeepLearningInstances._
    val weakOps = compose(id[Double], id[Double])
    val strongOps = weakOps.toStrong
    @volatile var inputReadCount = 0
    val forwardPass = strongOps.forward(Eval.always {
      inputReadCount += 1
      4.5
    })
    inputReadCount should be(0)
    forwardPass.output.value should be(4.5)
    inputReadCount should be(1)
    @volatile var outputDeltaReadCount = 0
    val backwardPass = forwardPass.backward(Eval.always {
      outputDeltaReadCount += 1
      2.4
    })
    outputDeltaReadCount should be(0)
    backwardPass.inputDelta.value should be(2.4)
    outputDeltaReadCount should be(1)
  }

  "update weight" in {

    def loss[F[_]](likelihood: F[Double])(implicit deepLearning: DeepLearning[F]): F[Double] = {
      import deepLearning._
      //      val l = -reduceSum(log(likelihood) * expectedLabels)
      val l = DeepLearning.constantDouble(3.4)(deepLearning)
      l + l

    }

    val l = loss[Lambda[A => WeakOps[Double => A]]](DeepLearning[WeakOps].id)

    println(l.toStrong.forward(Eval.now(2.0)).output.value)

  }
}