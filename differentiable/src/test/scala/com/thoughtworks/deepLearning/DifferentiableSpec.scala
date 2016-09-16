package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.Differentiable.DifferentiableArray2D.Array2DLiteral
import com.thoughtworks.deepLearning.Differentiable.DifferentiableDouble.DoubleLiteral
import com.thoughtworks.deepLearning.Differentiable._
import com.thoughtworks.deepLearning.Dsl.DslFactory
import org.scalatest._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DifferentiableSpec extends FreeSpec with Matchers with Inside {

  implicit def learningRate = new Differentiable.LearningRate {
    def apply() = 0.03
  }

  "+ Array2D" in {

    final case class AddDouble[D <: Dsl](dsl: D) extends DslFactory {

      import dsl._

      type Out = Array2D
      type In = Array2D

      override def apply(inputNeuronNetwork: In): Out = {
        inputNeuronNetwork + Double.weight(2.0)
      }
    }

    val f1 = Differentiable.fromDsl[AddDouble]

    def train(inputValue: Array[Array[scala.Double]]): Eval[INDArray] = {
      val minibatchInput: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]] = Array2DLiteral(inputValue)
      val minibatchOutput = f1.forward(minibatchInput)
      val outputValue: Eval[INDArray] = minibatchOutput.value
      minibatchOutput.backward(outputValue.map[Option[INDArray]](Some(_)))
      outputValue
    }

    val output0 = train(Array(Array(1.0), Array(1.0))).value

    output0.shape should be(Array(2, 1))

    val initialLoss = output0.sumT

    for (_ <- 0 until 100) {
      train(Array(Array(1.0), Array(1.0)))
    }

    math.abs(train(Array(Array(1.0), Array(1.0))).value.sumT) should be < initialLoss
    math.abs(train(Array(Array(1.0), Array(1.0))).value.sumT) should be < 0.05
  }

  "dot" in {
    final case class DotArray2D[D <: Dsl](dsl: D) extends DslFactory {

      import dsl._

      type Out = Array2D
      type In = Array2D

      override def apply(inputNeuronNetwork: In): Out = {
        val weight = Array2D.weight(Array(Array(1.0, 1.0)))
        inputNeuronNetwork dot weight
      }
    }

    val f1 = Differentiable.fromDsl[DotArray2D]

    def train(inputValue: Array[Array[scala.Double]]): Eval[INDArray] = {
      val minibatchInput: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]] = Array2DLiteral(inputValue)
      val minibatchOutput = f1.forward(minibatchInput)
      val outputValue: Eval[INDArray] = minibatchOutput.value
      minibatchOutput.backward(outputValue.map[Option[INDArray]](Some(_)))
      outputValue
    }

    val output0 = train(Array(Array(1.0), Array(1.0))).value
    output0.shape should be(Array(2, 2))
    val initialLoss = output0.sumT
    for (_ <- 0 until 100) {
      train(Array(Array(1.0), Array(1.0)))
    }
    math.abs(train(Array(Array(1.0), Array(1.0))).value.sumT) should be < initialLoss
  }

  "max" in {
    final case class MaxDouble[D <: Dsl](dsl: D) extends DslFactory {

      import dsl._

      type Out = Double
      type In = Double

      override def apply(input: In): Out = {
        max(input - Double.weight(0.0), 2.0)
      }
    }

    val f1 = Differentiable.fromDsl[MaxDouble]
    def train(inputValue: scala.Double): scala.Double = {
      val minibatchInput: Batch.Aux[Eval[scala.Double], Eval[scala.Double]] = DoubleLiteral(inputValue)
      val minibatchOutput = f1.forward(minibatchInput)
      minibatchOutput.backward(minibatchOutput.value)
      minibatchOutput.value.value
    }
    val initialLoss = train(10.11)
    math.abs(initialLoss) should be(10.11)
    for (_ <- 0 until 100) {
      train(10.11)
    }
    math.abs(train(10.11)) should be < math.abs(initialLoss)

    math.abs(train(10.11)) should be < 2.5
    math.abs(train(10.11)) should be >= 2.0
  }

  "-" in {
    final case class SubtractDouble0[D <: Dsl](dsl: D) extends DslFactory {

      import dsl._

      type Out = Double
      type In = Double

      override def apply(input: In): Out = {
        import dsl._
        val negtiveInput: dsl.Double = -input
        negtiveInput - Double.weight(3.0)
      }
    }

    final case class SubtractDouble1[D <: Dsl](dsl: D) extends DslFactory {

      import dsl._

      type Out = Double
      type In = Double

      override def apply(input: In): Out = {
        input - Double.weight(1.2)
      }
    }
    val f1 = Differentiable.fromDsl[SubtractDouble0]
    val g1 = Differentiable.fromDsl[SubtractDouble1]
    val nn = Compose(f1, g1)
    def train(input: scala.Double): scala.Double = {
      val output = nn.forward(DoubleLiteral(input))
      output.backward(output.value)
      output.value.value
    }
    val initialLoss = train(10.11)
    for (_ <- 0 until 100) {
      train(10.11)
    }
    math.abs(train(10.11)) should be < math.abs(initialLoss)

    math.abs(train(10.11)) should be < 0.5
  }

}
