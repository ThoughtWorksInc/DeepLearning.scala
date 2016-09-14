package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.Differentiable.DifferentiableArray2D.Array2DLiteral
import com.thoughtworks.deepLearning.Differentiable.DifferentiableDouble.DoubleLiteral
import com.thoughtworks.deepLearning.Differentiable._
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

  "+ Double" in {
    def f(dsl: Dsl)(inputNeuronNetwork: dsl.Array2D): dsl.Array2D = {
      import dsl._
      inputNeuronNetwork + Double.weight(2.0)
    }

    val id = new Id[Eval[INDArray], Eval[Option[INDArray]]]
    val dsl = SymbolicDsl[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
    val f1 = f(dsl)(dsl.Array2D.specialize(id))

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
    def f(dsl: Dsl)(inputNeuronNetwork: dsl.Array2D): dsl.Array2D = {
      import dsl._
      val weight = Array2D.weight(Array(Array(1.0, 1.0)))
      inputNeuronNetwork dot weight
    }

    val id = new Id[Eval[INDArray], Eval[Option[INDArray]]]
    val dsl = SymbolicDsl[Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]]]
    val f1 = f(dsl)(dsl.Array2D.specialize(id))

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
    def f(dsl: Dsl)(input: dsl.Double): dsl.Double = {
      import dsl._
      max(input - Double.weight(0.0), Double(2.0))
    }
    val id = new Id[Eval[scala.Double], Eval[scala.Double]]
    val dsl = SymbolicDsl[Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]
    val f1 = f(dsl)(dsl.Double.specialize(id))

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
    def f(dsl: Dsl)(input: dsl.Double): dsl.Double = {
      import dsl._
      val negtiveInput: dsl.Double = -input
      negtiveInput - Double.weight(3.0)
    }
    def g(dsl: Dsl)(input: dsl.Double): dsl.Double = {
      import dsl._
      input - Double.weight(1.2)
    }

    val id = new Id[Eval[scala.Double], Eval[scala.Double]]
    val dsl = SymbolicDsl[id.Input]
    val f1 = f(dsl)(dsl.Double.specialize(id))


    val g1 = g(dsl)(dsl.Double.specialize(id))
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
