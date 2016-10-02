package com.thoughtworks.deepLearning

import cats.Eval
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

  "+ Array2D" in {

    def addArray(implicit symbolicInput: SymbolicInput {val ast: dsl.Array2D}) = {
      import symbolicInput.dsl._
      symbolicInput.ast + Double.weight(2.0)
    }
    val f1 = addArray.underlying

    def train(inputValue: Array[Array[scala.Double]]): Eval[INDArray] = {
      val minibatchInput: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]] = Literal(Eval.now(inputValue.toNDArray))
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
    def dotArray2D(dsl: Dsl)(inputNeuronNetwork: dsl.Array2D): dsl.Array2D = {
      import dsl._
      val weight = Array2D.weight(Array(Array(1.0, 1.0)))
      inputNeuronNetwork dot weight
    }

    val symbolicInput = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = D#Array2D}]
    val f1 = dotArray2D(symbolicInput.dsl)(symbolicInput.ast).underlying

    def train(inputValue: Array[Array[scala.Double]]): Eval[INDArray] = {
      val minibatchInput: Batch.Aux[Eval[INDArray], Eval[Option[INDArray]]] = Literal(Eval.now(inputValue.toNDArray))
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

    def maxDouble(dsl: Dsl)(ast: dsl.Double) = {
      import dsl._
      max(ast - Double.weight(0.0), 2.0)
    }

    val symbolicInput = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = D#Double}]
    val f1 = maxDouble(symbolicInput.dsl)(symbolicInput.ast).underlying

    def train(inputValue: scala.Double): scala.Double = {
      val minibatchInput: Batch.Aux[Eval[scala.Double], Eval[scala.Double]] = Literal(Eval.now(inputValue))
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
    def subtractDouble0(dsl: Dsl)(input: dsl.Double) = {
      import dsl._
      val negtiveInput: dsl.Double = -input
      negtiveInput - Double.weight(3.0)
    }

    def subtractDouble1(dsl: Dsl)(input: dsl.Double) = {
      import dsl._
      input - Double.weight(1.2)
    }

    val symbolicInput0 = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = D#Double}]
    val f1 = subtractDouble0(symbolicInput0.dsl)(symbolicInput0.ast).underlying
    val symbolicInput1 = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = D#Double}]
    val g1 = subtractDouble1(symbolicInput1.dsl)(symbolicInput1.ast).underlying
    val nn = Compose(f1, g1)
    def train(input: scala.Double): scala.Double = {
      val output = nn.forward(Literal(Eval.now(input)))
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

  "construct a coproduct" in {
    def getWeight(dsl: Dsl)(isLeft: dsl.Boolean) = {
      import dsl._
      val left: Double = Double.weight(100.0)
      val right: Double = Double.weight(1.0)
      isLeft.`if`[Double :+: Double :+: CNil] {
        Inl[Double, Double :+: CNil](left)
      } {
        Inr[Double, Double :+: CNil](Inl[Double, CNil](right))
      }(:+:(Double, :+:(Double, CNil)))
    }
    val symbolicIsLeft = shapeless.the[SymbolicInput {type Ast[D <: SymbolicDsl] = D#Boolean}]
    import symbolicIsLeft.dsl._
    val nn = getWeight(symbolicIsLeft.dsl)(symbolicIsLeft.ast).toDifferentiable(:+:(Double, :+:(Double, CNil)))

    val trainLeft0 = nn.forward(Literal(Eval.now(true)))
    inside(trainLeft0.value) {
      case shapeless.Inl(v) =>
        v.value should be(100.0)
    }
    trainLeft0.backward(trainLeft0.value)


    val trainLeft1 = nn.forward(Literal(Eval.now(true)))

    inside(trainLeft1.value) {
      case shapeless.Inl(v) =>
        v.value should be < 100.0
    }
    trainLeft1.backward(trainLeft1.value)


    val trainLeft2 = nn.forward(Literal(Eval.now(false)))
    inside(trainLeft2.value) {
      case shapeless.Inr(shapeless.Inl(v)) =>
        v.value should be(1.0)
    }
    trainLeft2.backward(trainLeft2.value)


    val trainLeft3 = nn.forward(Literal(Eval.now(false)))

    inside(trainLeft3.value) {
      case shapeless.Inr(shapeless.Inl(v)) =>
        v.value should be < 1.0
    }
    trainLeft3.backward(trainLeft3.value)


  }
}
