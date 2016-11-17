package com.thoughtworks.deepLearning.any

import com.thoughtworks.deepLearning.{Batch, NeuralNetwork}
import shapeless._

import scala.language.existentials

// TODO: Move to dsl library
private[deepLearning] sealed trait ToNeuralNetworkLowPriorityImplicits {

  implicit def toNeuralNetworkOfType[Input0 <: Batch, OutputType <: Type[_, _]]
    : ToNeuralNetwork.OfType[NeuralNetwork.Aux[Input0, OutputType#Batch], Input0, OutputType] = {
    ToNeuralNetwork
      .neuralNetworkToNeuralNetwork[Input0, OutputType#Data, OutputType#Delta]
      .asInstanceOf[ToNeuralNetwork.OfType[NeuralNetwork.Aux[Input0, OutputType#Batch], Input0, OutputType]]
  }

}

object ToNeuralNetwork extends ToNeuralNetworkLowPriorityImplicits {
  trait AstPoly1 extends Poly1 {
    implicit def toNeuralNetworkCase[Operand, Input <: Batch, OperandData, OperandDelta](
        implicit toNeuralNetwork: ToNeuralNetwork.Aux[Operand, Input, OperandData, OperandDelta],
        astCase: Lazy[Case[NeuralNetwork.Aux[Input, Batch.Aux[OperandData, OperandDelta]]]]
    ): Case.Aux[Operand, astCase.value.Result] = {
      at { operand =>
        astCase.value(toNeuralNetwork(operand))
      }
    }
  }

  trait AstPoly2 extends Poly2 {
    implicit def toNeuralNetworkCase[LeftOperand,
                                     RightOperand,
                                     Input <: Batch,
                                     LeftData,
                                     LeftDelta,
                                     RightData,
                                     RightDelta](
        implicit leftToNeuralNetwork: ToNeuralNetwork.Aux[LeftOperand, Input, LeftData, LeftDelta],
        rightToNeuralNetwork: ToNeuralNetwork.Aux[RightOperand, Input, RightData, RightDelta],
        astCase: Lazy[Case[NeuralNetwork.Aux[Input, Batch.Aux[LeftData, LeftDelta]],
                           NeuralNetwork.Aux[Input, Batch.Aux[RightData, RightDelta]]]]
    ): Case.Aux[LeftOperand, RightOperand, astCase.value.Result] = {
      at { (left, right) =>
        val leftAst = leftToNeuralNetwork(left)
        val rightAst = rightToNeuralNetwork(right)
        astCase.value(leftAst, rightAst)
      }
    }
  }

  type Aux[From, Input <: Batch, OutputData0, OutputDelta0] = ToNeuralNetwork[From, Input] {
    type OutputData = OutputData0
    type OutputDelta = OutputDelta0
  }

  type OfType[From, Input <: Batch, OutputType <: Type[_, _]] =
    ToNeuralNetwork.Aux[From, Input, differentiableType.Data, differentiableType.Delta] forSome {
      val differentiableType: OutputType
    }

  // FIXME: I don't know if invariance is required, please remove this line if NeuralNetwork.Aux is enough
  //  type NeuralNetwork.Aux[Input <: Batch, Output <: Batch] = NeuralNetwork.Aux[Input, Output]

  implicit def neuralNetworkToNeuralNetwork[Input <: Batch, OutputData0, OutputDelta0]
    : ToNeuralNetwork.Aux[NeuralNetwork.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]],
                          Input,
                          OutputData0,
                          OutputDelta0] =
    new ToNeuralNetwork[NeuralNetwork.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input] {
      override type OutputData = OutputData0
      override type OutputDelta = OutputDelta0

      override def apply(ast: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = ast
    }

  implicit def literalToNeuralNetwork[From, InputData, InputDelta, OutputData0, OutputDelta0](
      implicit inputType: Type[InputData, InputDelta],
      toLiteral: ToLiteral.Aux[From, OutputData0, OutputDelta0])
    : ToNeuralNetwork.Aux[From, Batch.Aux[InputData, InputDelta], OutputData0, OutputDelta0] = {
    new ToNeuralNetwork[From, Batch.Aux[InputData, InputDelta]] {
      override type OutputData = OutputData0
      override type OutputDelta = OutputDelta0
      override def apply(from: From) = toLiteral(from)
    }
  }

}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait ToNeuralNetwork[From, Input <: Batch] extends DepFn1[From] {
  type OutputData
  type OutputDelta
  type Output = Batch.Aux[OutputData, OutputDelta]
  type Out = NeuralNetwork.Aux[Input, Output]
}
