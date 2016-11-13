package com.thoughtworks.deepLearning

import shapeless.{Lazy, Poly1, Poly2}

import shapeless.DepFn1
import com.thoughtworks.deepLearning.any.ast.Identity

import scala.language.existentials

private[deepLearning] sealed trait ToNeuralNetworkLowPriorityImplicits {

  implicit def toNeuralNetworkOfType[Input0 <: Batch, OutputType <: Type[_, _]]
    : ToNeuralNetwork.OfType[NeuralNetwork.Aux[Input0, OutputType#Batch], Input0, OutputType] = {
    ToNeuralNetwork
      .astToNeuralNetwork[Input0, OutputType#Data, OutputType#Delta]
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

  type OfBatch[From, Input <: Batch, Output0 <: Batch] = ToNeuralNetwork[From, Input] {
    type Output = Output0
  }

  type OfType[From, Input <: Batch, OutputType <: Type[_, _]] =
    ToNeuralNetwork.Aux[From, Input, differentiableType.Data, differentiableType.Delta] forSome {
      val differentiableType: OutputType
    }

  // FIXME: I don't know if invariance is required, please remove this line if NeuralNetwork.Aux is enough
  //  type NeuralNetwork.Aux[Input <: Batch, Output <: Batch] = NeuralNetwork.Aux[Input, Output]

  implicit def astToNeuralNetwork[Input <: Batch, OutputData0, OutputDelta0]
    : ToNeuralNetwork.Aux[NeuralNetwork.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]],
                          Input,
                          OutputData0,
                          OutputDelta0] =
    new ToNeuralNetwork[NeuralNetwork.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input] {
      override type OutputData = OutputData0
      override type OutputDelta = OutputDelta0

      override def apply(ast: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = ast
    }

  implicit def inputTypeToNeuralNetwork[InputData, InputDelta]
    : ToNeuralNetwork.Aux[Type[InputData, InputDelta], Batch.Aux[InputData, InputDelta], InputData, InputDelta] =
    new ToNeuralNetwork[Type[InputData, InputDelta], Batch.Aux[InputData, InputDelta]] {
      override type OutputData = InputData
      override type OutputDelta = InputDelta

      override def apply(input: Type[InputData, InputDelta]) =
        Identity[Batch.Aux[InputData, InputDelta]]()
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
