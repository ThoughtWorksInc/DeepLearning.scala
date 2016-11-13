package com.thoughtworks.deepLearning

import cats._
import cats.implicits._
import shapeless.PolyDefns._
import shapeless.{Lazy, Poly1, Poly2}

import shapeless.DepFn1
import org.nd4s.Implicits._
import com.thoughtworks.deepLearning._
import com.thoughtworks.deepLearning.any.ast.Identity
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.language.existentials

private[deepLearning] sealed trait IsNeuralNetworkLowPriorityImplicits {

  implicit def isNeuralNetworkOfType[Input0 <: Batch, OutputType <: Type[_, _]]
    : IsNeuralNetwork.OfType[NeuralNetwork.Aux[Input0, OutputType#Batch], Input0, OutputType] = {
    IsNeuralNetwork
      .astIsNeuralNetwork[Input0, OutputType#Data, OutputType#Delta]
      .asInstanceOf[IsNeuralNetwork.OfType[NeuralNetwork.Aux[Input0, OutputType#Batch], Input0, OutputType]]
  }

}

object IsNeuralNetwork extends IsNeuralNetworkLowPriorityImplicits {
  trait AstPoly1 extends Poly1 {
    implicit def isNeuralNetworkCase[Operand, Input <: Batch, OperandData, OperandDelta](
        implicit isNeuralNetwork: IsNeuralNetwork.Aux[Operand, Input, OperandData, OperandDelta],
        astCase: Lazy[Case[NeuralNetwork.Aux[Input, Batch.Aux[OperandData, OperandDelta]]]]
    ): Case.Aux[Operand, astCase.value.Result] = {
      at { operand =>
        astCase.value(isNeuralNetwork(operand))
      }
    }
  }

  trait AstPoly2 extends Poly2 {
    implicit def isNeuralNetworkCase[LeftOperand, RightOperand, Input <: Batch, LeftData, LeftDelta, RightData, RightDelta](
        implicit leftIsNeuralNetwork: IsNeuralNetwork.Aux[LeftOperand, Input, LeftData, LeftDelta],
        rightIsNeuralNetwork: IsNeuralNetwork.Aux[RightOperand, Input, RightData, RightDelta],
        astCase: Lazy[Case[NeuralNetwork.Aux[Input, Batch.Aux[LeftData, LeftDelta]],
                           NeuralNetwork.Aux[Input, Batch.Aux[RightData, RightDelta]]]]
    ): Case.Aux[LeftOperand, RightOperand, astCase.value.Result] = {
      at { (left, right) =>
        val leftAst = leftIsNeuralNetwork(left)
        val rightAst = rightIsNeuralNetwork(right)
        astCase.value(leftAst, rightAst)
      }
    }
  }

  type Aux[From, Input <: Batch, OutputData0, OutputDelta0] = IsNeuralNetwork[From, Input] {
    type OutputData = OutputData0
    type OutputDelta = OutputDelta0
  }

  type OfBatch[From, Input <: Batch, Output0 <: Batch] = IsNeuralNetwork[From, Input] {
    type Output = Output0
  }

  type OfType[From, Input <: Batch, OutputType <: Type[_, _]] =
    IsNeuralNetwork.Aux[From, Input, differentiableType.Data, differentiableType.Delta] forSome {
      val differentiableType: OutputType
    }

  // FIXME: I don't know if invariance is required, please remove this line if NeuralNetwork.Aux is enough
  //  type NeuralNetwork.Aux[Input <: Batch, Output <: Batch] = NeuralNetwork.Aux[Input, Output]

  implicit def astIsNeuralNetwork[Input <: Batch, OutputData0, OutputDelta0]
    : IsNeuralNetwork.Aux[NeuralNetwork.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]],
                          Input,
                          OutputData0,
                          OutputDelta0] =
    new IsNeuralNetwork[NeuralNetwork.Aux[Input, Batch.Aux[OutputData0, OutputDelta0]], Input] {
      override type OutputData = OutputData0
      override type OutputDelta = OutputDelta0

      override def apply(ast: NeuralNetwork.Aux[Input, Batch.Aux[OutputData, OutputDelta]]) = ast
    }

  implicit def inputTypeIsNeuralNetwork[InputData, InputDelta]
    : IsNeuralNetwork.Aux[Type[InputData, InputDelta],
                          Batch.Aux[InputData, InputDelta],
                          InputData,
                          InputDelta] =
    new IsNeuralNetwork[Type[InputData, InputDelta], Batch.Aux[InputData, InputDelta]] {
      override type OutputData = InputData
      override type OutputDelta = InputDelta

      override def apply(input: Type[InputData, InputDelta]) =
        Identity[Batch.Aux[InputData, InputDelta]]()
    }
}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait IsNeuralNetwork[From, Input <: Batch] extends DepFn1[From] {
  type OutputData
  type OutputDelta
  type Output = Batch.Aux[OutputData, OutputDelta]
  type Out = NeuralNetwork.Aux[Input, Output]
}
