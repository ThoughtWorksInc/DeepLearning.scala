package com.thoughtworks.deepLearning

import cats.Eval
import cats.implicits._
import com.thoughtworks.deepLearning.Ast._
import com.thoughtworks.deepLearning.hlist._
import com.thoughtworks.deepLearning.double._
import com.thoughtworks.deepLearning.array2D._
import com.thoughtworks.deepLearning.any._
import com.thoughtworks.deepLearning.coproduct._
import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest._

import scala.language.implicitConversions
import scala.language.existentials
//
//final case class DifferentiableOps[InputTypePair <: dsl.Any]() {
//
//  type InputData = InputTypePair#Data
//  type InputDelta = InputTypePair#Delta
//
//  type Input = Batch.Aux[InputData, InputDelta]
//  val input = Identity[InputData, InputDelta]()
//
//  type Any = DifferentiableOps.Any[Input]
//
//  type Boolean = DifferentiableOps.Boolean[Input]
//
//  implicit def toBoolean(
//      differentiable: Ast.Aux[Input, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]]) =
//    new Boolean(differentiable)
//
//  type Double = DifferentiableOps.Double[Input]
//
//  implicit def toDouble(differentiable: Ast.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]]) =
//    new Double(differentiable)
//
//  type Coproduct = DifferentiableOps.CoproductOps[Input]
//  type CNil = DifferentiableOps.CNilOps[Input]
//
//  type :+:[Head <: Any, Tail <: Coproduct] =
//    DifferentiableOps.CConsOps[Input, Head#OutputData, Head#OutputDelta, Tail#OutputData, Tail#OutputDelta]
//
//  implicit def toCCons[HeadData, HeadDelta, TailData <: shapeless.Coproduct, TailDelta <: shapeless.Coproduct](
//      differentiable: Ast.Aux[
//        Input,
//        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) =
//    new DifferentiableOps.CConsOps(differentiable)
//
//  type HList = DifferentiableOps.HList[Input]
//  type HNil = DifferentiableOps.HNil[Input]
//  type ::[Head <: Any, Tail <: HList] =
//    DifferentiableOps.HCons[Input, Head#OutputData, Head#OutputDelta, Tail#OutputData, Tail#OutputDelta]
//
//  implicit def toHCons[HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.Coproduct](
//      differentiable: Ast.Aux[
//        Input,
//        Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]]) =
//    new DifferentiableOps.HCons(differentiable)
//
//  def `throw`(throwable: => Throwable) = {
//    Throw(Eval.later(throwable))
//  }
//
//  // TODO: upcast for HCons and CConsOps
//}
//
//object DifferentiableOps {
//
//  // TODO: Let Input contravariant
//  trait Any[Input <: Batch] extends scala.Any {
//    type OutputData
//    type OutputDelta
//    type Network = Ast.Aux[Input, Batch.Aux[OutputData, OutputDelta]]
//
//    val differentiable: Network
//
//  }
//
//  object Any {
//
//    private type Aux[Input <: Batch, OutputData0, OutputDelta0] = Any[Input] {
//      type OutputData = OutputData0
//      type OutputDelta = OutputDelta0
//    }
//
//    implicit def getDifferentiable[Input <: Batch, OutputData0, OutputDelta0](
//        any: Any.Aux[Input, OutputData0, OutputDelta0]) =
//      any.differentiable
//
//  }
//
//  final class Boolean[Input <: Batch](
//      val differentiable: Ast.Aux[Input, Batch.Aux[Eval[scala.Boolean], Eval[scala.Boolean]]])
//      extends AnyVal
//      with Any[Input] {
//    override type OutputData = Eval[scala.Boolean]
//    override type OutputDelta = Eval[scala.Boolean]
//  }
//
//  final class Double[Input <: Batch](
//      val differentiable: Ast.Aux[Input, Batch.Aux[Eval[scala.Double], Eval[scala.Double]]])
//      extends AnyVal
//      with Any[Input] {
//    override type OutputData = Eval[scala.Double]
//    override type OutputDelta = Eval[scala.Double]
//  }
//
//  sealed trait CoproductOps[Input <: Batch] extends scala.Any with Any[Input] {
//    override type OutputData <: shapeless.Coproduct
//    override type OutputDelta <: shapeless.Coproduct
//  }
//
//  final class CConsOps[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.Coproduct,
//  TailDelta <: shapeless.Coproduct](
//      val differentiable: Ast.Aux[
//        Input,
//        Batch.Aux[shapeless.:+:[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]])
//      extends CoproductOps[Input] {
//    override type OutputData = shapeless.:+:[HeadData, TailData]
//    override type OutputDelta = shapeless.:+:[HeadDelta, TailDelta]
//
//    private lazy val head = Head(differentiable)
//    private lazy val tail = Tail(differentiable)
//
//    def choice[ResultData, ResultDelta](
//        caseHead: Ast.Aux[Input, Batch.Aux[HeadData, HeadDelta]] => Ast.Aux[
//          Input,
//          Batch.Aux[ResultData, ResultDelta]])(
//        caseTail: Ast.Aux[Input, Batch.Aux[TailData, TailDelta]] => Ast.Aux[
//          Input,
//          Batch.Aux[ResultData, ResultDelta]]): Ast.Aux[Input, Batch.Aux[ResultData, ResultDelta]] = {
//      If(IsInl(differentiable), caseHead(head), caseTail(tail))
//    }
//
//  }
//
//  object CConsOps {
//    implicit def covariant[
//        Input <: Batch,
//        HeadData,
//        HeadDelta,
//        TailData <: shapeless.Coproduct,
//        TailDelta <: shapeless.Coproduct,
//        HeadData1 >: HeadData,
//        HeadDelta1 <: HeadDelta,
//        TailData1 >: TailData <: shapeless.Coproduct,
//        TailDelta1 <: TailDelta
//    ](ccons: CConsOps[Input, HeadData, HeadDelta, TailData, TailDelta])
//      : CConsOps[Input, HeadData1, HeadDelta1, TailData1, TailDelta1] = {
//      new CConsOps[Input, HeadData1, HeadDelta1, TailData1, TailDelta1](ccons.differentiable)
//    }
//  }
//
//  sealed trait CNilOps[Input <: Batch] extends scala.Any with CoproductOps[Input] {
//    override type OutputData = shapeless.CNil
//    override type OutputDelta = shapeless.CNil
//  }
//
//  sealed trait HList[Input <: Batch] extends scala.Any with Any[Input] {
//    override type OutputData <: shapeless.HList
//    override type OutputDelta <: shapeless.Coproduct
//  }
//
//  final class HCons[Input <: Batch, HeadData, HeadDelta, TailData <: shapeless.HList, TailDelta <: shapeless.Coproduct](
//      val differentiable: Ast.Aux[
//        Input,
//        Batch.Aux[shapeless.::[HeadData, TailData], shapeless.:+:[HeadDelta, TailDelta]]])
//      extends AnyVal
//      with HList[Input] {
//    override type OutputData = shapeless.::[HeadData, TailData]
//    override type OutputDelta = shapeless.:+:[HeadDelta, TailDelta]
//
//    def head = Head(differentiable)
//
//    def tail = Tail(differentiable)
//  }
//
//  final class HNil[Input <: Batch](
//      val differentiable: Ast.Aux[Input, Batch.Aux[shapeless.HNil, shapeless.CNil]])
//      extends AnyVal
//      with HList[Input] {
//    override type OutputData = shapeless.HNil
//    override type OutputDelta = shapeless.CNil
//  }
//
//}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class VectorizeSpec extends FreeSpec with Matchers {

  import VectorizeSpec._

  "Convert HMatrix to Array2D" in {
    /*
     TODO: 最终目标是生成一个预测神经网络和一个训练神经网络
     为了生成这两个网络，需要生成若干处理Array2D的全连接层、InputData到Array2D的转换、Array2D到Row的转换、Array2D到Double loss的转换

     InputData到Array2D的转换可以从InputData到若干Double的转换做起

     目前可以暂时使用HList而不是直接用case class的神经网络，将来可以直接使用case class

     */

    implicit val learningRate = new LearningRate {
      override def apply() = 0.0003
    }

//    val factory = DifferentiableOps[InputTypePair]()
//    import factory._

    type Network[OutputTypePair <: any.Any] = Ast.Aux[Batch.Aux[InputTypePair#Data, InputTypePair#Delta],
                                                                 Batch.Aux[OutputTypePair#Data, OutputTypePair#Delta]]

    val toArray2D: Network[Array2D] = {
//
      val field0 = input[InputTypePair].head
      val rest0 = input[InputTypePair].tail
      val field1 = rest0.head
      val rest1 = rest0.tail
      val field2 = rest1.head
      val rest2 = rest1.tail
      val field3 = rest2.head
      val rest3 = rest2.tail

      val field0Flag0 = field0.choice { _ =>
        doubleLiteral[Batch.Aux[InputTypePair#Data, InputTypePair#Delta]](0.0)
      } { _ =>
        doubleLiteral[Batch.Aux[InputTypePair#Data, InputTypePair#Delta]](1.0)
      }

//      val field0Flag1: Double#Network = field0.choice { unknown: HNil#Network =>
//        DoubleWeight(0.5)
//      } {
//        _.choice { knownField0 =>
//          knownField0.choice { unset: HNil#Network =>
//            Literal(Eval.now(0.0))
//          } { someValue: (Double :+: CNil)#Network =>
//            Literal(Eval.now(1.0))
//          }
//        } { _: CNil#Network =>
//          `throw`(new IllegalArgumentException)
//        }
//      }
//
//      val field0Value: Double#Network = field0.choice { unknown: HNil#Network =>
//        DoubleWeight(0.5)
//      } {
//        _.choice { knownField0 =>
//          knownField0.choice { unset: HNil#Network =>
//            DoubleWeight(0.5)
//          } {
//            _.choice[Eval[scala.Double], Eval[scala.Double]] { nativeDouble: Double#Network =>
//              nativeDouble
//            } { _: CNil#Network =>
//              `throw`(new IllegalArgumentException): Double#Network
//            }
//          }
//        } { _: CNil#Network =>
//          `throw`(new IllegalArgumentException)
//        }
//
//      }

      //      val isField1Unknown: Boolean = IsInl(field1)
      //      val defaultValueForField1Value0 = DoubleWeight(0.5)
      //      val field1Choice0 = Head(Tail(field1))
      //      val isField1Value0: Boolean = IsInl(field1Choice0)
      //      //      val isField1Value1 = IsInl(field1Choice1)
      //
      //      val field1Flag0 = {
      //        If(
      //          isField1Unknown,
      //          Literal(Eval.now(0.0)),
      //          Literal(Eval.now(1.0))
      //        )
      //      }
      //      val field1Value0 = {
      //        val defaultValue = DoubleWeight(0.5)
      //        If(
      //          isField1Unknown,
      //          defaultValue,
      //          If(
      //            isField1Value0,
      //            Literal(Eval.now(0.0)),
      //            Literal(Eval.now(1.0))
      //          )
      //        )
      //      }
      //
      //      val isField3Unknown: Boolean = IsInl(field3)
      //      val defaultValueForField3Value0 = DoubleWeight(0.5)
      //      val field3Choice0 = Head(Tail(field3))
      //      val field3Choice1 = Tail(field3Choice0)
      //      val isField3Value0: Boolean = IsInl(field3Choice0)
      //      val isField3Value1: Boolean = IsInl(field3Choice1)
      //
      //      val field3Flag0 = {
      //        If(
      //          isField3Unknown,
      //          Literal(Eval.now(0.0)),
      //          Literal(Eval.now(1.0))
      //        )
      //      }
      //      val field3Value0 = {
      //        val defaultValue = DoubleWeight(0.5)
      //        If(
      //          isField3Unknown,
      //          defaultValue,
      //          If(
      //            isField3Value0,
      //            Literal(Eval.now(0.0)),
      //            Literal(Eval.now(1.0))
      //          )
      //        )
      //      }
      //      val field3Value1 = {
      //        val defaultValue = DoubleWeight(0.5)
      //        If(
      //          isField3Unknown,
      //          defaultValue,
      //          If(
      //            isField3Value1,
      //            Literal(Eval.now(0.0)),
      //            Literal(Eval.now(1.0))
      //          )
      //        )
      //      }
      //
      ???

    }

    //    val predict: Ast.Aux[Batch.Aux[InputData, _], Batch.Aux[Row, _]] = ???
    //
    //    val train: Ast.Aux[Batch.Aux[InputData :: ExpectedLabelData :: HNil, _], Batch.Aux[Eval[Double], _]] = ???

  }

}

object VectorizeSpec {

//  import shapeless._
  type Nullable[A <: Any] = HNil :+: A :+: CNil

//  // TODO: 将来要重构，把类型映射的元信息提到某个trait上。就不用写一遍Data再写一遍Delta了。

  type InputField[A <: Any] = HNil :+: A :+: CNil

  type LabelField[A <: Any] = HNil :+: A :+: CNil

  type Enum0 = HNil :+: HNil :+: CNil
  type Enum1 = HNil :+: HNil :+: HNil :+: CNil

  type Row = Nullable[Double] :: Enum0 :: Double :: Enum1 :: HNil

  type InputTypePair =
    InputField[Nullable[Double]] :: InputField[Enum0] :: InputField[Double] :: InputField[Enum1] :: HNil

  type ExpectedLabel =
    LabelField[Nullable[Double]] :: LabelField[Enum0] :: LabelField[Double] :: LabelField[Enum1] :: HNil

}
