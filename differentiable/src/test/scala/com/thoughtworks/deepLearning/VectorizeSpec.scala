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

    type NN[OutputTypePair <: any.Any] = Ast.FromTypePair[InputTypePair, OutputTypePair]

    val toArray2D: NN[Array2D] = {

      val field0 = input[Batch.FromTypePair[InputTypePair]].head
      val rest0 = input[Batch.FromTypePair[InputTypePair]].tail
      val field1 = rest0.head
      val rest1 = rest0.tail
      val field2 = rest1.head
      val rest2 = rest1.tail
      val field3 = rest2.head
      val rest3 = rest2.tail

      val field0Flag0: NN[Double] = field0.choice { _ =>
        0.0
      } { _ =>
        1.0
      }

      val field0Flag1 = field0.choice { unknown =>
        0.5.toWeight
      } {
        _.choice { knownField0 =>
          knownField0.choice { unset =>
            0.0
          } { someValue =>
            1.0
          }
        } { _ =>
          `throw`(new IllegalArgumentException)
        }
      }

      val field0Value: NN[Double] = field0.choice { unknown: NN[HNil] =>
        0.5.toWeight: NN[Double]
      } {
        _.choice { knownField0 =>
          knownField0.choice { unset: NN[HNil] =>
            0.5.toWeight: NN[Double]
          } {
            _.choice { nativeDouble =>
              nativeDouble: NN[Double]
            } { _: NN[CNil] =>
              `throw`(new IllegalArgumentException): NN[Double]
            }: NN[Double]
          }: NN[Double]
        } { _: NN[CNil] =>
          `throw`(new IllegalArgumentException): NN[Double]
        }: NN[Double]

      }

      val field1Flag0 = {
        field1.choice { unknown =>
          0.0
        } {
          _.choice { known =>
            1.0
          } { _ =>
            `throw`(new IllegalArgumentException)
          }
        }
      }

      val field1Value0 = {
        field1.choice { unknown =>
          0.5.toWeight
        } {
          _.choice { known =>
            known.choice { unset =>
              0.0
            } {
              _.choice { set =>
                1.0
              } { cnil =>
                `throw`(new IllegalArgumentException)
              }
            }
          } { cnil =>
            `throw`(new IllegalArgumentException)
          }
        }
      }

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

  type Nullable[A <: Any] = HNil :+: A :+: CNil

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
