package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.hlist._
import com.thoughtworks.deepLearning.boolean._
import com.thoughtworks.deepLearning.double._
import com.thoughtworks.deepLearning.array2D._
import com.thoughtworks.deepLearning.any._
import com.thoughtworks.deepLearning.coproduct._
import org.scalatest._

import scala.language.implicitConversions
import scala.language.existentials

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class VectorizeSpec extends FreeSpec with Matchers {

  import VectorizeSpec._
  type NN[OutputTypePair <: { type Data; type Delta }] = Ast.FromTypePair[InputTypePair, OutputTypePair]

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

    implicit val row = input[Batch.FromTypePair[InputTypePair]]

    val toArray2D: NN[Array2D] = {

      val field0 = row.head
      val rest0 = row.tail
      val field1 = rest0.head
      val rest1 = rest0.tail
      val field2 = rest1.head
      val rest2 = rest1.tail
      val field3 = rest2.head
      val rest3 = rest2.tail

      val field0Flag0: NN[Double] = field0.choice { _ =>
        1.0
      } { _ =>
        0.0
      }

      val field0Flag1 = field0.choice { unknown =>
        0.5.toWeight
      } {
        _.choice { knownField0 =>
          knownField0.choice { unset =>
            1.0
          } { someValue =>
            0.0
          }
        } { cnil =>
          `throw`(new IllegalArgumentException)
        }
      }

      val field0Value0: NN[Double] = field0.choice { unknown: NN[HNil] =>
        0.5.toWeight: NN[Double]
      } {
        _.choice { knownField0 =>
          knownField0.choice { unset: NN[HNil] =>
            0.5.toWeight: NN[Double]
          } {
            _.choice { nativeDouble: NN[Double] =>
              nativeDouble: NN[Double]
            } { cnil: NN[CNil] =>
              `throw`(new IllegalArgumentException): NN[Double]
            }: NN[Double]
          }: NN[Double]
        } { cnil: NN[CNil] =>
          `throw`(new IllegalArgumentException): NN[Double]
        }: NN[Double]

      }

      val isField1Unknown = field1.isInl
      val field1Enum = field1.tail.head
      val isField1Case0 = field1Enum.isInl
      val isField1Case1 = field1Enum.tail.isInl

      val field1Flag0 = isField1Unknown.`if` {
        1.0
      } {
        0.0
      }

      val field1Value0: NN[Double] = isField1Unknown.`if` {
        0.5.toWeight
      } {
        isField1Case0.`if` {
          1.0
        } {
          0.0
        }
      }

      val field1Value1 = isField1Unknown.`if` {
        0.5.toWeight: NN[Double]
      } {
        isField1Case0.`if` {
          0.0: NN[Double]
        } {
          1.0: NN[Double]
        }: NN[Double]
      }

      val isField2Unknown = field2.isInl
      val field2Flag0 = isField2Unknown.`if` {
        1.0
      } {
        0.0
      }

      val field2Value0 = isField2Unknown.`if` {
        0.5.toWeight
      } {
        field2.tail.head
      }

      val isField3Unknown = field3.isInl
      val field3Enum = field3.tail.head
      val isField3Case0 = field3Enum.isInl
      val isField3Case1 = field3Enum.tail.isInl
      val field3Flag0 = isField3Unknown.`if` {
        1.0
      } {
        0.0
      }

      val field3Value0 = isField3Unknown.`if` {
        0.5.toWeight
      } {
        isField3Case0.`if` {
          1.0
        } {
          0.0
        }
      }

      val field3Value1 = isField3Unknown.`if` {
        0.5.toWeight
      } {
        isField3Case0.`if` {
          0.0
        } {
          1.0
        }
      }

      val field3Value2 = isField3Unknown.`if` {
        0.5.toWeight
      } {
        isField3Case0.`if` {
          0.0
        } {
          isField3Case1.`if` {
            0.0
          } {
            1.0
          }
        }
      }

      val encodedAstRow0 = Vector(field0Flag0,
                                  field0Flag1,
                                  field0Value0,
                                  field1Flag0,
                                  field1Value0,
                                  field1Value1,
                                  field2Flag0,
                                  field2Value0,
                                  field3Flag0,
                                  field3Value0,
                                  field3Value1,
                                  field3Value2)

      Vector(encodedAstRow0).toArray2D
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
