package com.thoughtworks
package deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Layer.Tape._
import com.thoughtworks.deeplearning.DifferentiableHList._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableBoolean._
import com.thoughtworks.deeplearning.DifferentiableNothing._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import com.thoughtworks.deeplearning.DifferentiableSeq._
import com.thoughtworks.deeplearning.Symbolic._
import com.thoughtworks.deeplearning.DifferentiableCoproduct._
import com.thoughtworks.deeplearning.DifferentiableDouble.Optimizers.LearningRate
import org.scalatest._
import com.thoughtworks.deeplearning.Poly.MathOps
import com.thoughtworks.deeplearning.Poly.MathFunctions._

import language.implicitConversions
import language.existentials
import Predef.{any2stringadd => _, _}
import util.Random

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
@enableMembersIf(!scala.util.Properties.versionNumberString.startsWith("2.12."))
object FortuneTeller {
  import com.thoughtworks.deeplearning.DifferentiableINDArray._
  import org.nd4j.linalg.factory.Nd4j
  import org.nd4s.Implicits._

  type Seq2D = SeqPlaceholder[SeqPlaceholder[DoublePlaceholder]]

  type Nullable[A <: Placeholder[_, _]] = HNilPlaceholder :++: A :++: CNilPlaceholder

  type InputField[A <: Placeholder[_, _]] = HNilPlaceholder :++: A :++: CNilPlaceholder

  type LabelField[A <: Placeholder[_, _]] = HNilPlaceholder :++: A :++: CNilPlaceholder

  type Enum0 = HNilPlaceholder :++: HNilPlaceholder :++: CNilPlaceholder
  type Enum1 =
    HNilPlaceholder :++: HNilPlaceholder :++: HNilPlaceholder :++: CNilPlaceholder

  type Field0 = Nullable[DoublePlaceholder]
  type Field1 = Enum0
  type Field2 = DoublePlaceholder
  type Field3 = Enum1

  type InputTypePair =
    InputField[Nullable[DoublePlaceholder]] :**: InputField[Enum0] :**: InputField[DoublePlaceholder] :**: InputField[
      Enum1] :**: HNilPlaceholder

  type ExpectedLabel =
    LabelField[Nullable[DoublePlaceholder]] :**: LabelField[Enum0] :**: LabelField[DoublePlaceholder] :**: LabelField[
      Enum1] :**: HNilPlaceholder

  type UnsetProbability = DoublePlaceholder
  type NullableFieldPrediction[Value <: Placeholder[_, _]] =
    UnsetProbability :**: Value :**: HNilPlaceholder

  type Enum0Prediction = DoublePlaceholder :**: DoublePlaceholder :**: HNilPlaceholder
  type Enum1Prediction =
    DoublePlaceholder :**: DoublePlaceholder :**: DoublePlaceholder :**: HNilPlaceholder

  type PredictionResult =
    NullableFieldPrediction[DoublePlaceholder] :**: Enum0Prediction :**: DoublePlaceholder :**: Enum1Prediction :**: HNilPlaceholder

  implicit val optimizer = new DifferentiableINDArray.Optimizers.L2Regularization
  with DifferentiableINDArray.Optimizers.LearningRate with DifferentiableDouble.Optimizers.L2Regularization
  with DifferentiableDouble.Optimizers.LearningRate {
    override def currentLearningRate() = 0.0003

    override protected def l2Regularization = 0.1
  }

  def probabilityLoss(implicit x: DoublePlaceholder): x.To[DoublePlaceholder] = {
    0.5 + 0.5 / (1.0 - log(x)) - 0.5 / (1.0 - log(1.0 - x))
  }
  val probabilityLossNetwork = probabilityLoss
  def loss(implicit rowAndExpectedLabel: INDArrayPlaceholder :**: ExpectedLabel :**: HNilPlaceholder)
    : rowAndExpectedLabel.To[DoublePlaceholder] = {
    val row: rowAndExpectedLabel.To[INDArrayPlaceholder] = rowAndExpectedLabel.head
    val expectedLabel: rowAndExpectedLabel.To[ExpectedLabel] = rowAndExpectedLabel.tail.head
    val rowSeq: rowAndExpectedLabel.To[SeqPlaceholder[SeqPlaceholder[DoublePlaceholder]]] = row.toSeq

    // 暂时先在CPU上计算

    val expectedLabelField0: rowAndExpectedLabel.To[LabelField[Nullable[DoublePlaceholder]]] = expectedLabel.head
    val expectedLabelRest1 = expectedLabel.tail
    val expectedLabelField1 = expectedLabelRest1.head
    val expectedLabelRest2 = expectedLabelRest1.tail
    val expectedLabelField2 = expectedLabelRest2.head
    val expectedLabelRest3 = expectedLabelRest2.tail
    val expectedLabelField3 = expectedLabelRest3.head

    val loss0 = expectedLabelField0.choice { _ =>
      0.0 // Drop out
    } {
      _.head.choice { _ =>
//        probabilityLossNetwork.compose()
        probabilityLossNetwork.compose(min(exp(-rowSeq(0)(0)), 1.0))
//        max(1.0 - rowSeq(0)(0), 0.0)
      } { inr =>
        val expectedValue = inr.head
        (rowSeq(0)(0) + abs(rowSeq(0)(1) - expectedValue)): rowAndExpectedLabel.To[DoublePlaceholder]
      }
    }

    val loss1 = expectedLabelField1.choice { _ =>
      0.0 // Drop out
    } { expectedEnum =>
      val score0 = rowSeq(0)(2)
      val score1 = rowSeq(0)(3)
      val sum = score0 + score1 + 0.00000001
      val probability0 = score0 / sum
      val probability1 = score1 / sum
      expectedEnum.head.choice { _ =>
        1.0 - probability0
      } { _ =>
        1.0 - probability1
      }
    }

    val loss2 = expectedLabelField2.choice { _ =>
      0.0 // Drop out
    } { expectedDouble =>
      abs(expectedDouble.head - rowSeq(0)(4) + 1.0)
    }

    val loss3 = expectedLabelField3.choice { _ =>
      0.0 // Drop out
    } { expectedEnum =>
      val score0 = rowSeq(0)(5)
      val score1 = rowSeq(0)(6)
      val score2 = rowSeq(0)(7)
      val sum = score0 + score1 + score2 + 0.00000001
      val probability0 = score0 / sum
      val probability1 = score1 / sum
      val probability2 = score2 / sum
      expectedEnum.head.choice { _ =>
        1.0 - probability0
      } {
        _.choice { _ =>
          1.0 - probability1
        } { _ =>
          1.0 - probability2
        }
      }
    }

    loss0 + loss1 + loss2 + loss3

  }

  val lossNetwork = loss

  def array2DToRow(implicit input: INDArrayPlaceholder): input.To[PredictionResult] = {
    val rowSeq = input.toSeq
    val field0
      : input.To[DoublePlaceholder :**: DoublePlaceholder :**: HNilPlaceholder] = min(rowSeq(0)(0), 1.0) :: rowSeq(0)(
      1) :: shapeless.HNil.toLayer
    val field1: input.To[Enum0Prediction] = rowSeq(0)(2) :: rowSeq(0)(3) :: shapeless.HNil.toLayer
    val field2: input.To[DoublePlaceholder] = rowSeq(0)(4)
    val field3 = rowSeq(0)(5) :: rowSeq(0)(6) :: rowSeq(0)(7) :: shapeless.HNil.toLayer
    field0 :: field1 :: field2 :: field3 :: shapeless.HNil.toLayer
  }
  val array2DToRowNetwork = array2DToRow

  def rowToDifferentiableINDArray(implicit row: InputTypePair): row.To[INDArrayPlaceholder] = {
    val field0 = row.head
    val rest0 = row.tail
    val field1 = rest0.head
    val rest1 = rest0.tail
    val field2 = rest1.head
    val rest2 = rest1.tail
    val field3 = rest2.head
    val rest3 = rest2.tail

    val field0Flag0: row.To[DoublePlaceholder] = field0.choice { _ =>
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

    val field0Value0: row.To[DoublePlaceholder] = field0.choice { unknown: row.To[HNilPlaceholder] =>
      0.5.toWeight: row.To[DoublePlaceholder]
    } {
      _.choice { knownField0 =>
        knownField0.choice { unset: row.To[HNilPlaceholder] =>
          0.5.toWeight: row.To[DoublePlaceholder]
        } {
          _.choice { nativeDouble: row.To[DoublePlaceholder] =>
            nativeDouble: row.To[DoublePlaceholder]
          } { cnil: row.To[CNilPlaceholder] =>
            `throw`(new IllegalArgumentException): row.To[DoublePlaceholder]
          }: row.To[DoublePlaceholder]
        }: row.To[DoublePlaceholder]
      } { cnil: row.To[CNilPlaceholder] =>
        `throw`(new IllegalArgumentException)
      }: row.To[DoublePlaceholder]

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

    val field1Value0: row.To[DoublePlaceholder] = isField1Unknown.`if` {
      0.5.toWeight
    } {
      isField1Case0.`if` {
        1.0
      } {
        0.0
      }
    }

    val field1Value1 = isField1Unknown.`if` {
      0.5.toWeight: row.To[DoublePlaceholder]
    } {
      isField1Case0.`if` {
        0.0: row.To[DoublePlaceholder]
      } {
        1.0: row.To[DoublePlaceholder]
      }: row.To[DoublePlaceholder]
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

    val encodedLayerRow0 = Vector(
      field0Flag0,
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
      field3Value2
    )

    Vector(encodedLayerRow0).toINDArray
  }

  val rowToDifferentiableINDArrayNetwork = rowToDifferentiableINDArray

  def fullyConnectedThenRelu(inputSize: Int, outputSize: Int)(implicit row: INDArrayPlaceholder) = {
    val w = (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
//    val b = (Nd4j.randn(1, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    max((row dot w) + b, 0.0)
  }

  def hiddenLayers(implicit encodedInput: INDArrayPlaceholder) = {
    fullyConnectedThenRelu(50, 8).compose(
      fullyConnectedThenRelu(50, 50).compose(
        fullyConnectedThenRelu(50, 50).compose(fullyConnectedThenRelu(12, 50).compose(encodedInput))))
  }

  val hiddenLayersNetwork = hiddenLayers

  def predict(implicit input: InputTypePair): input.To[PredictionResult] = {
    val encodedInput = rowToDifferentiableINDArrayNetwork.compose(input)
    val encodedResult = hiddenLayersNetwork.compose(encodedInput)
    array2DToRowNetwork.compose(encodedResult)
  }

  val predictNetwork = predict

  def train(implicit input: :**:[InputTypePair, :**:[ExpectedLabel, HNilPlaceholder]]) = {
    val inputRow = input.head
    val expectedLabel = input.tail.head
    val encodedInput = rowToDifferentiableINDArrayNetwork.compose(inputRow)
    val encodedResult = hiddenLayersNetwork.compose(encodedInput)

    lossNetwork.compose(encodedResult :: expectedLabel :: shapeless.HNil.toLayer)
  }

  val trainNetwork = train

}
