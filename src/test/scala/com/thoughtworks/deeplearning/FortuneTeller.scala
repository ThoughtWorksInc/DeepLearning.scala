package com.thoughtworks.deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.Layer._
import com.thoughtworks.deeplearning.Layer.Batch._
import com.thoughtworks.deeplearning.BpHList._
import com.thoughtworks.deeplearning.BpAny._
import com.thoughtworks.deeplearning.BpBoolean._
import com.thoughtworks.deeplearning.BpNothing._
import com.thoughtworks.deeplearning.BpDouble._
import com.thoughtworks.deeplearning.BpSeq._
import com.thoughtworks.deeplearning.Bp2DArray._
import com.thoughtworks.deeplearning.Conversion._
import com.thoughtworks.deeplearning.Bp2DArray.Optimizers.LearningRate
import com.thoughtworks.deeplearning.BpCoproduct._
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
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
object FortuneTeller {

  type Seq2D = BpSeq[BpSeq[BpDouble]]

  type Nullable[A <: BackPropagationType[_, _]] = BpHNil :++: A :++: BpCNil

  type InputField[A <: BackPropagationType[_, _]] = BpHNil :++: A :++: BpCNil

  type LabelField[A <: BackPropagationType[_, _]] = BpHNil :++: A :++: BpCNil

  type Enum0 = BpHNil :++: BpHNil :++: BpCNil
  type Enum1 = BpHNil :++: BpHNil :++: BpHNil :++: BpCNil

  type Field0 = Nullable[BpDouble]
  type Field1 = Enum0
  type Field2 = BpDouble
  type Field3 = Enum1

  type InputTypePair =
    InputField[Nullable[BpDouble]] :**: InputField[Enum0] :**: InputField[BpDouble] :**: InputField[Enum1] :**: BpHNil

  type ExpectedLabel =
    LabelField[Nullable[BpDouble]] :**: LabelField[Enum0] :**: LabelField[BpDouble] :**: LabelField[Enum1] :**: BpHNil

  type UnsetProbability = BpDouble
  type NullableFieldPrediction[Value <: BackPropagationType[_, _]] = UnsetProbability :**: Value :**: BpHNil

  type Enum0Prediction = BpDouble :**: BpDouble :**: BpHNil
  type Enum1Prediction = BpDouble :**: BpDouble :**: BpDouble :**: BpHNil

  type PredictionResult =
    NullableFieldPrediction[BpDouble] :**: Enum0Prediction :**: BpDouble :**: Enum1Prediction :**: BpHNil

  implicit val optimizer = new Bp2DArray.Optimizers.L2Regularization with BpDouble.Optimizers.L2Regularization {
    override def currentLearningRate() = 0.0003

    override protected def l2Regularization = 0.1
  }

  def probabilityLoss(implicit x: BpDouble): x.To[BpDouble] = {
    0.5 + 0.5 / (1.0 - log(x)) - 0.5 / (1.0 - log(1.0 - x))
  }
  val probabilityLossNetwork = probabilityLoss
  def loss(implicit rowAndExpectedLabel: Bp2DArray :**: ExpectedLabel :**: BpHNil): rowAndExpectedLabel.To[BpDouble] = {
    val row: rowAndExpectedLabel.To[Bp2DArray] = rowAndExpectedLabel.head
    val expectedLabel: rowAndExpectedLabel.To[ExpectedLabel] = rowAndExpectedLabel.tail.head
    val rowSeq: rowAndExpectedLabel.To[BpSeq[BpSeq[BpDouble]]] = row.toSeq

    // 暂时先在CPU上计算

    val expectedLabelField0: rowAndExpectedLabel.To[LabelField[Nullable[BpDouble]]] = expectedLabel.head
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
        (rowSeq(0)(0) + abs(rowSeq(0)(1) - expectedValue)): rowAndExpectedLabel.To[BpDouble]
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

  def array2DToRow(implicit input: Bp2DArray): input.To[PredictionResult] = {
    val rowSeq = input.toSeq
    val field0: input.To[BpDouble :**: BpDouble :**: BpHNil] = min(rowSeq(0)(0), 1.0) :**: rowSeq(0)(1) :**: BpHNil
    val field1: input.To[Enum0Prediction] = rowSeq(0)(2) :**: rowSeq(0)(3) :**: BpHNil
    val field2: input.To[BpDouble] = rowSeq(0)(4)
    val field3 = rowSeq(0)(5) :**: rowSeq(0)(6) :**: rowSeq(0)(7) :**: BpHNil
    field0 :**: field1 :**: field2 :**: field3 :**: BpHNil
  }
  val array2DToRowNetwork = array2DToRow

  def rowToBp2DArray(implicit row: InputTypePair): row.To[Bp2DArray] = {
    val field0 = row.head
    val rest0 = row.tail
    val field1 = rest0.head
    val rest1 = rest0.tail
    val field2 = rest1.head
    val rest2 = rest1.tail
    val field3 = rest2.head
    val rest3 = rest2.tail

    val field0Flag0: row.To[BpDouble] = field0.choice { _ =>
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

    val field0Value0: row.To[BpDouble] = field0.choice { unknown: row.To[BpHNil] =>
      0.5.toWeight: row.To[BpDouble]
    } {
      _.choice { knownField0 =>
        knownField0.choice { unset: row.To[BpHNil] =>
          0.5.toWeight: row.To[BpDouble]
        } {
          _.choice { nativeDouble: row.To[BpDouble] =>
            nativeDouble: row.To[BpDouble]
          } { cnil: row.To[BpCNil] =>
            `throw`(new IllegalArgumentException): row.To[BpDouble]
          }: row.To[BpDouble]
        }: row.To[BpDouble]
      } { cnil: row.To[BpCNil] =>
        `throw`(new IllegalArgumentException)
      }: row.To[BpDouble]

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

    val field1Value0: row.To[BpDouble] = isField1Unknown.`if` {
      0.5.toWeight
    } {
      isField1Case0.`if` {
        1.0
      } {
        0.0
      }
    }

    val field1Value1 = isField1Unknown.`if` {
      0.5.toWeight: row.To[BpDouble]
    } {
      isField1Case0.`if` {
        0.0: row.To[BpDouble]
      } {
        1.0: row.To[BpDouble]
      }: row.To[BpDouble]
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

    val encodedLayerRow0 = Vector(field0Flag0,
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

    Vector(encodedLayerRow0).toBp2DArray
  }

  val rowToBp2DArrayNetwork = rowToBp2DArray

  def fullyConnectedThenRelu(inputSize: Int, outputSize: Int)(implicit row: Bp2DArray) = {
    val w = (Nd4j.randn(inputSize, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
//    val b = (Nd4j.randn(1, outputSize) / math.sqrt(outputSize / 2.0)).toWeight
    val b = Nd4j.zeros(outputSize).toWeight
    max((row dot w) + b, 0.0)
  }

  def hiddenLayers(implicit encodedInput: Bp2DArray) = {
    fullyConnectedThenRelu(50, 8).compose(
      fullyConnectedThenRelu(50, 50).compose(
        fullyConnectedThenRelu(50, 50).compose(fullyConnectedThenRelu(12, 50).compose(encodedInput))))
  }

  val hiddenLayersNetwork = hiddenLayers

  def predict(implicit input: InputTypePair): input.To[PredictionResult] = {
    val encodedInput = rowToBp2DArrayNetwork.compose(input)
    val encodedResult = hiddenLayersNetwork.compose(encodedInput)
    array2DToRowNetwork.compose(encodedResult)
  }

  val predictNetwork = predict

  def train(implicit input: InputTypePair :**: ExpectedLabel :**: BpHNil) = {
    val inputRow = input.head
    val expectedLabel = input.tail.head
    val encodedInput = rowToBp2DArrayNetwork.compose(inputRow)
    val encodedResult = hiddenLayersNetwork.compose(encodedInput)

    lossNetwork.compose(encodedResult :**: expectedLabel :**: BpHNil)
  }

  val trainNetwork = train

}
