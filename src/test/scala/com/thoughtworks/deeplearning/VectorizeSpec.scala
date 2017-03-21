package com.thoughtworks
package deeplearning

import cats.Eval
import com.thoughtworks.deeplearning.Symbolic._
import org.scalatest._
import com.thoughtworks.deeplearning.DifferentiableAny._
import com.thoughtworks.deeplearning.DifferentiableDouble._
import language.implicitConversions
import language.existentials
import Predef.{any2stringadd => _, _}
import shapeless.{::, HNil, _}

import util.Random

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
@enableMembersIf(!scala.util.Properties.versionNumberString.startsWith("2.12."))
final class VectorizeSpec extends FreeSpec with Matchers {

  import FortuneTeller._
  import com.thoughtworks.deeplearning.DifferentiableINDArray._

  "Convert HMatrix to DifferentiableINDArray" in {

    val trainingData = {
      import shapeless._
      import shapeless.ops.coproduct._
      IndexedSeq(
        Coproduct[Field0#Data](HNil: HNil) :: Inl(HNil) :: 3.5 :: Inl(HNil) :: HNil,
        Coproduct[Field0#Data](5.1) :: Inr(Inl(HNil)) :: 8.3 :: Inr(Inl(HNil)) :: HNil,
        Coproduct[Field0#Data](HNil: HNil) :: Inl(HNil) :: 91.3 :: Inr(Inr(Inl(HNil))) :: HNil
      )
    }

    def makeMinitape = {
      import shapeless._
      val field0 :: field1 :: field2 :: field3 :: HNil = trainingData(Random.nextInt(trainingData.length))

      def fieldPair[A](field: A) = {
        if (Random.nextBoolean) {
          (Inl(HNil), Inr(Inl(field)))
        } else {
          (Inr(Inl(field)), Inl(HNil))
        }
      }
      val (input0, label0) = fieldPair(field0)
      val (input1, label1) = fieldPair(field1)
      val (input2, label2) = fieldPair(field2)
      val (input3, label3) = fieldPair(field3)

      (input0 :: input1 :: input2 :: input3 :: HNil) :: (label0 :: label1 :: label2 :: label3 :: HNil) :: HNil
    }
    val predictionData0: InputTypePair#Data = {
      import shapeless._
      Inl(HNil: HNil) :: Inl(HNil: HNil) :: Inl(HNil: HNil) :: Inr(Inl(Inl(HNil: HNil))) :: HNil
    }
    val predictionData1: InputTypePair#Data = {
      import shapeless._
      Inl(HNil: HNil) :: Inl(HNil: HNil) :: Inl(HNil: HNil) :: Inr(Inl(Inr(Inl(HNil: HNil)))) :: HNil
    }
    val predictionData2: InputTypePair#Data = {
      import shapeless._
      Inl(HNil: HNil) :: Inl(HNil: HNil) :: Inl(HNil: HNil) :: Inr(Inl(Inr(Inr(Inl(HNil: HNil))))) :: HNil
    }
    def predictAndPrint(data: InputTypePair#Data) = {
      val (result0NullProbability :: result0Value :: HNil) :: (result1Case0Probability :: result1Case1Probability :: HNil) :: result2 :: (result3Case0Probability :: result3Case1Probability :: result3Case2Probability :: HNil) :: HNil =
        predictNetwork.predict(data)

//      println(s"result0NullProbability: ${result0NullProbability.value}")
//      println(s"result0Value: ${result0Value.value}")
//      println(s"result1Case0Probability: ${result1Case0Probability.value}")
//      println(s"result1Case1Probability: ${result1Case1Probability.value}")
//      println(s"result2: ${result2.value}")
//      println(s"result3Case0Probability: ${result3Case0Probability.value}")
//      println(s"result3Case1Probability: ${result3Case1Probability.value}")
//      println(s"result3Case2Probability: ${result3Case2Probability.value}")
//      println()
    }

    def predictField2(data: InputTypePair#Data): Double = {
      val (result0NullProbability :: result0Value :: HNil) :: (result1Case0Probability :: result1Case1Probability :: HNil) :: result2 :: (result3Case0Probability :: result3Case1Probability :: result3Case2Probability :: HNil) :: HNil =
        predictNetwork.predict(data)
      result2
    }
    for (i <- 0 until 20) {
//      println(predictNetwork)

      predictAndPrint(predictionData0)

      trainNetwork.train(makeMinitape)
      def assertClear(layer: Any): Unit = {
        layer match {
          case cached: CumulativeLayer =>
            cached.cache shouldBe empty
          case _ =>
        }
        layer match {
          case parent: Product =>
            for (upstreamLayer <- parent.productIterator) {
              assertClear(upstreamLayer)
            }
          case _ =>
        }
      }
      assertClear(trainNetwork)
    }

    val sample0Field2 = predictField2(predictionData0)
    val sample1Field2 = predictField2(predictionData1)
    val sample2Field2 = predictField2(predictionData2)
//    sample0Field2 should be < sample1Field2
//    sample1Field2 should be < sample2Field2
    /*
     最终目标是生成一个预测神经网络和一个训练神经网络
     为了生成这两个网络，需要生成若干处理DifferentiableINDArray的全连接层、InputData到DifferentiableINDArray的转换、DifferentiableINDArray到Row的转换、DifferentiableINDArray到Double loss的转换

     InputData到DifferentiableINDArray的转换可以从InputData到若干Double的转换做起

     目前可以暂时使用HList而不是直接用case class的神经网络，将来可以直接使用case class

   */

  }

}
