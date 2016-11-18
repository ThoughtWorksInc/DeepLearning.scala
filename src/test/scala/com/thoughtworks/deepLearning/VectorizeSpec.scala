package com.thoughtworks.deepLearning

import cats.Eval
import com.thoughtworks.deepLearning.any._
import org.scalatest._

import scala.language.implicitConversions
import scala.language.existentials
import Predef.{any2stringadd => _, _}
import shapeless._

import scala.util.Random

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class VectorizeSpec extends FreeSpec with Matchers {

  import FortuneTeller._

  "Convert HMatrix to Array2D" in {

    val trainingData = {
      import shapeless._
      import shapeless.ops.coproduct._
      IndexedSeq(
        Coproduct[Field0#Data](HNil: HNil) :: Inl(HNil) :: Eval.now(3.5) :: Inl(HNil) :: HNil,
        Coproduct[Field0#Data](Eval.now(5.1)) :: Inr(Inl(HNil)) :: Eval.now(8.3) :: Inr(Inl(HNil)) :: HNil,
        Coproduct[Field0#Data](HNil: HNil) :: Inl(HNil) :: Eval.now(91.3) :: Inr(Inr(Inl(HNil))) :: HNil
      )
    }

    def makeMinibatch = {
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
    val predictionData0 = {
      import shapeless._
      Inr(Inl(Coproduct[Field0#Data](HNil: HNil))) :: Inl(HNil) :: Inr(Inl(Eval.now(85.9))) :: Inl(HNil) :: HNil
    }

    for (i <- 0 until 100) {

      val (result0NullProbability :: result0Value :: HNil) :: (result1Case0Probability :: result1Case1Probability :: HNil) :: result2 :: (result3Case0Probability :: result3Case1Probability :: result3Case2Probability :: HNil) :: HNil =
        predictNetwork.predict(predictionData0)
//      println(predictNetwork)

      println(s"result0NullProbability: ${result0NullProbability.value}")
      println(s"result0Value: ${result0Value.value}")
      println(s"result1Case0Probability: ${result1Case0Probability.value}")
      println(s"result1Case1Probability: ${result1Case1Probability.value}")
      println(s"result2: ${result2.value}")
      println(s"result3Case0Probability: ${result3Case0Probability.value}")
      println(s"result3Case1Probability: ${result3Case1Probability.value}")
      println(s"result3Case2Probability: ${result3Case2Probability.value}")
      println()

      trainNetwork.train(makeMinibatch)
      def assertClear(layer: scala.Any): Unit = {
        layer match {
          case cached: BufferedLayer =>
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

    /*
     最终目标是生成一个预测神经网络和一个训练神经网络
     为了生成这两个网络，需要生成若干处理Array2D的全连接层、InputData到Array2D的转换、Array2D到Row的转换、Array2D到Double loss的转换

     InputData到Array2D的转换可以从InputData到若干Double的转换做起

     目前可以暂时使用HList而不是直接用case class的神经网络，将来可以直接使用case class

   */

  }

}
