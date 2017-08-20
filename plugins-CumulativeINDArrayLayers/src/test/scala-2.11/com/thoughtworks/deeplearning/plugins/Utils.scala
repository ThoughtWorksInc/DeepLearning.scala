package com.thoughtworks.deeplearning.plugins
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.collection.GenTraversable
import scala.collection.generic.GenericTraversableTemplate
import scala.collection.immutable.IndexedSeq
import scala.reflect.ClassTag
import scala.util.Random
object Utils {

  /**
    * 处理标签数据：将N行一列的NDArray转换为N行NumberOfClasses列的NDArray，每行对应的正确分类的值为1，其它列的值为0
    *
    * @param ndArray 标签数据
    * @return N行NumberOfClasses列的NDArray
    */
  def makeVectorized(ndArray: INDArray, numberOfClasses: Int): INDArray = {
    val shape = ndArray.shape()
    val imageCount = shape(0)
    val p = Nd4j.zeros(imageCount, numberOfClasses)
    for (i <- 0 until imageCount) {
      val double = ndArray.getDouble(i, 0)
      p.put(i, double.toInt, 1)
    }
    p
  }

  def makeDyVectorized(ndArray: INDArray, numberOfClasses: Int): INDArray = {
    val shape = ndArray.shape()
    val imageCount = shape(0)
    val p = Nd4j.zeros(imageCount, numberOfClasses)
    for (i <- 0 until imageCount) {
      val double = ndArray.getDouble(i, 0)
      p.put(i, double.toInt, 1)
    }
    p
  }

  /**
    * 二维INDArray中获得值最大的元素所在的列组成的INDArray
    *
    * @param iNDArray
    * @return
    */
  def findMaxItemIndex(iNDArray: INDArray): INDArray = {
    Nd4j.argMax(iNDArray, 1)
  }

  /**
    * 计算准确率
    * @param score 预测结果
    * @param testExpectLabel 期望结果
    * @return 准确率
    */
  def getAccuracy(score: INDArray, testExpectLabel: INDArray, numberOfClass: Int): Double = {
    val scoreIndex = findMaxItemIndex(score)
    if (testExpectLabel.shape().toSeq.last == 1) { //not vectorized
      val acc = for (row <- 0 until scoreIndex.shape()(0)) yield {
        if (scoreIndex.getDouble(row, 0) ==
              testExpectLabel.getDouble(row, 0)) {
          1.0
        } else 0.0
      }
      (acc.sum / score.shape()(0)) * 100
    } else if (testExpectLabel
                 .shape()
                 .toSeq
                 .last == numberOfClass) { //vectorized
      val expectResultIndex = findMaxItemIndex(testExpectLabel)
      val accINDArray = scoreIndex.eq(expectResultIndex)
      (accINDArray.sumT / score.shape()(0)) * 100
    } else
      throw new IllegalArgumentException("Unacceptable testExpectLabel")
  }

  class Unzipper[A, CC[X] <: GenTraversable[X]](s: GenericTraversableTemplate[A, CC]) {
    def unzip4[A1, A2, A3, A4](implicit asQuad: A => (A1, A2, A3, A4)): (CC[A1], CC[A2], CC[A3], CC[A4]) = {
      val b1 = s.genericBuilder[A1]
      val b2 = s.genericBuilder[A2]
      val b3 = s.genericBuilder[A3]
      val b4 = s.genericBuilder[A4]
      for (e <- s) {
        val (a, b, c, d) = asQuad(e)
        b1 += a
        b2 += b
        b3 += c
        b4 += d
      }
      (b1.result, b2.result, b3.result, b4.result)
    }
  }

  implicit def toUnzipper[A, CC[X] <: GenTraversable[X]](s: GenericTraversableTemplate[A, CC]): Unzipper[A, CC] =
    new Unzipper(s)

  class Tuple2ToArray[A: ClassTag](tuple: (A, A)) {
    def toArray: Array[A] = {
      val (one: A, two: A) = tuple
      Array(one, two)
    }
  }

  implicit def tupleTwoToArray[A: ClassTag](tuple: (A, A)): Tuple2ToArray[A] =
    new Tuple2ToArray[A](tuple)

  def isSameThenOne(expected: Int, given: Int): Int = {
    if (expected == given) {
      1
    } else {
      0
    }
  }

  def sliceIndexArray(randomIndex: Array[Int], times: Int, MiniBatchSize: Int): Array[Int] = {
    randomIndex.slice(times * MiniBatchSize, (times + 1) * MiniBatchSize)
  }

  def getRandomIndex(range: Int): Array[Int] = {
    new Random()
      .shuffle[Int, IndexedSeq](0 until range) //https://issues.scala-lang.org/browse/SI-6948
      .toArray
  }

}
