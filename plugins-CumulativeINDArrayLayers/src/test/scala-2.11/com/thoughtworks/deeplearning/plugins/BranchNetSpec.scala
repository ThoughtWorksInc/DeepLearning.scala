package com.thoughtworks.deeplearning.plugins

import org.scalatest.{FreeSpec, Inside, Matchers}
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.deeplearning.plugins.CNNObject.{Adagrad, CNNs, LearningRate}
import com.thoughtworks.deeplearning.plugins.Utils.findMaxItemIndex
import com.thoughtworks.deeplearning.plugins._
import com.thoughtworks.each.Monadic.{monadic, _}
import com.thoughtworks.feature.{Factory, ImplicitApply}
import com.thoughtworks.raii.asynchronous._

import collection.immutable.IndexedSeq
import scala.io.Source
import scala.concurrent.ExecutionContext.Implicits.global
import scalaz.std.iterable._
import scalaz.syntax.all._
import scalaz.std.stream._
import com.thoughtworks.future._

import scala.concurrent.Await
import scala.concurrent.duration.Duration
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.Nd4j.argMax
import org.nd4j.linalg.api.ndarray.INDArray
import com.thoughtworks.deeplearning.plugins.Operators._
import com.thoughtworks.deeplearning.plugins.ReadCIFARToNDArray.{TestImageAndLabels, TrainData}
import com.thoughtworks.feature.Factory
import org.joda.time.LocalTime

import scala.language.postfixOps
import scala.util.Random

import plotly.Scatter
import plotly.Plotly._
import plotly._

class BranchNetSpec extends FreeSpec with Matchers with Inside {
  "BranchNet" in {
    case class TrainResult(loss: Double, coarseAcc: Double, fineAcc: Double)

    case class PredictResult(coarseClass: Int, fineClass: Int)

    val learningRate = 0.000001

    val hyperparameters =
      Factory[
        Adagrad with LearningRate with Logging with ImplicitsSingleton with DoubleTraining with INDArrayTraining with INDArrayLiterals with DoubleLiterals with CumulativeDoubleLayers with Operators with CumulativeINDArrayLayers with CNNs]
        .newInstance(learningRate = learningRate, eps = 1e-8)

    import hyperparameters.INDArrayWeight
    import hyperparameters.DoubleLayer
    import hyperparameters.INDArrayLayer
    import hyperparameters.implicits._

    val NumberOfCoarseClasses: Int = 20
    val SubClassOfCoarse: Int = 5
    val NumberOfPixels: Int = 32 * 32 * 3
    val NumberOfTestSize = 500
    val NumberOfTrainSize = 50000
    val MiniBatchSize = 50
    val batchSize = MiniBatchSize * NumberOfCoarseClasses
    val InputSize = 32 // W 输入数据尺寸
    val Stride = 1 // 步长
    val Padding = 1 //零填充数量
    val KernelSize = 3 //F 卷积核的空间尺寸

    lazy val testData: IndexedSeq[TestImageAndLabels] =
      ReadCIFARToNDArray.readTestDataFromCIFAR100("/cifar-100-binary/test.bin", NumberOfTestSize)

    def initialWeightAndBias(kernelNumber: Int, depth: Int): (INDArrayWeight, INDArrayWeight) = {
      import org.nd4s.Implicits._
      (
        INDArrayWeight(
          Nd4j.randn(Array(kernelNumber, depth, KernelSize, KernelSize)) *
            math.sqrt(2.0 / depth / KernelSize / KernelSize)),
        //When using RELUs, make sure biases are initialised with small *positive* values for example 0.1
        INDArrayWeight(Nd4j.ones(kernelNumber) * 0.1)
      )
    }

    type INDArrayDeepLearning[A] = DeepLearning.Aux[A, INDArray, INDArray]

    def convolutionThenRelu[Input: INDArrayDeepLearning](input: Input,
                                                         weight: INDArrayWeight,
                                                         bias: INDArrayWeight): INDArrayLayer = {
      val convResult =
        hyperparameters.conv2d(input, weight, bias, (KernelSize, KernelSize), (Stride, Stride), (Padding, Padding))
      max(convResult, 0.0)
    }

    def softmax(scores: INDArrayLayer): INDArrayLayer = {
      val expScores = hyperparameters.exp(scores)
      expScores / expScores.sum(1)
    }

    val coarseNetWeightAndBias: IndexedSeq[(INDArrayWeight, INDArrayWeight)] =
      IndexedSeq(initialWeightAndBias(20, 3),
                 initialWeightAndBias(20, 20),
                 initialWeightAndBias(20, 20),
                 initialWeightAndBias(20, 20))

    val coarseBias = INDArrayWeight(Nd4j.zeros(NumberOfCoarseClasses))

    val coarseWeight = {
      import org.nd4s.Implicits._
      INDArrayWeight(
        Nd4j.randn(20 * 16 * 16, NumberOfCoarseClasses) / math
          .sqrt(20))
    }

    val fineNetWeightAndBiases: IndexedSeq[IndexedSeq[(INDArrayWeight, INDArrayWeight)]] =
      for (_ <- 0 until 20)
        yield IndexedSeq(initialWeightAndBias(20, 20), initialWeightAndBias(20, 20))

    val fineBiases = for (_ <- 0 until 20)
      yield INDArrayWeight(Nd4j.zeros(SubClassOfCoarse))

    val fineWeights = {
      import org.nd4s.Implicits._
      for (_ <- 0 until 20)
        yield
          INDArrayWeight(
            Nd4j.randn(20 * 16 * 16, SubClassOfCoarse) / math
              .sqrt(20))
    }

    def coarseSubNet(images: INDArray): INDArrayLayer = {
      val layer0 = convolutionThenRelu(images, coarseNetWeightAndBias(0)._1, coarseNetWeightAndBias(0)._2)
      val layer1 = convolutionThenRelu(layer0, coarseNetWeightAndBias(1)._1, coarseNetWeightAndBias(1)._2)
      val layer2 = convolutionThenRelu(layer1, coarseNetWeightAndBias(2)._1, coarseNetWeightAndBias(2)._2)

      convolutionThenRelu(layer2, coarseNetWeightAndBias(3)._1, coarseNetWeightAndBias(3)._2)
    }

    def fineSubNet(coarseClass: Int, features: INDArrayLayer): INDArrayLayer = {
      val weightAndBias = fineNetWeightAndBiases(coarseClass)
      val layer0 =
        convolutionThenRelu(features, weightAndBias(0)._1, weightAndBias(0)._2)
      val layer1 =
        convolutionThenRelu(layer0, weightAndBias(1)._1, weightAndBias(1)._2)
      val poolSize = (2, 2)
      hyperparameters.maxPool(layer1, poolSize)
    }

    def fullyConnectedLayer(input: INDArrayLayer,
                            weight: INDArrayWeight,
                            bias: INDArrayWeight,
                            imageCount: Int,
                            pixelsOfPreLayer: Int): INDArrayLayer = {
      input
        .reshape(imageCount, pixelsOfPreLayer) dot weight + bias
    }

    def coarseMaxPoolThenFC(features: INDArrayLayer, imageCount: Int): INDArrayLayer = {
      // N * 20 * 16 * 16
      val layer0 = hyperparameters.maxPool(features, (2, 2))
      // N * 20
      fullyConnectedLayer(layer0, coarseWeight, coarseBias, imageCount, 20 * 16 * 16)
    }

    def lossFunction(possibility: INDArrayLayer, expectedLabel: INDArray): DoubleLayer = {
      -(hyperparameters
        .log(possibility * 0.9 + 0.1) * expectedLabel + (1.0 - expectedLabel) * hyperparameters
        .log(1.0 - possibility * 0.9)).mean
    }

    def fineFCThenSoftmax(features: INDArrayLayer,
                          coarseClass: Int,
                          imageCount: Int,
                          pixelsOfPreLayer: Int): INDArrayLayer = {
      val layer0 =
        fullyConnectedLayer(features, fineWeights(coarseClass), fineBiases(coarseClass), imageCount, pixelsOfPreLayer)
      softmax(layer0)
    }

    def calculateAcc(coarsePossibility: INDArray,
                     finePossibility: INDArray,
                     coarseClass: Int,
                     fineClasses: INDArray,
                     imageCount: Int): (Double, Double) = {
      import org.nd4s.Implicits._
      val coarseScoreIndex = findMaxItemIndex(coarsePossibility)
      val coarseSeq =
        for (row <- 0 until imageCount) yield {
          val predictClass = coarseScoreIndex.getDouble(row, 0)
          if (predictClass == coarseClass) {
            1.0
          } else 0.0
        }
      val coarseAcc = coarseSeq.sum / imageCount

      val predictFineResult = findMaxItemIndex(finePossibility)
      val expectFineResult = findMaxItemIndex(fineClasses)
      val fineAccINDArray =
        predictFineResult.eq(expectFineResult).reshape(imageCount, 1)

      val coarseAccINDArray = coarseSeq.toNDArray.reshape(imageCount, 1)
      val realFineAccINDArray = fineAccINDArray * coarseAccINDArray
      val fineAcc = realFineAccINDArray.sumT / imageCount

      (coarseAcc, fineAcc)
    }

    def train(trainData: TrainData): Future[TrainResult] = {
      @monadic[Do]
      def doTrain(images: INDArray,
                  fineClasses: INDArray,
                  coarseClasses: INDArray,
                  coarseClass: Int): Do[TrainResult] = {
        val imageCount = images.shape()(0)
        // N * 20 * 32 * 32
        val coarseCNNOutput = coarseSubNet(images)
        // N * 20
        val coarseSoftMaxOutput = softmax(coarseMaxPoolThenFC(coarseCNNOutput, imageCount))
        val coarseLoss =
          lossFunction(coarseSoftMaxOutput, coarseClasses)

        val fineOutput = fineSubNet(coarseClass, coarseCNNOutput)
        val fineSoftMaxOutput =
          fineFCThenSoftmax(fineOutput, coarseClass, imageCount, 20 * 16 * 16)
        val fineLoss = lossFunction(fineSoftMaxOutput, fineClasses)
        val lossLayer = coarseLoss + fineLoss

        val lossTape: Tape[Double, Double] = lossLayer.forward.each

        val backwardFuture = lossTape.backward(Do.now(1.0))
        Do.garbageCollected(backwardFuture).each

        val loss: Double = lossTape.data

        val coarsePossibility = coarseSoftMaxOutput.forward.each.data
        val finePossibility = fineSoftMaxOutput.forward.each.data

        val acc = calculateAcc(coarsePossibility, finePossibility, coarseClass, fineClasses, imageCount)

        TrainResult(loss, acc._1, acc._2)
      }

      doTrain(trainData.image, trainData.fineLabel, trainData.coarseLabel, trainData.coarseClass).run
    }

    def train0(trainData: TrainData): Future[Double] = {

      @monadic[Do]
      def doTrain(images: INDArray, fineClasses: INDArray, coarseClasses: INDArray, coarseClass: Int): Do[Double] = {
        val imageCount = images.shape()(0)
        // N * 20 * 32 * 32
        val coarseCNNOutput = coarseSubNet(images)
        // N * 20
        val coarseSoftMaxOutput = softmax(coarseMaxPoolThenFC(coarseCNNOutput, imageCount))
        val coarseLoss =
          lossFunction(coarseSoftMaxOutput, coarseClasses)

        val fineOutput = fineSubNet(coarseClass, coarseCNNOutput)
        val fineSoftMaxOutput =
          fineFCThenSoftmax(fineOutput, coarseClass, imageCount, 20 * 16 * 16)
        val fineLoss = lossFunction(fineSoftMaxOutput, fineClasses)
        val lossLayer = coarseLoss + fineLoss

        val lossTape: Tape[Double, Double] = lossLayer.forward.each

        val backwardFuture = lossTape.backward(Do.now(1.0))
        Do.garbageCollected(backwardFuture).each

        lossTape.data
      }

      doTrain(trainData.image, trainData.fineLabel, trainData.coarseLabel, trainData.coarseClass).run
    }

    def predict(images: INDArray): Future[PredictResult] = {

      @monadic[Do]
      def doPredict(images: INDArray, imageCount: Int): Do[PredictResult] = {
        import org.nd4s.Implicits._
        val coarseCNNOutput = coarseSubNet(images)
        val coarseFCOutput = coarseMaxPoolThenFC(coarseCNNOutput, imageCount)
        val coarseClass: Int =
          argMax(coarseFCOutput.forward.each.data, 1).get(0).toInt

        val fineOutput = fineSubNet(coarseClass, coarseCNNOutput)
        val fineSoftMaxOutput =
          fineFCThenSoftmax(fineOutput, coarseClass, imageCount, 20 * 16 * 16)
        val fineClass: Int =
          argMax(fineSoftMaxOutput.forward.each.data, 1).get(0).toInt
        PredictResult(coarseClass, fineClass)
      }

      val imageCount = images.shape()(0)

      doPredict(images, imageCount).run
    }

    var trainResultSeq: IndexedSeq[TrainResult] = IndexedSeq.empty

    @monadic[Future]
    val trainTask: Future[Unit] = {
      val lossStream: Stream[TrainResult] =
        (for (_ <- (0 until 200).toStream) yield {
          val randomIndex = Utils.getRandomIndex(NumberOfTrainSize)
          (for (times <- (0 until NumberOfTrainSize / batchSize).toStream) yield {
            val slicedIndexArray =
              Utils.sliceIndexArray(randomIndex, times, MiniBatchSize)
            val trainData =
              ReadCIFARToNDArray.processSGDTrainData(slicedIndexArray)
            for (index <- trainData.indices.toStream) yield {
              val currentInput = trainData(index)
              val futureTrainResult: Future[TrainResult] = train(currentInput)
              val trainResult = futureTrainResult.each
              println("loss:" + trainResult.loss)
              println("coarseAcc:" + trainResult.coarseAcc)
              println("fineAcc:" + trainResult.fineAcc)
              trainResult
            }
          }).flatten
        }).flatten
      trainResultSeq = IndexedSeq.concat(lossStream)
    }

    var lossSeq: IndexedSeq[Double] = IndexedSeq.empty

    @monadic[Future]
    val trainTask0: Future[Unit] = {
      val lossStream: Stream[Double] =
        (for (_ <- (0 until 500).toStream) yield {
          System.gc()
          val randomIndex = Utils.getRandomIndex(NumberOfTrainSize)
          (for (times <- (0 until NumberOfTrainSize / batchSize).toStream) yield {
            val slicedIndexArray =
              Utils.sliceIndexArray(randomIndex, times, MiniBatchSize)
            val trainData =
              ReadCIFARToNDArray.processSGDTrainData(slicedIndexArray)
            for (index <- trainData.indices.toStream) yield {
              val currentInput = trainData(index)
              val futureTrainResult: Future[Double] = train0(currentInput)
              val loss = futureTrainResult.each
              println("loss:" + loss)

              if (loss isNaN) {
                sys.exit(-11111111)
              }

              loss
            }
          }).flatten
        }).flatten
      lossSeq = IndexedSeq.concat(lossStream)
    }

    val startTime = LocalTime.now()
    //Await.result(trainTask.toScalaFuture, Duration.Inf)
    Await.result(trainTask0.toScalaFuture, Duration.Inf)
    val endTime = LocalTime.now()

    //  private val resultTuple
    //    : (IndexedSeq[Double], IndexedSeq[Double], IndexedSeq[Double]) =
    //    trainResultSeq.unzip3(result =>
    //      (result.loss, result.coarseAcc, result.fineAcc))
    //
    //  Seq(
    //    Scatter(trainResultSeq.indices, resultTuple._1, name = "loss"),
    //    Scatter(trainResultSeq.indices, resultTuple._2, name = "coarseAcc"),
    //    Scatter(trainResultSeq.indices, resultTuple._3, name = "fineAcc")
    //  ).plot(title = "loss & acc by time")

    def calculatePredictAcc(): (Double, Double) = {

      def calculatePredictResult: IndexedSeq[(Int, Int)] = {
        for (testItem <- testData)
          yield {
            val futurePredictResult: Future[PredictResult] = predict(testItem.imageData)
            val PredictResult(predictCoarse, predictFine) =
              Await.result(futurePredictResult.toScalaFuture, Duration.Inf)
            if (predictCoarse == testItem.coarse) {
              (1, Utils.isSameThenOne(predictFine, testItem.fine))
            } else {
              (0, 0)
            }
          }
      }

      val predictResult: IndexedSeq[(Int, Int)] = calculatePredictResult
      val resultTuple: (IndexedSeq[Int], IndexedSeq[Int]) = predictResult.unzip
      val total = resultTuple._1.size
      (resultTuple._1.sum / total, resultTuple._2.sum / total)
    }

    val (coarseAcc, fineAcc) = calculatePredictAcc()

    println(s"The coarse accuracy is $coarseAcc ,The fine accuracy is $fineAcc , start at $startTime , end at $endTime")

    Seq(
      Scatter(lossSeq.indices, lossSeq, name = "loss")
    ).plot(title = s"loss by time - learningRate : $learningRate, start at $startTime , end at $endTime")
  }
}
