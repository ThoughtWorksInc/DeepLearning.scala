package com.thoughtworks.deeplearning.benchmark

import java.util.concurrent.{ExecutorService, Executors}

import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.etl.Cifar100
import com.thoughtworks.deeplearning.etl.Cifar100.Batch
import com.thoughtworks.deeplearning.plugins.Builtins
import com.thoughtworks.feature.Factory
import org.openjdk.jmh.annotations._
import com.thoughtworks.future._
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{ActivationLayer, DenseLayer, LossLayer, OutputLayer}
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.graph.{ElementWiseVertex, MergeVertex, StackVertex}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, MultiDataSet}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.ops.transforms.Transforms

import scala.concurrent.{ExecutionContext, ExecutionContextExecutorService}

/**
  * @author 杨博 (Yang Bo)
  */
object benchmark {

  import $exec.`https://gist.github.com/Atry/1fb0608c655e3233e68b27ba99515f16/raw/39ba06ee597839d618f2fcfe9526744c60f2f70a/FixedLearningRate.sc`

  trait LayerOutput {
    def numberOfFeatures: Int
    type Output
    def output: Output
    def typeClassInstance: DeepLearning.Aux[Output, INDArray, INDArray]
  }
  object LayerOutput {
    def input(indArray: INDArray): LayerOutput = new LayerOutput {
      def numberOfFeatures: Int = indArray.shape().apply(1)

      type Output = INDArray
      def output = indArray

      def typeClassInstance: DeepLearning.Aux[INDArray, INDArray, INDArray] = ???
    }
  }

  @Threads(value = 1)
  @State(Scope.Benchmark)
  class BranchNetBenchmark {

    private def deeplearning4jConf = {

      val builder = new NeuralNetConfiguration.Builder()
        .updater(Updater.SGD)
        .learningRate(1.0)
        .graphBuilder
        .addInputs("input")

      for (i <- 0 until numberOfBranches) {
        builder
          .addLayer(
            s"coarse${i}_dense0",
            new DenseLayer.Builder()
              .activation(Activation.RELU)
              .nIn(Cifar100.NumberOfPixelsPerSample)
              .nOut(numberOfHiddenFeatures)
              .build,
            "input"
          )
          .addLayer(
            s"coarse${i}_dense1",
            new DenseLayer.Builder()
              .activation(Activation.RELU)
              .nIn(numberOfHiddenFeatures)
              .nOut(numberOfHiddenFeatures)
              .build,
            s"coarse${i}_dense0"
          )
      }

      builder
        .addVertex("fusion",
                   new ElementWiseVertex(ElementWiseVertex.Op.Add),
                   (for (i <- 0 until numberOfBranches) yield s"coarse${i}_dense1"): _*)
        .addLayer(
          "coarse_probabilities",
          new DenseLayer.Builder()
            .activation(Activation.SOFTMAX)
            .nIn(numberOfHiddenFeatures)
            .nOut(Cifar100.NumberOfCoarseClasses)
            .build,
          "fusion"
        )
        .addLayer("coarse_loss", new LossLayer.Builder(LossFunction.MCXENT).build(), "coarse_probabilities")

      for (i <- 0 until Cifar100.NumberOfCoarseClasses) {
        builder
          .addLayer(
            s"fine${i}_dense0",
            new DenseLayer.Builder()
              .activation(Activation.RELU)
              .nIn(numberOfHiddenFeatures)
              .nOut(numberOfHiddenFeatures)
              .build,
            "fusion"
          )
          .addLayer(
            s"fine${i}_dense1",
            new DenseLayer.Builder()
              .activation(Activation.RELU)
              .nIn(numberOfHiddenFeatures)
              .nOut(numberOfHiddenFeatures)
              .build,
            s"fine${i}_dense0"
          )
          .addLayer(
            s"fine${i}_scores",
            new DenseLayer.Builder()
              .activation(Activation.IDENTITY)
              .nIn(numberOfHiddenFeatures)
              .nOut(Cifar100.NumberOfFineClassesPerCoarseClass)
              .build,
            s"fine${i}_dense1"
          )
      }

      builder
        .addVertex("fine_stack", new StackVertex(), (for (i <- 0 until Cifar100.NumberOfCoarseClasses) yield s"fine${i}_scores"): _*)
//        .addLayer("fine_probabilities",
//                  new ActivationLayer.Builder().activation(Activation.SOFTMAX).build(),
//                  "fine_stack")
        .addLayer("fine_loss",
                  new LossLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX).build(),
                  "fine_stack")
        .setOutputs("coarse_loss", "fine_loss")
        .build
    }

    private var computationGraph: ComputationGraph = _
    @Param(Array("8", "16"))
    protected var batchSize: Int = _

    @Param(Array("1", "2", "4"))
    protected var sizeOfThreadPool: Int = _

    @Param(Array("16", "32", "64"))
    protected var numberOfHiddenFeatures: Int = _

    @Param(Array("16", "8", "4", "2", "1"))
    protected var numberOfBranches: Int = _

    private implicit var executionContext: ExecutionContextExecutorService = _

    private lazy val batches = {
      val cifar100: Cifar100 = Cifar100.load().blockingAwait
      Iterator.continually(cifar100.epochByCoarseClass(batchSize)).flatten
    }

    class Model {
      val hyperparameters = Factory[Builtins with FixedLearningRate].newInstance(learningRate = 0.0001)

      import hyperparameters._, implicits._

      object CoarseFeatures extends (INDArray => INDArrayLayer) {

        val branches = Seq.fill(numberOfBranches)(new (INDArray => INDArrayLayer) {
          object Dense1 extends (INDArray => INDArrayLayer) {
            val weight = INDArrayWeight(Nd4j.randn(Cifar100.NumberOfPixelsPerSample, numberOfHiddenFeatures))
            val bias = INDArrayWeight(Nd4j.randn(1, numberOfHiddenFeatures))

            def apply(input: INDArray) = {
              max(input dot weight + bias, 0.0)
            }
          }

          val weight = INDArrayWeight(Nd4j.randn(numberOfHiddenFeatures, numberOfHiddenFeatures))
          val bias = INDArrayWeight(Nd4j.randn(1, numberOfHiddenFeatures))

          def apply(input: INDArray) = {
            max(Dense1(input) dot weight + bias, 0.0)
          }
        })

        def apply(input: INDArray) = {
          branches.map(_.apply(input)).reduce(_ + _)
        }
      }

      object CoarseProbabilityModel {
        val weight = INDArrayWeight(Nd4j.randn(numberOfHiddenFeatures, Cifar100.NumberOfCoarseClasses))
        val bias = INDArrayWeight(Nd4j.randn(1, Cifar100.NumberOfCoarseClasses))

        def apply(input: INDArrayLayer) = {
          val scores = input dot weight + bias

          val expScores = exp(scores)
          expScores / expScores.sum(1)
        }
      }

      val fineScoreModels = Seq.fill(Cifar100.NumberOfCoarseClasses)(new (INDArrayLayer => INDArrayLayer) {
        object Dense2 extends (INDArrayLayer => INDArrayLayer) {

          object Dense1 extends (INDArrayLayer => INDArrayLayer) {
            val weight = INDArrayWeight(Nd4j.randn(numberOfHiddenFeatures, numberOfHiddenFeatures))
            val bias = INDArrayWeight(Nd4j.randn(1, numberOfHiddenFeatures))

            def apply(coarseFeatures: INDArrayLayer) = {
              max(coarseFeatures dot weight + bias, 0.0)
            }
          }

          val weight = INDArrayWeight(Nd4j.randn(numberOfHiddenFeatures, numberOfHiddenFeatures))
          val bias = INDArrayWeight(Nd4j.randn(1, numberOfHiddenFeatures))

          def apply(coarseFeatures: INDArrayLayer) = {
            max(Dense1(coarseFeatures) dot weight + bias, 0.0)
          }
        }

        val weight = INDArrayWeight(Nd4j.randn(numberOfHiddenFeatures, Cifar100.NumberOfFineClassesPerCoarseClass))
        val bias = INDArrayWeight(Nd4j.randn(1, Cifar100.NumberOfFineClassesPerCoarseClass))

        def apply(coarseFeatures: INDArrayLayer) = {
          Dense2(coarseFeatures) dot weight + bias
        }
      })

      def loss(expectedCoarseLabel: Int, batch: Batch, excludeUnmatchedFineGrainedNetwork: Boolean): DoubleLayer = {
        def crossEntropy(prediction: INDArrayLayer, expectOutput: INDArray): DoubleLayer = {
          -(hyperparameters.log(prediction) * expectOutput).mean
        }

        val Array(batchSize, width, height, channels) = batch.pixels.shape()
        val coarseFeatures = CoarseFeatures(batch.pixels.reshape(batchSize, width * height * channels))
        val coarseProbabilities = CoarseProbabilityModel(coarseFeatures)

        crossEntropy(coarseProbabilities, batch.coarseClasses) + {
          if (excludeUnmatchedFineGrainedNetwork) {
            val fineScores = fineScoreModels(expectedCoarseLabel)(coarseFeatures)
            val expScores = exp(fineScores)
            val fineProbabilities = expScores / expScores.sum(1)
            crossEntropy(fineProbabilities, batch.localFineClasses)
          } else {
            val expScoresByCoarseLabel = for (coarseLabel <- 0 until Cifar100.NumberOfCoarseClasses) yield {
              val fineScores = fineScoreModels(expectedCoarseLabel)(coarseFeatures)
              exp(fineScores)
            }
            val expSum = expScoresByCoarseLabel.map(_.sum(1)).reduce(_ + _)
            val lossPerCoarseLabel = for ((expScores, coarseLabel) <- expScoresByCoarseLabel.zipWithIndex) yield {
              val fineProbabilities = expScores / expSum

              crossEntropy(
                fineProbabilities,
                if (coarseLabel == expScoresByCoarseLabel) {
                  batch.localFineClasses
                } else {
                  Nd4j.zeros(batchSize, Cifar100.NumberOfFineClassesPerCoarseClass)
                }
              )
            }
            lossPerCoarseLabel.reduce(_ + _)
          }
        }
      }

      def train(coarseLabel: Int, batch: Batch, excludeUnmatchedFineGrainedNetwork: Boolean) = {
        loss(coarseLabel, batch, excludeUnmatchedFineGrainedNetwork).train
      }

    }

    private var model: Model = null

    @Setup
    final def setup(): Unit = {
      computationGraph = new ComputationGraph(deeplearning4jConf)
      computationGraph.init()

      executionContext = ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(sizeOfThreadPool))
      model = new Model
    }

    @TearDown
    final def tearDown(): Unit = {
      model = null
      executionContext.shutdown()
      executionContext = null
      computationGraph = null
    }

    @Benchmark
    final def deeplearning4j(): Double = {
      val (coarseClass, batch) = batches.synchronized {
        batches.next()
      }

      val dataset = new MultiDataSet()

      val pixels2d = batch.pixels2d

      dataset.setFeatures(Array(pixels2d))

      val coarseLabels = Nd4j.zeros(1, Cifar100.NumberOfCoarseClasses)
      coarseLabels.put(0, coarseClass, 1.0)

      val broadcastCoarseLabels = coarseLabels.broadcast(pixels2d.rows(), Cifar100.NumberOfCoarseClasses)

      val fineLabels = Nd4j.concat(
        1,
        (for (i <- 0 until Cifar100.NumberOfCoarseClasses) yield {
          if (i == coarseClass) {
            batch.localFineClasses
          } else {
            Nd4j.zeros(pixels2d.rows(), Cifar100.NumberOfFineClassesPerCoarseClass)
          }
        }): _*
      )

      dataset.setLabels(Array(broadcastCoarseLabels, fineLabels))

      computationGraph.score(dataset, true)

    }

    @Benchmark
    final def deepLearningDotScalaExcludeUnmatchedFineGrainedNetwork(): Double = {
      val (coarseClass, batch) = batches.synchronized {
        batches.next()
      }
      model.train(coarseClass, batch, true).blockingAwait
    }
    @Benchmark
    final def deepLearningDotScala(): Double = {
      val (coarseClass, batch) = batches.synchronized {
        batches.next()
      }
      model.train(coarseClass, batch, false).blockingAwait
    }

  }

}
