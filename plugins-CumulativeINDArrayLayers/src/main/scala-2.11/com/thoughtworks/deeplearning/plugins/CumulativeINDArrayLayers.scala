package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.ImplicitApply
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.covariant.Releasable
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4s.IndexRange

import scalaz.concurrent.Future
import scala.util.{Success, Try}
import scalaz.{-\/, \/-}
import scalaz.syntax.all._

private object CumulativeINDArrayLayers {
  private val Zero = Nd4j.zeros(1, 1)
}

/** A plugin that provides differentiable operators
  * on neural networks whose [[DeepLearning.Data Data]] and [[DeepLearning.Delta Delta]] is [[org.nd4j.linalg.api.ndarray.INDArray]].
  *
  * @note Unlike [[INDArrayLayers]], [[INDArrayLayer]] in this `CumulativeINDArrayLayers` will share [[DeepLearning.Tape Tape]]s
  *       created in [[INDArrayLayer.forward forward]] pass for all dependencies, avoiding re-evaluation
  *       in the case of diamond dependencies in a neural network.
  *
  * @author 杨博 (Yang Bo)
  */
trait CumulativeINDArrayLayers extends INDArrayLayers {
  import CumulativeINDArrayLayers._
  import INDArrayLayers._

  trait INDArrayLayerApi extends super[INDArrayLayers].INDArrayLayerApi {

    private final class Accumulator(val data: INDArray, flushBackward: Do[INDArray] => Future[Unit])
        extends Releasable[Future, Try[Accumulator]] {
      @volatile
      var currentDelta: INDArray = CumulativeINDArrayLayers.Zero

      override def value: Try[Accumulator] = Success(this)

      override def release(): Future[Unit] = {
        synchronized {
          val deltaOption = currentDelta
          currentDelta = null
          deltaOption
        } match {
          case Zero =>
            Future.now(())
          case nonZeroDelta =>
            flushBackward(Do.now(nonZeroDelta))
        }
      }
    }

    private lazy val sharedAccumulator = {
      super.forward.flatMap {
        case Tape(data, flushBackward) =>
          Do(Future.delay(new Accumulator(data, flushBackward)))
      }
    }

    /** @usecase def apply(indices: Int*): DoubleLayer = ???
      *
      * @example Given an [[INDArrayLayer]],
      *
      *          {{{
      *          import org.nd4j.linalg.factory.Nd4j
      *          import com.thoughtworks.feature.Factory
      *          import com.thoughtworks.deeplearning.plugins._
      *          val hyperparameters = Factory[DoubleTraining with CumulativeINDArrayLayers with INDArrayWeights with ImplicitsSingleton with Operators].newInstance()
      *          import hyperparameters.implicits._
      *          val weight = hyperparameters.INDArrayWeight(Nd4j.ones(2, 3))
      *          val indArrayLayer = hyperparameters.INDArrayLayer(weight.forward)
      *          }}}
      *
      *          and select one element in the [[INDArrayLayer]],
      *
      *          {{{
      *          val doubleLayer: hyperparameters.DoubleLayer = indArrayLayer(0, 2)
      *          }}}
      *
      *          when training the selected element,
      *
      *          then the data of the element should be 1.0,
      *          in the original weight,
      *          only the element corresponding to the index get trained.
      *
      *          {{{
      *          doubleLayer.train.map { output =>
      *            output should be(1.0)
      *
      *            import org.nd4s.Implicits._
      *            weight.data(0, 0) should be(1.0)
      *            weight.data(0, 1) should be(1.0)
      *            weight.data(0, 2) should be < 1.0
      *            weight.data(1, 0) should be(1.0)
      *            weight.data(1, 1) should be(1.0)
      *            weight.data(1, 2) should be(1.0)
      *          }
      *          }}}
      *
      */
    def apply[Out <: DoubleLayer](indices: Int*)(
        implicit implicitApply: ImplicitApply.Aux[doublePartialApplyRawForward.Rest, Out]): Out = {
      val doDoubleTape = sharedAccumulator.map { accumulator =>
        def cumulativeBackward(doDelta: Do[Double]): Future[Unit] = {
          Do.run(doDelta)
            .map { delta: Double =>
              accumulator.synchronized {
                accumulator.currentDelta = accumulator.currentDelta match {
                  case null =>
                    throw new IllegalStateException("Cannot perform Tape.backward after the Tape is released")
                  case Zero =>
                    val zeros = Nd4j.zeros(accumulator.data.shape(): _*)
                    val indexRanges = indices.map[IndexRange, Array[IndexRange]](i => i)(collection.breakOut)
                    zeros(indexRanges) = delta
                  case nonZeroDelta =>
                    val broadcasted = nonZeroDelta.broadcastFix(accumulator.data.shape(): _*)
                    val oldDelta = broadcasted(indices: _*)
                    val indexRanges = indices.map[IndexRange, Array[IndexRange]](i => i)(collection.breakOut)
                    broadcasted(indexRanges) = oldDelta + delta
                    broadcasted
                }
              }
            }
            .get
            .map {
              case \/-(()) => // Success. Do nothing
              case -\/(e)  => handleException(e)
            }
        }
        Tape(accumulator.data(indices: _*), cumulativeBackward)
      }
      DoubleLayer(doDoubleTape)
    }

    abstract override def forward: Do[Tape[INDArray, INDArray]] = {
      sharedAccumulator.map { accumulator =>
        def cumulativeBackward(doDelta: Do[INDArray]): Future[Unit] = {
          Do.run(doDelta)
            .map { delta =>
              accumulator.synchronized {
                accumulator.currentDelta = accumulator.currentDelta match {
                  case null =>
                    throw new IllegalStateException("Cannot perform Tape.backward after the Tape is released")
                  case Zero => delta
                  case nonZeroDelta =>
                    def autoBroadcastShape(shape1: Array[Int], shape2: Array[Int]): Array[Int] = {
                      require(shape1.length == shape2.length)
                      shape1.zip(shape2).map {
                        case (1, bSize)                       => bSize
                        case (aSize, 1)                       => aSize
                        case (aSize, bSize) if aSize == bSize => aSize
                      }
                    }

                    val shape = autoBroadcastShape(nonZeroDelta.shape(), delta.shape())
                    nonZeroDelta.broadcastFix(shape: _*) + delta.broadcastFix(shape: _*)
                }
              }
            }
            .get
            .map {
              case \/-(()) => // Success. Do nothing
              case -\/(e)  => handleException(e)
            }
        }
        Tape(accumulator.data, cumulativeBackward)
      }

    }

  }
  override type INDArrayLayer <: INDArrayLayerApi with Layer
}
