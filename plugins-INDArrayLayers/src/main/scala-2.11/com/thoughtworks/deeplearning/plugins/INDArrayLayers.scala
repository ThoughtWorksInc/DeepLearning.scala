package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._
import com.thoughtworks.raii.shared._
import com.thoughtworks.raii.covariant.{Releasable, ResourceT}
import com.thoughtworks.tryt.covariant.TryT
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scalaz.concurrent.Future
import scala.util.{Success, Try}
import scalaz.{-\/, \/-}
import scalaz.syntax.all._

private object INDArrayLayers {
  private val Zero = Nd4j.zeros(1, 1)
}

/**
  * @author 杨博 (Yang Bo)
  */
trait INDArrayLayers extends RawINDArrayLayers {
  import INDArrayLayers._

  trait INDArrayLayerApi extends super[RawINDArrayLayers].INDArrayLayerApi {

    private lazy val forward0: Do[Tape[INDArray, INDArray]] = {
      val Do(future) = super.forward.flatMap {
        case Tape(data, flushBackward) =>
          Do(Future.delay(new Releasable[Future, Try[Tape[INDArray, INDArray]]] {

            @volatile
            private var currentDelta: INDArray = INDArrayLayers.Zero

            override def value: Try[Tape[INDArray, INDArray]] = {
              def cumulativeBackward(doDelta: Do[INDArray]) = {
                Do.run(doDelta)
                  .map {
                    delta =>
                      synchronized {
                        currentDelta = currentDelta match {
                          case null =>
                            throw new IllegalStateException("Cannot perform Tape.backward after the Tape is released")
                          case Zero => delta
                          case nonZeroDelta =>
                            def autoBroadcastShape(shape1: Array[Int], shape2: Array[Int]): Array[Int] = {
                              require(shape1.length == shape2.length)
                              shape1.zip(shape2).map {
                                case (1, bSize) => bSize
                                case (aSize, 1) => aSize
                                case (aSize, bSize) if aSize == bSize => aSize
                              }
                            }
                            val shape = autoBroadcastShape(nonZeroDelta.shape(), delta.shape())
                            nonZeroDelta.broadcast(shape: _*) + delta.broadcast(shape: _*)
                        }
                      }
                  }
                  .get
                  .map {
                    case \/-(()) => // Success. Do nothing
                    case -\/(e) => handleException(e)
                  }
              }
              Success(Tape(data, cumulativeBackward))
            }

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

          }))
      }
      val ResourceT(sharedFuture) = ResourceT(future).shared
      Do(sharedFuture)
    }
    abstract override def forward: Do[DeepLearning.Tape[INDArray, INDArray]] = forward0

  }
  override type INDArrayLayer <: INDArrayLayerApi with Layer
}
