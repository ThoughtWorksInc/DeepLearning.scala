package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Batch._
import cats._

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._

import scala.annotation.elidable

//import scala.annotation.elidable
//sealed trait LowLowLowPriortyDifferentiableFunction {
//
//  implicit def subtypingToNeuralNetwork[Input <: Batch, OutputData, OutputDelta, From](
//      implicit view: Lazy[From <:< NeuralNetwork.Aux[Input, ConcreteBatch[OutputData, OutputDelta]]])
//    : ToNeuralNetwork[From, Input, OutputData, OutputDelta] = {
//    new ToNeuralNetwork[From, Input, OutputData, OutputDelta] {
//      override def apply(value: From) = view.value(value)
//    }
//
//  }
//}
//sealed trait LowLowPriortyDifferentiableFunction extends LowLowLowPriortyDifferentiableFunction {
//  this: LowPriortyDifferentiableFunction =>
//
//  import NeuralNetwork._
//
//  implicit def toNeuralNetworkNN[Input <: Batch : Identity, NN[_ <: Batch], OutputPair <: Batch](
//      implicit nn: Lazy[NN[OutputPair] <:< NeuralNetwork.Aux[Input, OutputPair#ConcreteBatch]])
//    : ToNeuralNetwork[NN[OutputPair], Input, OutputPair#Data, OutputPair#Delta] = {
//    new ToNeuralNetwork[NN[OutputPair], Input, OutputPair#Data, OutputPair#Delta] {
//      override def apply(value: NN[OutputPair]): NeuralNetwork.Aux[Input, ConcreteBatch[OutputPair#Data, OutputPair#Delta]] = {
//        toNeuralNetworkPair.apply(nn.value(value))
//      }
//    }
//  }
//  //
//  // implicit def toNeuralNetworkNN2[Input <: Batch, NN[_ <: Batch], OutputPair <: Batch](
//  //     implicit nn: Lazy[NN[OutputPair] <:< NeuralNetwork.Aux[Input, OutputPair#ConcreteBatch]])
//  //   : ToNeuralNetwork[NN[OutputPair], Input, OutputPair#Data, OutputPair#Delta] = {
//  //   new ToNeuralNetwork[NN[OutputPair], Input, OutputPair#Data, OutputPair#Delta] {
//  //     override def apply(value: NN[OutputPair]): NeuralNetwork.Aux[Input, ConcreteBatch[OutputPair#Data, OutputPair#Delta]] = {
//  //       toNeuralNetworkPair.apply(nn.value(value))
//  //     }
//  //   }
//  // }
//
//
//}
//
//sealed trait LowPriortyDifferentiableFunction extends LowLowPriortyDifferentiableFunction {
//
//  import NeuralNetwork._
//
//  implicit def toNeuralNetworkPair[Input <: Batch, OutputPair <: Batch]
//    : ToNeuralNetwork[NeuralNetwork.Aux[Input, OutputPair#ConcreteBatch], Input, OutputPair#Data, OutputPair#Delta] = {
//    // I can't prove this because the lack of for-all type in Scala language. Force it as a workaround.
//    new ToNeuralNetwork[NeuralNetwork.Aux[Input, OutputPair#ConcreteBatch], Input, OutputPair#Data, OutputPair#Delta] {
//      override def apply(value: NeuralNetwork.Aux[Input, OutputPair#ConcreteBatch]): NeuralNetwork.Aux[Input, ConcreteBatch[OutputPair#Data, OutputPair#Delta]] =
//        value.asInstanceOf[NeuralNetwork.Aux[Input, ConcreteBatch[OutputPair#Data, OutputPair#Delta]]]
//    }
//  }
//
//}

object NeuralNetwork /*extends LowPriortyDifferentiableFunction*/ {

  /** @template */
  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    NeuralNetwork {
      type Input >: Input0
      type Output <: Output0
    }

//  trait ToNeuralNetwork[T, Input <: Batch, OutputData, OutputDelta] {
//    def apply(value: T): NeuralNetwork.Aux[Input, Batch.ConcreteBatch[OutputData, OutputDelta]]
//  }
//
//  implicit def toNeuralNetwork[Input <: Batch, OutputData, OutputDelta]
//    : ToNeuralNetwork[NeuralNetwork.Aux[Input, Batch.ConcreteBatch[OutputData, OutputDelta]],
//            Input,
//            OutputData,
//            OutputDelta] = {
//    new ToNeuralNetwork[NeuralNetwork.Aux[Input, Batch.ConcreteBatch[OutputData, OutputDelta]],
//              Input,
//              OutputData,
//              OutputDelta] {
//      override def apply(
//          value: NeuralNetwork.Aux[Input, ConcreteBatch[OutputData, OutputDelta]]): NeuralNetwork.Aux[Input, ConcreteBatch[OutputData, OutputDelta]] = value
//    }
//
//  }

  trait Cached extends NeuralNetwork {

    private val cache =
      java.util.Collections.synchronizedMap(new java.util.IdentityHashMap[Input, SharedBatch](1))

    protected trait ReferenceCount extends Batch { this: SharedBatch =>

      /**
        * Returns a wrapped [[Batch]] able to detect error of closing more than once if ASSERTION is enabled,
        * or returns this [[ReferenceCount]] itself when ASSERTION is disabled hence no check.
        */
      private[Cached] def maybeCheckIfCloseOnlyOnce: Self = {

        // Returns a [[Batch]] able to detect error of closing more than once.
        @elidable(elidable.ASSERTION)
        def checkIfCloseOnlyOnce = new Batch {
          type Delta = ReferenceCount.this.Delta
          type Data = ReferenceCount.this.Data

          override final def backward(delta: Delta) = ReferenceCount.this.backward(delta)

          def value = ReferenceCount.this.value

          private var closed = false

          override final def close(): Unit = {
            ReferenceCount.this.synchronized {
              if (closed) {
                throw new IllegalStateException("close() method must be called once and only once.")
              } else {
                closed = true
              }
            }
            ReferenceCount.this.close()
          }
        }

        Option(checkIfCloseOnlyOnce).getOrElse(ReferenceCount.this.self)
      }

      type Self >: Batch.Aux[Data, Delta] <: Batch.Aux[Data, Delta]

      private final def self: Self = this: Batch.Aux[Data, Delta]

      private[Cached] var count: Int = 1

      protected def flush(): Unit

      /**
        * Closes upstream batch of this [[SharedBatch]]
        */
      protected def closeUpstreams(): Unit

      def input: Input

      override final def close(): Unit = {
        val newCount = synchronized {
          count -= 1
          count
        }
        assert(newCount >= 0)
        if (newCount == 0) {
          val batch = cache.remove(input)
          assert(batch == this)
          flush()
          closeUpstreams()
        }
      }

    }

    protected trait MonoidBatch extends ReferenceCount { this: SharedBatch =>

      private var currentDelta: Delta = monoid.empty

      /**
        * Performs the underlying backward pass with all `upstreamDelta`s that previously received from [[#backward]].
        */
      protected def rawBackward(delta: Delta): Unit

      implicit protected def monoid: Monoid[Delta]

      override protected final def flush(): Unit = {
        rawBackward(synchronized {
          val delta = currentDelta
          currentDelta = monoid.empty
          delta
        })
      }

      override final def backward(delta: Delta): Unit = {
        synchronized {
          currentDelta = currentDelta |+| delta
        }
      }
    }

    protected trait SemigroupBatch extends ReferenceCount { this: SharedBatch =>

      private var currentDelta: Option[Delta] = None

      protected def rawBackward(delta: Delta): Unit

      implicit protected def semigroup: Semigroup[Delta]

      override protected final def flush(): Unit = {
        synchronized {
          val delta = currentDelta
          currentDelta = None
          delta
        }.foreach(rawBackward)
      }

      override final def backward(delta: Delta): Unit = {
        synchronized {
          currentDelta = currentDelta |+| Some(delta)
        }
      }
    }

    type Output = SharedBatch#Self

    protected type SharedBatch <: ReferenceCount

    /**
      * Performs the underlying forward pass.
      *
      * @return a [[Batch]] that will be cached for subsequent [[#forward]]
      */
    protected def rawForward(input: Input): SharedBatch

    override final def forward(input: Input): Output = {
      val sharedBatch: SharedBatch = cache.get(input) match {
        case null =>
          val sharedBatch = rawForward(input)
          val oldBatch = cache.put(input, sharedBatch)
          assert(oldBatch == null)
          sharedBatch
        case sharedBatch =>
          sharedBatch.synchronized {
            val count = sharedBatch.count
            assert(count >= 0)
            sharedBatch.count += count + 1
          }
          sharedBatch
      }
      sharedBatch.maybeCheckIfCloseOnlyOnce
    }
  }

}

trait NeuralNetwork {

  import NeuralNetwork._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Output

}
