package com.thoughtworks.deeplearning

import com.dongxiguo.fastring.Fastring

import language.existentials
import language.implicitConversions
import language.higherKinds
import scala.annotation.elidable

object Layer {

  private[deeplearning] trait CloseableOnce extends AutoCloseable {

    private[CloseableOnce] final class ClosingFlag {
      var closed = false
      @elidable(elidable.ASSERTION)
      def close() = {
        assert(!closed)
        closed = true
      }

      @elidable(elidable.ASSERTION)
      def assertClosed() = {
        assert(closed)
      }
    }

    // FIXME: @elidable should only be used for def
    @elidable(elidable.ASSERTION)
    private val closingFlag = new ClosingFlag

    override def close() = {
      closingFlag.close()
    }

    override protected def finalize(): Unit = {
      closingFlag.assertClosed()
    }
  }

  object Batch {

    /** @template */
    type Aux[+Data0, -Delta0] = Batch {
      type Data <: Data0
      type Delta >: Delta0
    }

  }

  trait Batch extends AutoCloseable {
    type Data
    type Delta

    def addReference(): Batch.Aux[Data, Delta]

    protected def forceBackward(delta: Delta): Unit

    def isTrainable: Boolean

    @inline
    final def backward(delta: => Delta): Unit = {
      if (isTrainable) {
        forceBackward(delta)
      }
    }

    def value: Data
  }

  /** @template */
  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    Layer {
      type Input >: Input0
      type Output <: Output0
    }

  final case class OpenClBlock(code: Fastring, definitions: String, valueName: String)

  final case class ClBatch(
                            forwardCode: Fastring,
                            outputDataName: String,
                              backwardCode: Fastring,
                            inputDeltaName:String
                          )

}

trait Layer {

  import Layer._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Output

  def symbolicForward(inputName:String): (Fastring, Fastring) => Fastring

}

