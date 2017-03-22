package com.thoughtworks.deeplearning

import com.qifun.statelessFuture.Future
import com.thoughtworks.deeplearning.Closeables.AssertionAutoCloseable
import com.thoughtworks.deeplearning.Layer.Tape

import scala.annotation.elidable

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object CheckedTape {

  /**
    * Returns a [[Tape]] able to detect error of closing more than once.
    *
    * @note The error detection behavior can be disabled by setting scalac flag `-Xelide-below=MAXIMUM`,
    *       or `-Xelide-below=&lt;a number greater than elidable.ASSERTION&gt;`
    */
  def assertionModeOnly[Data0, Delta0](underlying: Tape.Aux[Data0, Delta0]): Tape.Aux[Data0, Delta0] = {
    @inline
    @elidable(elidable.ASSERTION)
    def elidableCheckedTape(underlying: Tape.Aux[Data0, Delta0]) = {
      CheckedTape(underlying)
    }
    val checkedTape = elidableCheckedTape(underlying)
    if (checkedTape == null) {
      underlying
    } else {
      checkedTape
    }
  }

}

/**
  * A [[Tape]] able to detect error of closing more than once.
  */
final case class CheckedTape[Data0, Delta0](underlying: Tape.Aux[Data0, Delta0])
    extends Tape
    with AssertionAutoCloseable {
  override type Data = Data0
  override type Delta = Delta0

  override def duplicate(): Tape.Aux[Data, Delta] = CheckedTape(underlying.duplicate())
  override def forceBackward(delta: Delta): Future[Unit] = underlying.forceBackward(delta)
  override def isTrainable: Boolean = underlying.isTrainable
  override def value: Data = underlying.value

  override protected def forceClose(): Unit = underlying.close()
}
