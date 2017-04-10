package com.thoughtworks.deeplearning

import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Tape {

  type Data
  type Delta

  def data: Data
  def backward(delta: Future[Delta]): Future[Unit]

  /** @see https://github.com/scala/bug/issues/10251 */
  @inline private[deeplearning] final def workaround10251: {
    type Data = Tape.this.Data
    type Delta = Tape.this.Delta
  } = this
}

object Tape {

  type Aux[+Data0, -Delta0] = Tape {
    type Data <: Data0
    type Delta >: Delta0
  }

  final case class Literal[Data0](data: Data0) extends Tape {
    override type Data = Data0
    override type Delta = Any

    override def backward(delta: Future[Delta]): Future[Unit] = Future.now(())
  }

}
