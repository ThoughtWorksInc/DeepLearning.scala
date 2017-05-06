package com.thoughtworks.deeplearning

import com.thoughtworks.raii.asynchronous.Do

import scalaz.{-\/, \/-}
import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Tape {

  type Data
  type Delta

  def data: Data
  def backward(outputDelta: Do[_ <: Delta]): Future[Unit]
}

object Tape {

  type Aux[+Data0, -Delta0] = Tape {
    type Data <: Data0
    type Delta >: Delta0
  }

  final case class Literal[Data0](data: Data0) extends Tape {
    override type Data = Data0
    override type Delta = Any

    override def backward(outputDelta: Do[_ <: Delta]): Future[Unit] = Future.now(())
  }

}
