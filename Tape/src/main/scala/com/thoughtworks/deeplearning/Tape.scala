package com.thoughtworks.deeplearning

import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
sealed trait Tape {

  type Data
  type Delta

  def data: Data

}

object Tape {

  type Aux[+Data0, -Delta0] = Tape {
    type Data <: Data0
    type Delta >: Delta0
  }

  trait Untrainable extends Tape

  object Untrainable {
    type Aux[+Data0, -Delta0] = Untrainable {
      type Data <: Data0
      type Delta >: Delta0
    }
  }

  final case class Literal[Data0](data: Data0) extends Untrainable {
    override type Data = Data0
    override type Delta = Any
  }

  trait Trainable extends Tape {
    def backward(delta: Delta): Future[Unit]
  }

  object Trainable {
    type Aux[+Data0, -Delta0] = Trainable {
      type Data <: Data0
      type Delta >: Delta0
    }
  }

}
