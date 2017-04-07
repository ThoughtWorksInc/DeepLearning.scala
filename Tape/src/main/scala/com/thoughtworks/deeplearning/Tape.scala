package com.thoughtworks.deeplearning

import scalaz.concurrent.Future

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait Tape {

  type Data
  type Delta

  def data: Data

}

object Tape {

  trait Trainable extends Tape {
    def backward(delta: Delta): Future[Unit]
  }

}
