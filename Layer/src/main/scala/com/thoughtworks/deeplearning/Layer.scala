package com.thoughtworks.deeplearning

import com.thoughtworks.future.Continuation.Task

import language.existentials
import language.implicitConversions
import language.higherKinds
import scala.annotation.elidable
import scala.util.control.TailCalls.TailRec

object Layer {

  object Tape {

    /** @template */
    type Aux[+Data0, -Delta0] = Tape {
      type Data <: Data0
      type Delta >: Delta0
    }

    trait Untrainable extends Tape

    trait Trainable extends Tape {
      def retain(): TailRec[Unit]
      def release(): Task[Unit]
      def backward(delta: Delta): TailRec[Unit]
    }

  }

  sealed trait Tape {
    type Data
    type Delta

    def value: Data

  }

  /** @template */
  type Aux[-Input0, +Output0] =
    Layer {
      type Input >: Input0
      type Output <: Output0
    }

}

trait Layer {

  import Layer._

  type Input

  type Output

  def forward(input: Input): Output

}
