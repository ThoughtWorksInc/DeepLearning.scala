package com.thoughtworks.deepLearning

import cats._

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._

import scala.annotation.elidable

// TODO: Split this file into multiple modules
object Differentiable {

  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    Differentiable {
      type Input >: Input0
      type Output <: Output0
    }
}

trait Differentiable {

  import Differentiable._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Output

}
