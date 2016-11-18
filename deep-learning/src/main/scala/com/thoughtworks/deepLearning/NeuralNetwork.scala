package com.thoughtworks.deepLearning

import com.thoughtworks.deepLearning.Batch._
import cats._

import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds
import cats.implicits._

import scala.annotation.elidable

object NeuralNetwork /*extends LowPriortyDifferentiableFunction*/ {

  /** @template */
  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    NeuralNetwork {
      type Input >: Input0
      type Output <: Output0
    }

}

trait NeuralNetwork {

  import NeuralNetwork._

  type Input <: Batch

  type Output <: Batch

  def forward(input: BatchId.Aux[Input]): BatchId.Aux[Output]

}
