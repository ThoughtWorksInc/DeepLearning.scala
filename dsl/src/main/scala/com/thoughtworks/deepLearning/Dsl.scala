package com.thoughtworks.deepLearning


import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds

object Dsl {

  trait DoubleApi {
    type Companion[_]
    type Array2D <: Array2DApi.Aux[Companion, Array2D, Double, Boolean]
    type Double >: this.type <: DoubleApi.Aux[Companion, Array2D, Double, Boolean]
    type Boolean <: BooleanApi.Aux[Companion, Boolean]

    def unary_- : Double

    def -(rightHandSide: Double): Double = {
      this + -rightHandSide
    }

    def -(rightHandSide: Array2D): Array2D = {
      this + -rightHandSide
    }

    def +(rightHandSide: Double): Double

    def +(rightHandSide: Array2D): Array2D = {
      rightHandSide + (this: Double)
    }

    def /(rightHandSide: Double): Double

    def /(rightHandSide: Array2D): Array2D

    def *(rightHandSide: Double): Double

    def *(rightHandSide: Array2D): Array2D = {
      rightHandSide * (this: Double)
    }

    def <(rightHandSide: Double): Boolean

    def >=(rightHandSide: Double): Boolean = {
      !(rightHandSide < (this: Double))
    }

  }

  object DoubleApi {
    type Aux[Companion0[_], Array2D0, Double0, Boolean0] = DoubleApi {
      type Companion[A] = Companion0[A]
      type Array2D = Array2D0
      type Double = Double0
      type Boolean = Boolean0
    }
  }

  trait BooleanApi {
    type Companion[_]
    type Boolean >: this.type <: BooleanApi.Aux[Companion, Boolean]


    def unary_! : Boolean

    def `if`[A: Companion](`then`: A)(`else`: A): A
  }

  object BooleanApi {
    type Aux[Companion0[_], Boolean0] = BooleanApi {
      type Companion[A] = Companion0[A]
      type Boolean = Boolean0
    }
  }

  trait Array2DApi {
    type Double <: DoubleApi.Aux[Companion, Array2D, Double, Boolean]
    type Array2D <: Array2DApi.Aux[Companion, Array2D, Double, Boolean]
    type Boolean <: BooleanApi
    type Companion[_]

    def dot(rightHandSide: Array2D): Array2D

    def +(rightHandSide: Array2D): Array2D

    def +(rightHandSide: Double): Array2D

    def /(rightHandSide: Array2D): Array2D

    def /(rightHandSide: Double): Array2D

    def *(rightHandSide: Array2D): Array2D

    def *(rightHandSide: Double): Array2D

    def -(rightHandSide: Array2D): Array2D = {
      this + -rightHandSide
    }

    def -(rightHandSide: Double): Array2D = {
      this + -rightHandSide
    }

    def unary_- : Array2D

    def reduceSum: Double

    def sum(dimensions: Int*): Array2D

  }

  object Array2DApi {
    type Aux[Companion0[_], Array2D0, Double0, Boolean0] = Array2DApi {
      type Companion[A] = Companion0[A]
      type Array2D = Array2D0
      type Double = Double0
      type Boolean = Boolean0
    }
  }

  object Lifter {
    type Aux[LiftFrom0, LiftTo0] = Lifter {
      type LiftFrom = LiftFrom0
      type LiftTo = LiftTo0
    }
  }

  trait Lifter {
    type LiftFrom
    type LiftTo

    def weight(initialValue: LiftFrom): LiftTo

    def apply(value: LiftFrom): LiftTo
  }

}

trait Dsl {

  import Dsl._

  type Companion[_]

  type Any
  implicit val Any: Companion[Any]

  type Boolean <: Any with BooleanApi.Aux[Companion, Boolean]
  implicit val Boolean: Companion[Boolean]

  type Double <: Any with DoubleApi.Aux[Companion, Array2D, Double, Boolean]
  implicit val Double: Companion[Double] with Lifter.Aux[scala.Double, Double]

  type Array2D <: Any with Array2DApi.Aux[Companion, Array2D, Double, Boolean]
  implicit val Array2D: Companion[Array2D] with Lifter.Aux[Array[Array[scala.Double]], Array2D]

  def max(leftHandSide: Double, rightHandSide: Double): Double = {
    (leftHandSide < rightHandSide).`if`(rightHandSide)(leftHandSide)
  }

  def min(leftHandSide: Double, rightHandSide: Double): Double = {
    (leftHandSide < rightHandSide).`if`(leftHandSide)(rightHandSide)
  }

  def exp(array: Array2D): Array2D

  def log(array: Array2D): Array2D
}
