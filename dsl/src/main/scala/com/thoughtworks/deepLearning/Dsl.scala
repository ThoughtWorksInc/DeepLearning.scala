package com.thoughtworks.deepLearning


import scala.language.existentials
import scala.language.implicitConversions
import scala.language.higherKinds

object Dsl {


  object Lifter {
    type Aux[LiftFrom0, LiftTo0] = (LiftFrom0 => LiftTo0) with Lifter {
      type LiftFrom = LiftFrom0
      type LiftTo = LiftTo0
    }
  }

  trait Lifter {
    _: (_ => _) =>
    type LiftFrom
    type LiftTo

    def weight(initialValue: LiftFrom): LiftTo

    def apply(value: LiftFrom): LiftTo
  }

  object DslFunction {
    type Aux[In0, Out0] = DslFunction {
      type In = In0
      type Out = Out0
    }
  }

  trait DslFunction {
    type In
    type Out

    def apply(in: In): Out
  }

}

trait Dsl {

  import Dsl._

  trait HListApi {
    _: HList =>
    def ::[Head <: Any : Companion, Tail >: this.type <: HList : HListCompanion](head: Head): Head :: Tail
  }

  trait HConsApi[+Head <: Any, +Tail <: HList] extends HListApi {
    _: Head :: Tail =>

    def head: Head

    def tail: Tail

  }

  trait Array2DApi {
    _: Array2D =>

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

  trait BooleanApi {
    _: Boolean =>
    def unary_! : Boolean

    def `if`[A <: Any : Companion](`then`: A)(`else`: A): A
  }

  trait DoubleApi {
    _: Double =>

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

  type Companion[_ <: Any]
  type HListCompanion[Ast <: HList] <: Companion[Ast]

  type Any

  type Boolean <: BooleanApi with Any
  implicit val Boolean: Companion[Boolean]

  type Double <: DoubleApi with Any
  implicit val Double: Companion[Double] with Lifter.Aux[scala.Double, Double]


  trait Array2DCompanionApi extends Lifter with (scala.Array[scala.Array[scala.Double]] => Array2D) {
    override type LiftTo = Array2D
    override type LiftFrom = scala.Array[scala.Array[scala.Double]]

    def randn(numberOfRows: Int, numberOfColumns: Int): Array2D

    def randn(numberOfColumns: Int): Array2D = randn(1, numberOfColumns)

    def zeros(numberOfRows: Int, numberOfColumns: Int): Array2D

    def zeros(numberOfColumns: Int): Array2D = zeros(1, numberOfColumns)
  }

  type Array2D <: Array2DApi with Any
  implicit val Array2D: Companion[Array2D] with Array2DCompanionApi

  type ::[+Head <: Any, +Tail <: HList] <: HConsApi[Head, Tail] with HList

  implicit def ::[Head <: Any : Companion, Tail <: HList : HListCompanion]: HListCompanion[Head :: Tail]

  type HList <: HListApi with Any

  type HNil <: HList
  val HNil: HNil with HListCompanion[HNil]

  def max(leftHandSide: Double, rightHandSide: Double): Double = {
    (leftHandSide < rightHandSide).`if`(rightHandSide)(leftHandSide)
  }

  def max(leftHandSide: Array2D, rightHandSide: Double): Array2D

  def max(leftHandSide: Double, rightHandSide: Array2D): Array2D = {
    max(rightHandSide, leftHandSide)
  }

  def min(leftHandSide: Double, rightHandSide: Double): Double = {
    (leftHandSide < rightHandSide).`if`(leftHandSide)(rightHandSide)
  }

  def exp(array: Array2D): Array2D

  def log(array: Array2D): Array2D
}
