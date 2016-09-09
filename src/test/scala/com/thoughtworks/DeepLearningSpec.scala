package com.thoughtworks

import org.scalatest._
import cats.implicits._
import com.thoughtworks.DeepLearning.DifferentiableFunction.Id
import shapeless.DepFn1

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class DeepLearningSpec extends FreeSpec with Matchers with Inside {

  "-" in {
    //    import DeepLearning._

    //    val input = Input[scala.Double, scala.Double]
    //    val output = input - Double(3.0)
    //

    implicit def learningRate = new DeepLearning.LearningRate {
      def apply() = 0.0003
    }

    def f(dsl: Dsl)(input: dsl.Double): dsl.Double = {
      import dsl._
      -input - Double(3.0)
    }

    val id = new Id[scala.Double, scala.Double]
    val dsl = new DeepLearning[id.Input]
    val f1 = f(dsl)(id)
    val nn = f1.compose(f1)


    //    def xxx[Input0 >: DeepLearning.Cache.Aux[Input0, scala.Double, scala.Double]] = {
    //
    //      val dsl = new DeepLearning[Input0]
    //      val f1: Differentiable.Aux[Input0, Cache.Aux[Input0, scala.Double, scala.Double]] = f(dsl)(new Id[scala.Double, scala.Double])
    //
    //      f1.compose(f1)
    //    }
    //
    //    val f1 = xxx[T forSome {type T <: DeepLearning.Cache.Aux[T, scala.Double, scala.Double]}]
    //
    //    type Input = T forSome {type T <: DeepLearning.Cache.Aux[T, scala.Double, scala.Double]}
    //    val dsl = new DeepLearning[Input]
    //    val f1 = f(dsl)((new Id[Input]).self)

//        DeepLearning.Differentiable.compose(f1,f1)
    //    f1.compose(f1)

    //    implicitly[Cache.Aux[Cache, scala.Double, scala.Double] <:< Cache.Aux[Cache.Aux[Cache, scala.Double, scala.Double], scala.Double, scala.Double]]


    //    DeepLearning.Differentiable (new DepFn1 {
    //
    //    })
    //
    //    DeepLearning.Differentiable { dsl: DeepLearning[DeepLearning.DoubleCache.Aux[DeepLearning.DoubleCache]] =>
    //      import dsl._
    //      input: Double =>
    //        dsl.doubleOps(input) - Double(3.0)
    //    }
    //    val output = Differentiable { input =>
    //      input - Double(3.0)
    //    }


    //    network.forwardApply(Double(2.9))

  }

}
