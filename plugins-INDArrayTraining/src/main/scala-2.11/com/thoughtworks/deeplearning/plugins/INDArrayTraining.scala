package com.thoughtworks.deeplearning
package plugins

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import spire.algebra.MultiplicativeMonoid
import org.nd4s.Implicits._
private object INDArrayTraining {

  private val One: INDArray = {
    // Workaround for https://github.com/deeplearning4j/nd4j/issues/1869
    def workaround1869(underlying: INDArray): INDArray = {
      import java.lang.reflect.{Proxy, InvocationHandler, Method}
      val invocationHandler = new InvocationHandler {
        override def invoke(proxy: scala.Any, method: Method, args: Array[AnyRef]): AnyRef = {
          method.getName match {
            case "reshape" =>
              workaround1869(method.invoke(underlying, args: _*).asInstanceOf[INDArray])
            case "broadcast" =>
              val Array(newShape: Array[Int]) = args
              workaround1869(Nd4j.ones(newShape: _*))
            case _ =>
              method.invoke(underlying, args: _*)
          }
        }
      }
      Proxy
        .newProxyInstance(getClass.getClassLoader, Array(classOf[INDArray]), invocationHandler)
        .asInstanceOf[INDArray]
    }
    workaround1869(Nd4j.ones(1, 1))
  }
}

/** A DeepLearning.scala plugin that enable [[DeepLearning.Ops.train train]] method for neural networks whose loss is a [[org.nd4j.linalg.api.ndarray.INDArray]].
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait INDArrayTraining extends Training {

  trait ImplicitsApi extends super.ImplicitsApi {
    implicit final def indArrayMultiplicativeMonoid: MultiplicativeMonoid[INDArray] =
      new MultiplicativeMonoid[INDArray] {
        override def one: INDArray = INDArrayTraining.One

        override def times(x: INDArray, y: INDArray): INDArray = x * y
      }
  }
  type Implicits <: ImplicitsApi
}
