package com.thoughtworks.deeplearning

import com.qifun.statelessFuture.Future

import language.existentials
import language.implicitConversions
import language.higherKinds
import scala.annotation.elidable

object Layer {

  object Tape {

    /** @template */
    type Aux[+Data0, -Delta0] = Tape {
      type Data <: Data0
      type Delta >: Delta0
    }

  }

  /**
    * Tape是对Data和Delta的封装，每个Tape都包含`backward()`,详细信息可参看[[Layer]]
    */
  trait Tape {
    type Data
    type Delta

    /**
      * Returns a new [[Tape]] that shares the same [[value]] and [[backward]] behavior with this [[Tape]].
      * @note The newly created [[Tape]] and this [[Tape]] must be [[close]]d independently.
      */
    def duplicate(): Tape.Aux[Data, Delta]

    def isTrainable: Boolean

    def close(): Future[Unit]

    def backward(delta: Delta): Future[Unit]

    def value: Data
  }

  /** @template */
  type Aux[-Input0, +Output0 <: Tape] =
    Layer {
      type Input >: Input0
      type Output <: Output0
    }

}

/**
  * Layer包括Input，Output和forward，Input和Output都是[[com.thoughtworks.deeplearning.Layer.Tape]],
  * 而Tape包含[[com.thoughtworks.deeplearning.Layer.Tape.backward()]],所以Layer所组成的网络会包含输入和输出，正向传播和反向传播。
  *
  * @example{{{
  *  val depthKernelKernel: Layer.Aux[Input, Tape.Aux[Int, Float]] =
  *    Times(
  *       Times(depth, Literal(kernel._1)),
  *       Literal(kernel._2)
  *     )
  *  val bSeq: Seq[Layer.Aux[Input, Tape.Aux[Int, Float]]] = Seq(kernelNumber, depthKernelKernel)
  *  val reshapeWeightTo: Layer.Aux[Input, Tape.Aux[Seq[Int], (Int, Float)]] = DifferentiableSeq.Layers.ToSeq(bSeq)
  *  val reshapedWeight = Reshape(weight, reshapeWeightTo)
  * }}}
  *
  * 以上代码等价于weight.reshape(kernelNumber,depth * KernelSize * KernelSize),
  * 在DeepLearning.scala中，`Reshape()`和`reshape()`其实是等价的(可以参考[[com.thoughtworks.deeplearning.DifferentiableINDArray#reshape]]的具体实现),
  * `reshape()`只是一个语法糖，其实最终还是调用`Reshape()`，调用`reshape()`会产生一个''case class''，而示例中的多个方法嵌套调用会生成类似这样的树：
  * Reshape(Weight([1,2,3]),ToSeq(Times(Times(kernel._1,kernel._2),depth)))，然后forward就从最里面的Times()开始，直到最外面的Reshape(),
  * 然后backward从Reshape()开始，直到最里面的Times()结束。
  */
trait Layer {

  import Layer._

  type Input

  type Output <: Tape

  def forward(input: Input): Future[Output]

}
