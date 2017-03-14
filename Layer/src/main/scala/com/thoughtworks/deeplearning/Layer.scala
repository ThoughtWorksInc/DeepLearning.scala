package com.thoughtworks.deeplearning

import language.existentials
import language.implicitConversions
import language.higherKinds
import scala.annotation.elidable

object Layer {

  private[deeplearning] trait CloseableOnce extends AutoCloseable {

    private[CloseableOnce] final class ClosingFlag {
      var closed = false
      @elidable(elidable.ASSERTION)
      def close() = {
        assert(!closed)
        closed = true
      }

      @elidable(elidable.ASSERTION)
      def assertClosed() = {
        assert(closed)
      }
    }

    // FIXME: @elidable should only be used for def
    @elidable(elidable.ASSERTION)
    private val closingFlag = new ClosingFlag

    override def close() = {
      closingFlag.close()
    }

    override protected def finalize(): Unit = {
      closingFlag.assertClosed()
    }
  }

  object Batch {

    /** @template */
    type Aux[+Data0, -Delta0] = Batch {
      type Data <: Data0
      type Delta >: Delta0
    }

  }

  /**
    * Batch是对Data和Delta的封装，每个Batch都包含`backward()`,详细信息可参看[[Layer]]
    */
  trait Batch extends AutoCloseable {
    type Data
    type Delta

    def addReference(): Batch.Aux[Data, Delta]

    protected def forceBackward(delta: Delta): Unit

    def isTrainable: Boolean

    @inline
    final def backward(delta: => Delta): Unit = {
      if (isTrainable) {
        forceBackward(delta)
      }
    }

    def value: Data
  }

  /** @template */
  type Aux[-Input0 <: Batch, +Output0 <: Batch] =
    Layer {
      type Input >: Input0
      type Output <: Output0
    }

}

/**
  * 一个Layer表示一个神经网络。每个Layer可以作为子网络被包含在其它Layer中，构成更复杂的神经网络。Layer的嵌套结构可以用来表示数学公式或粗粒度神经网络结构。
  * 当神经网络被编写完成后，其中大部分元素都是占位符，当网络开始运行时数据才真正进入到网络。
  *
  * {{{
  * val myLayer: Layer.Aux[Batch.Aux[Double, Double], Batch.Aux[Double, Double]] = {
  *   Times(Plus(Literal(1.0), Identity[Double, Double]()), Literal(2.0))
  * }
  * }}}
  *
  * 以上代码等价的数学公式可以用[[Symbolic]]风格API写作：`(1.0 + x) * 2.0`。
  * [[com.thoughtworks.deeplearning.Double.Times]]、[[com.thoughtworks.deeplearning.Double.Plus]]都是 case class，因此myLayer是一个case class构成的嵌套结构的树。
  *
  * [[com.thoughtworks.deeplearning.Symbolic.Literal]]是一个包含常量的Layer。
  * [[com.thoughtworks.deeplearning.Symbolic.Identity]]是一个输入和输出相同的Layer，它会将输入原样返回。Identity在这里是Input的占位符（Placeholder）
  *
  * 对于一个`Layer.Aux[A,B]`，`A`是输入，`B`是输出，`A`和`B`都是一个[[com.thoughtworks.deeplearning.Layer.Batch]]，
  * 给定一个a 其类型是 A，当网络开始运行时 a将被从最外层的Times一层一层向内传递最终传递给Identity，然后开始`forward`阶段，这时Times将从外向内一层一层递归调用`forward`，然后一层一层返回。
  * `backward`阶段从最外层的Times开始，首先调用Times的outputBatch，Times的outputBatch就是Plus的outputBatch，Plus的outputBatch就是Identity的outputBatch，
  * Identity的outputBatch最终是Input
  *
  * [[https://gigiigig.github.io/posts/2015/09/13/aux-pattern.html Aux]]是一种实现了[[https://www.scala-lang.org/files/archive/spec/2.12/03-types.html type refinement]]的设计模式，可以用来限制类型参数的范围。
  *
  * Times()`和`*`是等价的(可以参考[[com.thoughtworks.deeplearning.DifferentiableDouble#`Double*Double`]]的具体实现),
  * `*`只是一个语法糖，其实最终还是调用`Times()`。嵌套方法调用是一种将多个Layer组合到一起的方法。
  *
  * 将从最外面的Times()开始递归调用里面Plus()的forward方法，然后从里面的Plus()的Output开始递归调用外面Times()的backward方法。
  *
  * 当一个神经网络调用[[com.thoughtworks.deeplearning.DifferentiableAny#train]]时，会首先触发`forward`，开始一层一层递归调用内层网络的`forward`，
  * `forward`调用结束后开始递归调用外层网络的`backward`，
  *
  * [[com.thoughtworks.deeplearning.DifferentiableAny#Compose]]是另外一种组合Layer的方法：
  *
  * {{{
  *  Compose(layer1,layer2)
  * }}}
  *
  * 上述代码代表一种网络结构，layer2的output作为layer1的input然后组合成的新网络结构。
  *
  * layer2首先开始forward然后调用layer1的forward，backward则是先从layer1开始然后layer2开始backward。
  *
  *
  * Layer包括Input，Output和forward，Input和Output都是[[com.thoughtworks.deeplearning.Layer.Batch]],
  * 而Batch包含[[com.thoughtworks.deeplearning.Layer.Batch.backward()]],所以Layer所组成的神经网络会包含输入和输出，正向传播和反向传播。
  */
trait Layer {

  import Layer._

  type Input <: Batch

  type Output <: Batch

  def forward(input: Input): Output

}
