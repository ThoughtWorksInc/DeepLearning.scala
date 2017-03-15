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

  object Tape {

    /** @template */
    type Aux[+Data0, -Delta0] = Tape {
      type Data <: Data0
      type Delta >: Delta0
    }

  }

  /**
    * Tape是神经网络运行中的中间数据结构，它有包含四部分，数据(value)及其导数，`duplicate`和`backward`，
    * `value`即`forward`的计算结果，`value`还有`backward`的功能
    *
    * @see [[https://en.wikipedia.org/wiki/Automatic_differentiation tape auto differentiation]]
    */
  trait Tape extends AutoCloseable {
    type Data
    type Delta

    /**
      * Returns a new [[Tape]] that shares the same [[value]] and [[backward]] behavior with this [[Tape]].
      * @note The newly created [[Tape]] and this [[Tape]] must be [[close]]d independently.
      */
    def duplicate(): Tape.Aux[Data, Delta]

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
  type Aux[-Input0 <: Tape, +Output0 <: Tape] =
    Layer {
      type Input >: Input0
      type Output <: Output0
    }

}

/**
  * 一个Layer表示一个神经网络。每个Layer可以作为子网络被包含在其它Layer中，构成更复杂的神经网络。Layer的嵌套结构可以用来表示数学公式或粗粒度神经网络结构。
  * 当神经网络被编写完成后，其中大部分元素都是占位符，当网络开始训练时数据才真正进入到网络。
  *
  * === Layer 的树结构 ===
  * {{{
  * val myLayer: Layer.Aux[Tape.Aux[Double, Double], Tape.Aux[Double, Double]] = {
  *   Times(
  *     Plus(
  *       Literal(1.0),
  *       Identity[Double, Double]()
  *     ),
  *     Weight(2.0)
  *   )
  * }
  * }}}
  *
  * 以上代码等价的数学公式可以用[[com.thoughtworks.deeplearning.Symbolic Symbolic API]]写作：`(1.0 + x) * 2.0.toWeight`。2.0.toWeight表示一个变量，其初始值是2，在神经网络迭代时，值会更新。
  * [[com.thoughtworks.deeplearning.DifferentiableDouble.Layers.Times Times]]、[[com.thoughtworks.deeplearning.DifferentiableDouble.Plus Plus]]都是 case class，
  * 因此myLayer是一个case class构成的嵌套结构的树。`Times`和`Plus`都是占位符。
  *
  * [[com.thoughtworks.deeplearning.DifferentiableDouble.Layers.Weight Weight]]是一个包含权重的`Layer`，初始值是`2.0`。
  * [[com.thoughtworks.deeplearning.Symbolic.Layers.Identity Identity]]是一个输入和输出相同的Layer，它会将输入原样返回。`Identity`在这里是`Input`的占位符。
  * [[com.thoughtworks.deeplearning.Symbolic.Layers.Literal Literal]]是一个包含常量的`Layer`。
  *
  * === 迭代 ===
  *
  * 网络每次训练称为一个迭代，分为[[forward]]和[[com.thoughtworks.deeplearning.Layer.Tape#backward backward]]两个阶段，构成一次完整的[[https://en.wikipedia.org/wiki/Backpropagation 反向传播]]流程。
  *
  * ==== forward ====
  *
  * 在`Layer.Aux[A,B]`中调用[[forward]]时，`A`是输入类型，`B`是输出类型，`A`和`B`都是[[com.thoughtworks.deeplearning.Layer.Tape Tape]]。下面开始逐段解释代码：
  *
  * 例如：
  * {{{
  * val inputTape: Tape.Aux[Double, Double] = Literal(a)
  * val outputTape = myLayer.forward(inputTape)
  * }}}
  *
  *
  * 当调用`myLayer.forward(inputData)`时，首先调用`Times`的`forward`，其伪代码如下：
  * {{{
  * final case class Times(operand1: Layer, operand2: Layer) extends Layer {
  *   def forward(inputData: Tape): Output = {
  *     val upstream1 = operand1.forward(input)
  *     val upstream2 = operand2.forward(input)
  *     new Output(upstream1, upstream2)//这里忽略具体实现，而关注递归细节
  *   }
  *   final class Output(upstream1: Tape, upstream2: Tape) extends Tape { ... }
  * }
  * }}}
  *
  * 在`myLayer.operand1`是`Plus`,`myLayer.operand2`是`Weight`，因此，upstream1和upstream2分别是`operand1`和`operand2` `forward` 的结果。
  *
  * 以此类推，`Plus`的`forward`代码与`Times`的`forward`类似，当调用`Plus`的`forward`时，[[com.thoughtworks.deeplearning.DifferentiableDouble.Layers.Plus#operand1 operand1]]是`Literal`，[[com.thoughtworks.deeplearning.DifferentiableDouble.Layers.Plus.operand2 operand2]]是`Identity`，这时会各自调用`Literal`和`Identity`的`forward`.
  *
  * 当调用`Literal`的`forward`时会原样返回输入。
  *
  * 当调用`Identity`的`forward`时会原样返回输入， `Identity`的`forward`的伪代码如下：
  * {{{
  * def forward(inputTape: Tape.Aux[Double, Double]) = inputTape
  * }}}
  *
  * 当调用`Weight`的`forward`时会原样返回输入。
  *
  * myLayer.forward的返回值[[com.thoughtworks.deeplearning.DifferentiableDouble.Layer.Times.Output outputTape]] 是 [[com.thoughtworks.deeplearning.Layer.Tape Tape]]类型，所以最终会生成一棵[[com.thoughtworks.deeplearning.Layer.Tape Tape]]构成的树，结构和myLayer一样。
  * 因此，通过层层传播 `myLayer.forward(inputTape)`最终被Identity原样返回，组合进新生成的Tape树。
  *
  * outputTape 的包含`forward` 的计算结果，计算结果可以用来 `backward` 比如
  * {{{
  *    try {
  *      val loss = outputTape.value
  *      outputTape.backward(loss)
  *      loss
  *    } finally {
  *      outputTape.close()
  *    }
  * }}}
  *
  * outputTape.value 是数学公式 (1.0 + x) * 2.0.toWeight 的计算结果。
  *
  * ==== backward ====
  *
  * outputTape.backward，即 Times.Output 的 backward ，伪代码如下：
  * {{{
  * case class Times(operand1: Layer, operand2: Layer) extends Layer {
  *   def forward = ...
  *   class Output(upstream1, upstream2) extends Tape {
  *     private def upstreamDelta1(outputDelta: Double) = ???
  *     private def upstreamDelta2(outputDelta: Double) = ???
  *     override protected def backward(outputDelta: Double): Unit = {
  *       upstream1.backward(upstreamDelta1(outputDelta))
  *       upstream2.backward(upstreamDelta2(outputDelta))
  *     }
  *   }
  * }
  * }}}
  *
  * `outputTape.upstream1`和`outputTape.upstream2`分别是`operand1`和`operand2` `forward` 的结果。然后`outputTape.upstream1`和`outputTape.upstream2`分别进行`backward`。
  *
  * 以此类推，`Plus`的`backward`代码与`Times`的`backward`类似，当调用`Plus`的`backward`时，`upstream1`和`upstream2`分别是`Literal`和`Identity` `forward`的结果，这时会各自调用`upstream1`和`upstream2`的`backward`。
  *
  * `Weight`在`backward`时会更新`Weight`，参考[[com.thoughtworks.deeplearning.DifferentiableDouble.LearningRate#updateDouble updateDouble]]
  *
  * === Aux & Symbolic API===
  *
  * Layer.Aux[A,B]表示Input的类型是A，Output的类型是B。Tape.Aux[C,D]表示Data的类型是C，Delta的类型是D。
  * Layer.Aux和Type.Aux可以组合起来使用，比如Layer.Aux[Tape.Aux[A,B],Tape.Aux[C,D]]可以用来表示一个layer的输入类型是一个Tape，这个Tape的数据类型为A，delta类型为B，layer的输出类型是一个Tape，这个Tape的数据类型为C，delta类型为D。
  *
  * [[https://gigiigig.github.io/posts/2015/09/13/aux-pattern.html Aux]]是一种实现了[[https://www.scala-lang.org/files/archive/spec/2.12/03-types.html type refinement]]的设计模式，可以用来限制类型参数的范围。
  *
  * [[com.thoughtworks.deeplearning.Symbolic Symbolic API]]是一个语法糖，可以减少编写Aux的繁琐，详情请看[[com.thoughtworks.deeplearning.Symbolic Symbolic API]]
  *
  * @see [[https://gigiigig.github.io/posts/2015/09/13/aux-pattern.html aux pattern]]
  * @see [[http://www.vlachjosef.com/aux-pattern-evolution/ aux pattern evolution]]
  * @see [[https://www.scala-lang.org/files/archive/spec/2.12/03-types.html type refinement]]
  * @see [[https://en.wikipedia.org/wiki/Backpropagation Backpropagation]]
  * @see [[com.thoughtworks.deeplearning.Symbolic Symbolic API]]
  */
trait Layer {

  import Layer._

  type Input <: Tape

  type Output <: Tape

  def forward(input: Input): Output

}
