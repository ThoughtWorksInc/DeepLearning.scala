package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.raii.asynchronous.Do
import com.thoughtworks.raii.asynchronous.Do._

import scalaz.syntax.all._
import scala.annotation.meta.getter
import scalaz.Apply
import scalaz.concurrent.Future
import DeepLearning.ops._

/** A plugin that provides differentiable operators
  * on neural networks whose [[DeepLearning.Data Data]] and [[DeepLearning.Delta Delta]] is [[scala.Float]].
  *
  * @note By default, the computation in a [[FloatLayer]] will re-evaluate again and again
  *       if the `FloatLayer` is used by multiple other operations.
  *
  *       This behavior is very inefficient if there is are diamond dependencies in a neural network.
  *       It's wise to use [[CumulativeFloatLayers]] instead of this `FloatLayers` in such neural network.
  *
  * @author 杨博 (Yang Bo)
  */
trait FloatLayers extends Layers {

  trait ImplicitsApi extends super[Layers].ImplicitsApi {

    /** An implicit wrapper that adds extension methods for differentiable float types
      * that support the [[DeepLearning]] type class.
      */
    implicit final class FloatLayerOps[Operand0](operand0: Operand0)(
        implicit deepLearning: DeepLearning.Aux[Operand0, Float, Float]) {
      /** @usecase def unary_- : FloatLayer = ???
        */
      def unary_-[Out <: FloatLayer](
          implicit implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]): Out = {
        FloatLayer(
          operand0.forward.map { tape0 =>
            val data = -tape0.data
            def backward(doOutputDelta: Do[Float]) = {
              tape0.backward(doOutputDelta.map(-_))
            }
            Tape(data, backward)
          }
        )
      }
    }

    implicit def `Float+Float`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.+.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) { (tape0, tape1) =>
          val data = tape0.data + tape1.data
          def backward(doOutputDelta: Do[Float]) = {
            tape0.backward(doOutputDelta) >>
              tape1.backward(doOutputDelta)
          }
          Tape(data, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `Float-Float`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.-.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) { (tape0, tape1) =>
          val data = tape0.data - tape1.data
          def backward(doOutputDelta: Do[Float]) = {
            tape0.backward(doOutputDelta) >>
              tape1.backward(doOutputDelta.map(-_))
          }
          Tape(data, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `Float*Float`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.*.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) { (tape0, tape1) =>
          val data = tape0.data * tape1.data
          def backward(doOutputDelta: Do[Float]) = {
            tape0.backward(doOutputDelta.map(_ * tape1.data)) >>
              tape1.backward(doOutputDelta.map(_ * tape0.data))
          }
          Tape(data, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `Float/Float`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators./.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) { (tape0, tape1) =>
          val data = tape0.data / tape1.data
          def backward(doOutputDelta: Do[Float]) = {
            tape0.backward(doOutputDelta.map(_ / tape1.data)) >>
              tape1.backward(doOutputDelta.map(-_ * tape0.data / (tape1.data * tape1.data)))
          }
          Tape(data, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `min(Float,Float)`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.min.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) { (tape0, tape1) =>
          if (tape0.data < tape1.data) {
            tape0
          } else {
            tape1
          }
        }
        FloatLayer(forward)
      }
    }

    implicit def `max(Float,Float)`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.max.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) { (tape0, tape1) =>
          if (tape0.data > tape1.data) {
            tape0
          } else {
            tape1
          }
        }
        FloatLayer(forward)
      }
    }

    implicit def `log(Float)`[Operand0, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.log.at[Operand0] { operand0 =>
        val forward = operand0.forward.map { tape0 =>
          val data = math.log(tape0.data).toFloat
          def backward(doOutputDelta: Do[Float]) = {
            tape0.backward(doOutputDelta.map(_ / tape0.data))
          }
          Tape(data, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `exp(Float)`[Operand0, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.exp.at[Operand0] { operand0 =>
        val forward = operand0.forward.map { tape0 =>
          val data = math.exp(tape0.data).toFloat
          def backward(doOutputDelta: Do[Float]) = {
            val doDelta = doOutputDelta.map { outputDelta =>
              outputDelta * data
            }
            tape0.backward(doDelta)
          }
          Tape(data, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `abs(Float)`[Operand0, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.abs.at[Operand0] { operand0 =>
        val forward = operand0.forward.map { tape0 =>
          if (tape0.data < 0) {
            val data = -tape0.data
            def backward(doOutputDelta: Do[Float]) = {
              tape0.backward(doOutputDelta.map(-_))
            }
            Tape(data, backward)
          } else {
            tape0
          }

        }
        FloatLayer(forward)
      }
    }

  }

  override type Implicits <: ImplicitsApi

  /** @template */
  type FloatLayer <: FloatLayerApi with Layer

  @inject
  protected val floatLayerFactory: Factory[FloatLayer]

  @inject
  protected def floatRawForwardParameter: Do[Tape[Float, Float]] <:< floatPartialApplyRawForward.Parameter

  @inject
  protected val floatPartialApplyRawForward: PartialApply[floatLayerFactory.Constructor,
                                                          shapeless.Witness.`"rawForward"`.T]

  trait FloatLayerApi extends super[Layers].LayerApi {
    override type Data = Float
    override type Delta = Float

    /** The original forward operation passed in [[FloatLayer$ FloatLayer.apply]].
      *
      * @note This [[rawForward]] may be different from [[forward]],
      *       in the case of [[forward]] was overriden by other plugins, e.g. [[CumulativeFloatLayers]].
      */
    protected val rawForward: Do[Tape[Float, Float]]

    override def forward: Do[Tape[Float, Float]] = rawForward
  }
  object FloatLayer {

    /** @usecase def apply(forward: Do[Tape[Float, Float]]): FloatLayer = ???
      *
      *          Returns a [[FloatLayer]] according to the given `forward` operation.
      */
    def apply[Out <: FloatLayer](forward: Do[Tape[Float, Float]])(
        implicit implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]): Out = {
      implicitApply(floatPartialApplyRawForward(floatLayerFactory.newInstance, floatRawForwardParameter(forward)))
    }

    /** Internal helper to create unary [[FloatLayer]]. */
    def unary[Operand0, Input0Data, Input0Delta, Out <: FloatLayer](
        operand0: Operand0
    )(f: Input0Data => (Float, Float => Input0Delta))(
        implicit deepLearning0: DeepLearning.Aux[Operand0, Input0Data, Input0Delta],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]
    ): Out = {
      FloatLayer(deepLearning0.forward(operand0).map {
        case Tape(data0, backward0) =>
          val (outputData, delta0) = f(data0)
          def backward(doOutputDelta: Do[Float]) = {
            backward0(doOutputDelta.map { outputDelta =>
              delta0(outputDelta)
            })
          }
          Tape(outputData, backward)
      })
    }

    /** Internal helper to create unary [[FloatLayer]]. */
    def binary[Operand0, Operand1, Input0Data, Input0Delta, Input1Data, Input1Delta, Out <: FloatLayer](
        operand0: Operand0,
        operand1: Operand1
    )(f: (Input0Data, Input1Data) => (Float, Float => Input0Delta, Float => Input1Delta))(
        implicit deepLearning0: DeepLearning.Aux[Operand0, Input0Data, Input0Delta],
        deepLearning1: DeepLearning.Aux[Operand1, Input1Data, Input1Delta],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]
    ): Out = {
      // TODO
      FloatLayer(Apply[Do].apply2(deepLearning0.forward(operand0), deepLearning1.forward(operand1)) {
        case (Tape(data0, backward0), Tape(data1, backward1)) =>
          val (outputData, delta0, delta1) = f(data0, data1)
          def backward(doOutputDelta: Do[Float]) = {
            backward0(doOutputDelta.map { outputDelta =>
              delta0(outputDelta)
            }) >> backward1(doOutputDelta.map { outputDelta =>
              delta1(outputDelta)
            })
          }
          Tape(outputData, backward)

      })
    }
  }
}
