package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.feature.{Factory, ImplicitApply, PartialApply}
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.raii.asynchronous._
import scalaz.syntax.all._

import scala.annotation.meta.getter
import scalaz.Apply
import com.thoughtworks.continuation._
import com.thoughtworks.future._
import DeepLearning.ops._
import com.thoughtworks.deeplearning.plugins.Layers.ToLayer
import com.thoughtworks.deeplearning.plugins.Layers.Eager
import com.thoughtworks.dsl.Dsl

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
    implicit def eagerFloatDsl[Differentiable, Data, Delta, Constructor, Out <: FloatLayer](
        implicit implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]
    ): Dsl[Eager[Differentiable, Data, Delta], FloatLayer, Data] = {
      new Dsl[Eager[Differentiable, Data, Delta], FloatLayer, Data] {
        def interpret(keyword: Eager[Differentiable, Data, Delta], handler: Data => FloatLayer): Out =
          FloatLayer(
            keyword.deepLearning.forward(keyword.operand0).flatMap { tape =>
              handler(tape.data).internalForward
            }
          )
      }
    }

    implicit def toFloatLayer[Out <: FloatLayer](
        implicit implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out])
      : Layers.ToLayer.Aux[Float, Float, FloatLayer] = new ToLayer[Float, Float] {
      type OutputLayer = FloatLayer

      override def toLayer(forward: Do[Tape[Float, Float]]): FloatLayer = {
        implicitApply(floatPartialApplyRawForward(floatLayerFactory.newInstance, floatRawForwardParameter(forward)))
      }
    }

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
          operand0.forward.map {
            case tape0 @ Tape(data0, backward0) =>
              val outputData = -data0
              def backward(doOutputDelta: Do[Float]) = {
                backward0(doOutputDelta.map(-_))
              }
              Tape(outputData, backward)
          }
        )
      }
    }

    implicit def `Float+Float`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.+.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) {
          case (tape0 @ Tape(data0, backward0), tape1 @ Tape(data1, backward1)) =>
            val outputData = data0 + data1
            def backward(doOutputDelta: Do[Float]) = {
              backward0(doOutputDelta) >>
                backward1(doOutputDelta)
            }
            Tape(outputData, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `Float-Float`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.-.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward: Do[Tape[Float, Float]] =
          Apply[Do].apply2(operand0.forward /* deepLearning0.forward(operand0) */, operand1.forward) {
            case (tape0 @ Tape(data0, backward0), tape1 @ Tape(data1, backward1)) =>
              val outputData = data0 - data1
              def backward(doOutputDelta: Do[Float]): UnitContinuation[Unit] = {
                backward0(doOutputDelta) >>
                  backward1(doOutputDelta.map(-_))
              }
              Tape(outputData, backward)
          }
        FloatLayer(forward)
      }
    }

    implicit def `Float*Float`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.*.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) {
          case (tape0 @ Tape(data0, backward0), tape1 @ Tape(data1, backward1)) =>
            val outputData = data0 * data1
            def backward(doOutputDelta: Do[Float]) = {
              backward0(doOutputDelta.map(_ * data1)) >>
                backward1(doOutputDelta.map(_ * data0))
            }
            Tape(outputData, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `Float/Float`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators./.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) {
          case (tape0 @ Tape(data0, backward0), tape1 @ Tape(data1, backward1)) =>
            val outputData = data0 / data1
            def backward(doOutputDelta: Do[Float]) = {
              backward0(doOutputDelta.map(_ / data1)) >>
                backward1(doOutputDelta.map(-_ * data0 / (data1 * data1)))
            }
            Tape(outputData, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `min(Float,Float)`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.min.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) {
          case (tape0 @ Tape(data0, backward0), tape1 @ Tape(data1, backward1)) =>
            if (data0 < data1) {
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
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) {
          case (tape0 @ Tape(data0, backward0), tape1 @ Tape(data1, backward1)) =>
            if (data0 > data1) {
              tape0
            } else {
              tape1
            }
        }
        FloatLayer(forward)
      }
    }

    implicit def `pow(Float,Float)`[Operand0, Operand1, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.pow.at[Operand0, Operand1] { (operand0, operand1) =>
        val forward = Apply[Do].apply2(operand0.forward, operand1.forward) {
          case (tape0 @ Tape(data0, backward0), tape1 @ Tape(data1, backward1)) =>
            val outputData = math.pow(data0, data1).toFloat
            def backward(doOutputDelta: Do[Float]) = {
              backward0(doOutputDelta.map(_ * data1 * math.pow(data0, data1 - 1).toFloat)) >>
                backward1(doOutputDelta.map(_ * math.log(data0).toFloat * outputData))
            }
            Tape(outputData, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `log(Float)`[Operand0, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.log.at[Operand0] { operand0 =>
        val forward = operand0.forward.map {
          case tape0 @ Tape(data0, backward0) =>
            val outputData = math.log(data0).toFloat
            def backward(doOutputDelta: Do[Float]) = {
              backward0(doOutputDelta.map(_ / data0))
            }
            Tape(outputData, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `exp(Float)`[Operand0, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.exp.at[Operand0] { operand0 =>
        val forward = operand0.forward.map {
          case tape0 @ Tape(data0, backward0) =>
            val outputData = math.exp(data0).toFloat
            def backward(doOutputDelta: Do[Float]) = {
              val doDelta = doOutputDelta.map { outputDelta =>
                outputDelta * outputData
              }
              backward0(doDelta)
            }
            Tape(outputData, backward)
        }
        FloatLayer(forward)
      }
    }

    implicit def `abs(Float)`[Operand0, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.abs.at[Operand0] { operand0 =>
        val forward = operand0.forward.map {
          case tape0 @ Tape(data0, backward0) =>
            if (data0 < 0) {
              val outputData = -data0
              def backward(doOutputDelta: Do[Float]) = {
                backward0(doOutputDelta.map(-_))
              }
              Tape(outputData, backward)
            } else {
              tape0
            }

        }
        FloatLayer(forward)
      }
    }

    implicit def `sqrt(Float)`[Operand0, Out <: FloatLayer](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]) = {
      Operators.sqrt.at[Operand0] { operand0 =>
        val forward = operand0.forward.map {
          case tape0 @ Tape(data0, backward0) =>
            val outputData = math.sqrt(data0).toFloat
            def backward(doOutputDelta: Do[Float]) = {
              val doDelta = doOutputDelta.map(_ * 0.5f / outputData)
              backward0(doDelta)
            }
            Tape(outputData, backward)
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
    type Data = Float
    type Delta = Float

    final def predict: Future[Data] = {
      val doData = forward.map(_.data)
      doData.run
    }

    final def train: Future[Data] = {
      val doData = forward.flatMap[Data] { tape =>
        Do.garbageCollected(tape.backward(Do.now(1.0f.toFloat))).intransitiveMap { _: Unit =>
          tape.data
        }
      }
      doData.run
    }

    /** A bridge for calling [[rawForward]] in [[FloatLayers]] */
    private[FloatLayers] final def internalForward: Do[Tape[Float, Float]] = rawForward

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

    /** Internal helper to create an unary [[FloatLayer]]. */
    def unary[Operand0, Input0Data, Input0Delta, Out <: FloatLayer](
        operand0: Operand0
    )(f: Input0Data => (Float, Float => Input0Delta))(
        implicit deepLearning0: DeepLearning.Aux[Operand0, Input0Data, Input0Delta],
        implicitApply: ImplicitApply.Aux[floatPartialApplyRawForward.Rest, Out]
    ): Out = {
      FloatLayer(deepLearning0.forward(operand0).map {
        case tape0 @ Tape(data0, backward0) =>
          val (outputData, delta0) = f(data0)
          def backward(doOutputDelta: Do[Float]) = {
            backward0(doOutputDelta.map { outputDelta =>
              delta0(outputDelta)
            })
          }
          Tape(outputData, backward)
      })
    }

    /** Internal helper to create a binary [[FloatLayer]]. */
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
        case (tape0 @ Tape(data0, backward0), tape1 @ Tape(data1, backward1)) =>
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
