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

/**
  * @example xxx
  *          {{{
  *          import com.thoughtworks.feature.Factory
  *          import com.thoughtworks.deeplearning.plugins._
  *          val hyperparameters = Factory[FloatLiterals with FloatTraining with RawFloatLayers with ImplicitsSingleton].newInstance()
  *          import hyperparameters.implicits._
  *          val network = (- (- (- FloatLayerOps(3.4f))))
  *          network.predict
  *          }}}
  *
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
trait RawFloatLayers extends Layers {

  trait ImplicitsApi extends super[Layers].ImplicitsApi {

    implicit final class FloatLayerOps[Operand0](operand0: Operand0)(
        implicit deepLearning: DeepLearning.Aux[Operand0, Float, Float]) {
      def unary_-(implicit implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]): implicitApply.Out = {
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

    implicit def `Float+Float`[Operand0, Operand1](implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
                                                   deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
                                                   implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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

    implicit def `Float-Float`[Operand0, Operand1](implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
                                                   deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
                                                   implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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

    implicit def `Float*Float`[Operand0, Operand1](implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
                                                   deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
                                                   implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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

    implicit def `Float/Float`[Operand0, Operand1](implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
                                                   deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
                                                   implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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

    implicit def `min(Float,Float)`[Operand0, Operand1](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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

    implicit def `max(Float,Float)`[Operand0, Operand1](
        implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
        deepLearning1: DeepLearning.Aux[Operand1, Float, Float],
        implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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

    implicit def `log(Float)`[Operand0](implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
                                        implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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

    implicit def `exp(Float)`[Operand0](implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
                                        implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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

    implicit def `abs(Float)`[Operand0](implicit deepLearning0: DeepLearning.Aux[Operand0, Float, Float],
                                        implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]) = {
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
    protected val rawForward: Do[Tape[Float, Float]]

    override def forward: Do[Tape[Float, Float]] = rawForward
  }
  object FloatLayer {
    def apply(forward: Do[Tape[Float, Float]])(
        implicit implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]): implicitApply.Out = {
      implicitApply(floatPartialApplyRawForward(floatLayerFactory.newInstance, floatRawForwardParameter(forward)))
    }

    def unary[Operand0, Input0Data, Input0Delta](
        operand0: Operand0
    )(f: Input0Data => (Float, Float => Input0Delta))(
        implicit deepLearning0: DeepLearning.Aux[Operand0, Input0Data, Input0Delta],
        implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]
    ): implicitApply.Out = {
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
    def binary[Operand0, Operand1, Input0Data, Input0Delta, Input1Data, Input1Delta](
        operand0: Operand0,
        operand1: Operand1
    )(f: (Input0Data, Input1Data) => (Float, Float => Input0Delta, Float => Input1Delta))(
        implicit deepLearning0: DeepLearning.Aux[Operand0, Input0Data, Input0Delta],
        deepLearning1: DeepLearning.Aux[Operand1, Input1Data, Input1Delta],
        implicitApply: ImplicitApply[floatPartialApplyRawForward.Rest]
    ): implicitApply.Out = {
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
