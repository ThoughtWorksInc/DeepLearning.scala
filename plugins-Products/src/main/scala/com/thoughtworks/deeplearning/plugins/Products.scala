package com.thoughtworks.deeplearning
package plugins
import com.thoughtworks.continuation.UnitContinuation
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous._
import shapeless.{::, Generic, HList}
import scalaz.syntax.all._
import shapeless.ops.hlist.Tupler

/**
  * @author 杨博 (Yang Bo)
  */
trait Products extends HLists {

  trait ImplicitsApi extends super.ImplicitsApi {

    implicit def productDeepLearning[Operand, L, DataHList <: HList, DeltaHList <: HList, DataTuple, DeltaTuple](
        implicit generic: Generic.Aux[Operand, L],
        deepLearning: DeepLearning.Aux[L, DataHList, DeltaHList],
        dataTupler: Tupler.Aux[DataHList, DataTuple],
        deltaTupler: Tupler.Aux[DeltaHList, DeltaTuple],
        genericDelta: Generic.Aux[DeltaTuple, DeltaHList]
    ): DeepLearning.Aux[Operand, DataTuple, DeltaTuple] =
      new DeepLearning[Operand] {
        type Data = DataTuple
        type Delta = DeltaTuple
        def forward(differentiable: Operand): Do[Tape[Data, Delta]] = {
          deepLearning.forward(generic.to(differentiable)).map {
            case Tape(hlistData, hlistForward) =>
              def forward(doDeltaTuple: Do[DeltaTuple]): UnitContinuation[Unit] = {
                hlistForward(doDeltaTuple.map(genericDelta.to))
              }
              Tape(dataTupler(hlistData), forward)
          }
        }
      }
  }

  type Implicits <: ImplicitsApi
}
