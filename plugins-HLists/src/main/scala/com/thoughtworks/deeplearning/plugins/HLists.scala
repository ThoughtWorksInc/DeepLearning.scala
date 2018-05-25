package com.thoughtworks.deeplearning.plugins

import com.thoughtworks.continuation._
import com.thoughtworks.future._
import com.thoughtworks.deeplearning.DeepLearning
import com.thoughtworks.deeplearning.DeepLearning.Tape
import com.thoughtworks.raii.asynchronous._
import scalaz.Applicative
import scalaz.syntax.all._
import scalaz.Tags.Parallel
import shapeless.{::, HList, HNil}

import java.io.{PrintStream, PrintWriter}

import scalaz.Semigroup

private object HLists {

  implicit val doParallelApplicative = asynchronousDoParallelApplicative

  private val noop: Do[HNil] => UnitContinuation[Unit] = {
    Function.const(UnitContinuation.now(()))
  }

}

/**
  * @author 杨博 (Yang Bo)
  */
trait HLists {
  import com.thoughtworks.deeplearning.plugins.HLists._

  trait ImplicitsApi {
    implicit def hnilDeepLearning[L <: HNil]: DeepLearning.Aux[L, HNil, HNil] = new DeepLearning[L] {
      type Data = HNil
      type Delta = HNil

      def forward(differentiable: L): Do[Tape[Data, Delta]] = {
        Do.now(Tape(HNil, noop))
      }
    }

    implicit def hconsDeepLearning[Head, Tail <: HList, HeadData, TailData <: HList, HeadDelta, TailDelta <: HList](
        implicit headDeepLearning: DeepLearning.Aux[Head, HeadData, HeadDelta],
        tailDeepLearning: DeepLearning.Aux[Tail, TailData, TailDelta])
      : DeepLearning.Aux[Head :: Tail, HeadData :: TailData, HeadDelta :: TailDelta] = new DeepLearning[Head :: Tail] {
      type Data = HeadData :: TailData
      type Delta = HeadDelta :: TailDelta

      def forward(differentiable: Head :: Tail): Do[Tape[Data, Delta]] = {
        val head :: tail = differentiable
        val doHead: ParallelDo[Tape[HeadData, HeadDelta]] = Parallel(headDeepLearning.forward(head))

        val doTail: ParallelDo[Tape[TailData, TailDelta]] = Parallel(tailDeepLearning.forward(tail))

        Parallel.unwrap(doParallelApplicative.tuple2(doHead, doTail)).map {
          case (Tape(headData, headBackward), Tape(tailData, tailBackward)) =>
            def backward(doDelta: Do[HeadDelta :: TailDelta]) = {
              val continuationHead: ParallelContinuation[Unit] = Parallel(headBackward(doDelta.map(_.head)))
              val continuationTail: ParallelContinuation[Unit] = Parallel(tailBackward(doDelta.map(_.tail)))
              Parallel.unwrap(continuationParallelApplicative.apply2(continuationHead, continuationTail) {
                (_: Unit, _: Unit) =>
                  ()
              })
            }
            Tape(headData :: tailData, backward)
        }

      }

    }
  }

  type Implicits <: ImplicitsApi

}
