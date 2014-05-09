package com.qifun.statelessFuture

import scala.collection.LinearSeqOptimized
import scala.collection.immutable.LinearSeq
import scala.collection.generic.GenericTraversableTemplate
import scala.collection.generic.SeqFactory
import scala.util.control.TailCalls._
import scala.util.control.Exception.Catcher

final object Generator extends SeqFactory[Generator] {
  
  def main(args:Array[String]) {
    val i = newBuilder[Int]
    i += 1
    println(i.result)
  }
  
  final class Builder[Element](private var future: Awaitable[Unit, Generator[Element]] = null) extends scala.collection.mutable.Builder[Element, Generator[Element]] {

    override final def clear() {
      future = null
    }

    override final def +=(element: Element): this.type = {
      if (future == null) {
        future = Yield[Element](element)
      } else {
        val oldFuture = future
        future = Awaitable[Unit, Generator[Element]] {
          oldFuture.await
          Yield[Element](element).await
        }
      }
      this
    }

    override final def result(): Generator[Element] = {
      future match {
        case null => Generator.Empty
        case notNull => notNull
      }
    }
  }

  override final def newBuilder[Element] = new Builder

  private final case object Empty extends Generator[Nothing] {

    override final def isEmpty: Boolean = true

    override final def head: Nothing = {
      throw new NoSuchElementException("head of empty list")
    }

    override final def tail: Nothing = {
      throw new UnsupportedOperationException("tail of empty list")
    }

  }

  private final case class NonEmpty[+Element](
    override val head: Element,
    val continue: Unit => TailRec[Generator[Element]]) extends Generator[Element] {

    override final def isEmpty: Boolean = false

    override final def tail: Generator[Element] = continue(()).result

  }

  import scala.language.implicitConversions

  @inline
  implicit final def fromFuture[Unit, Element](future: Awaitable[Unit, Generator[Element]]): Generator[Element] = {
    future.onComplete { u => done(Empty) }(PartialFunction.empty).result
  }

  trait Yield[Element] extends (Element => Awaitable[Unit, Generator[Element]]) {

    type Future[AwaitResult] = Awaitable[AwaitResult, Generator[Element]]

    object Future extends AwaitableFactory[Generator[Element]]

    @inline
    final def apply(element: Element): Future[Unit] = new Awaitable.Stateless[Unit, Generator[Element]] {
      private type TailRecResult = Generator[Element]
      private type AwaitResult = Unit
      override final def onComplete(handler: AwaitResult => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
        done(NonEmpty(element, handler))
      }
    }

  }

  @inline
  final def Yield[Element] = new Yield[Element] {}

}

sealed abstract class Generator[+Element] extends Seq[Element]
  with LinearSeq[Element]
  with Product
  with GenericTraversableTemplate[Element, Generator]
  with LinearSeqOptimized[Element, Generator[Element]] {

  override final def companion = Generator

}
