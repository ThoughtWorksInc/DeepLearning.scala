package com.qifun.statelessFuture

import scala.collection.LinearSeq
import scala.collection.generic.CanBuildFrom
import scala.collection.GenTraversableOnce
import scala.collection.generic.CanCombineFrom
import scala.collection.parallel.Combiner

object AwaitableSeq {

  import scala.language.implicitConversions

  implicit final class LinearSeqAsAwaitableSeq[A](underlying: LinearSeq[A]) {
    final def asFutureSeq = new AwaitableSeq[A, Unit](underlying)
    final def asAwaitableSeq[TailRecResult] = new AwaitableSeq[A, TailRecResult](underlying)
  }

  implicit class GeneratorAsAwaitableSeq[A, Origin](origin: Origin)(implicit toFuture: Origin => Generator[A]#Future[Unit]) {
    final def asFutureSeq = new AwaitableSeq[A, Unit](toFuture(origin))
    final def asAwaitableSeq[TailRecResult] = new AwaitableSeq[A, TailRecResult](toFuture(origin))
  }

  implicit final class ArrayAsAwaitableSeq[A](underlying: Array[A]) {
    final def asFutureSeq = new AwaitableSeq[A, Unit](underlying: Generator[A]#Future[Unit])
    final def asAwaitableSeq[TailRecResult] = new AwaitableSeq[A, TailRecResult](underlying: Generator[A]#Future[Unit])
  }

  implicit final class IndexedSeqAsAwaitableSeq[A](underlying: IndexedSeq[A]) {
    final def asFutureSeq = new AwaitableSeq[A, Unit](underlying: Generator[A]#Future[Unit])
    final def asAwaitableSeq[TailRecResult] = new AwaitableSeq[A, TailRecResult](underlying: Generator[A]#Future[Unit])
  }

  implicit final class IterableAsAwaitableSeq[A](underlying: Iterable[A]) {
    final def asFutureSeq = new AwaitableSeq[A, Unit](underlying: Generator[A]#Future[Unit])
    final def asAwaitableSeq[TailRecResult] = new AwaitableSeq[A, TailRecResult](underlying: Generator[A]#Future[Unit])
  }

  implicit final class IteratorAsAwaitableSeq[A](underlying: Iterator[A]) {
    final def asFutureSeq = new AwaitableSeq[A, Unit](underlying: Generator[A]#Future[Unit])
    final def asAwaitableSeq[TailRecResult] = new AwaitableSeq[A, TailRecResult](underlying: Generator[A]#Future[Unit])
  }

}

final class AwaitableSeq[A, TailRecResult] private (underlying: LinearSeq[A]) {

  private type Future[AwaitResult] = Awaitable[AwaitResult, TailRecResult]

  final def foldRight[B](right: B)(converter: (A, B) => Future[B]): Future[B] = {
    if (underlying.nonEmpty) {
      Awaitable[B, TailRecResult] {
        converter(underlying.head, new AwaitableSeq(underlying.tail).foldRight(right)(converter).await).await
      }
    } else {
      Awaitable[B, TailRecResult] { right }
    }
  }

  final def foldLeft[B](left: B)(converter: (B, A) => Future[B]): Future[B] = {
    if (underlying.nonEmpty) {
      Awaitable[B, TailRecResult] {
        new AwaitableSeq(underlying.tail).foldLeft(converter(left, underlying.head).await)(converter).await
      }
    } else {
      Awaitable[B, TailRecResult] { left }
    }
  }

  final def flatMap[B, That](f: A => Future[That])(implicit toGenerator: That => Generator[B]#Future[Unit]) = Awaitable[Generator.Seq[B], TailRecResult] {
    foldLeft(Generator[B].Future {}) { (left, current) =>
      f(current).map { that =>
        Generator[B].Future {
          left.await
          toGenerator(that).await
        }
      }
    }.await
  }

  final def map[B, That](f: A => Future[B]) = Awaitable[Generator.Seq[B], TailRecResult] {
    foldLeft(Generator[B].Future {}) { (left, current) =>
      f(current).map { that =>
        Generator[B].Future {
          left.await
          Generator[B].apply(that).await
        }
      }
    }.await
  }

  final def foreach[U](f: A => Future[U]) = Awaitable[Unit, TailRecResult] {
    foldLeft[Any](()) { (left, current) =>
      f(current)
    }.await
  }

  final def withFilter(f: A => Boolean): AwaitableSeq[A, TailRecResult] = {
    new AwaitableSeq(
      underlying.foldLeft(Generator[A].Future {}) { (left, current) =>
        if (f(current)) {
          Generator[A].Future {
            left.await
            Generator[A].apply(current).await
          }
        } else {
          left
        }
      })
  }
  //
  //  final def withFilter(f: A => Future[Boolean]) = Awaitable[Generator.Seq[A], TailRecResult] {
  //    foldLeft(Generator[A].Future {}) { (left, current) =>
  //      f(current).map { that =>
  //        if (that) {
  //          Generator[A].Future {
  //            left.await
  //            Generator[A].apply(current).await
  //          }
  //        } else {
  //          left
  //        }
  //      }
  //    }.await
  //  }

}