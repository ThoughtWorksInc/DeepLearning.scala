package com.qifun.statelessFuture

import scala.collection.LinearSeq
import scala.collection.generic.CanBuildFrom
import scala.collection.GenTraversableOnce
import scala.collection.generic.CanCombineFrom
import scala.collection.parallel.Combiner
import scala.reflect.macros.Context

object AwaitableSeq {

  def apply[A, TailRecResult](underlying: LinearSeq[A]) = new AwaitableSeq[A, TailRecResult](underlying)

  /**
   * @usecase def apply[A, TailRecResult](collection: Iterable[A]): AwaitableSeq[A, TailRecResult] = ???
   */
  def apply[A, TailRecResult, Origin](origin: Origin)(implicit toFuture: Origin => Generator[A]#Future[Unit]) = new AwaitableSeq[A, TailRecResult](toFuture(origin))

  type FutureSeq[A] = AwaitableSeq[A, Unit]

  object FutureSeq {
    def apply[A](underlying: LinearSeq[A]) = new FutureSeq[A](underlying)

    /**
     * @usecase def apply[A](collection: Iterable[A]): FutureSeq[A] = ???
     */
    def apply[A, Origin](underlying: Origin)(implicit toFuture: Origin => Generator[A]#Future[Unit]) = new FutureSeq[A](toFuture(underlying))
  }

  final def flatMapMacro(c: Context)(f: c.Expr[Nothing => Any]): c.Expr[Nothing] = {
    import c.universe._

    val Apply(TypeApply(Select(thisTree, _), List(b, _)), _) = c.macroApplication
    val thisName = newTermName(c.fresh("yangBoAwaitableSeq"))
    val thisVal = c.Expr(ValDef(Modifiers(), thisName, TypeTree(), thisTree))
    val TypeRef(_, _, List(_, tailRecResultType)) = thisTree.tpe.widen
    val functioExpr: c.Expr[Awaitable[Nothing, Nothing]] = f.tree match {
      case fTree @ Function(vparams, body) => {
        val trrTree = Select(Ident(thisName), newTypeName("TailRecResult"))
        c.Expr(Apply(
          Select(Ident(thisName), newTermName("awaitableFlatMap")),
          List(Function(
            vparams,
            ANormalForm.applyMacroWithType(c)(
              c.Expr(body),
              AppliedTypeTree(Ident(typeOf[AwaitableSeq[_, _]].typeSymbol), List(b, trrTree)),
              trrTree).tree))))
      }
      case fTree => {
        c.error(fTree.pos, "Expcet function literal.")
        reify { ??? }
      }
    }
    reify {
      thisVal.splice
      functioExpr.splice.await
    }
  }

  final def mapMacro(c: Context)(f: c.Expr[Nothing => Any]): c.Expr[Nothing] = {
    import c.universe._
    val Apply(TypeApply(Select(thisTree, _), List(b, _)), _) = c.macroApplication
    val thisName = newTermName(c.fresh("yangBoAwaitableSeq"))
    val thisVal = c.Expr(ValDef(Modifiers(), thisName, TypeTree(), thisTree))
    val TypeRef(_, _, List(_, tailRecResultType)) = thisTree.tpe.widen
    val functioExpr: c.Expr[Awaitable[Nothing, Nothing]] = f.tree match {
      case fTree @ Function(vparams, body) => {
        c.Expr(Apply(
          Select(Ident(thisName), newTermName("awaitableMap")),
          List(Function(
            vparams,
            ANormalForm.applyMacroWithType(c)(
              c.Expr(body),
              b,
              Select(Ident(thisName), newTypeName("TailRecResult"))).tree))))
      }
      case fTree => {
        c.error(fTree.pos, "Expcet function literal.")
        reify { ??? }
      }
    }
    reify {
      thisVal.splice
      functioExpr.splice.await
    }
  }

  final def foreachMacro(c: Context)(f: c.Expr[Nothing => Any]): c.Expr[Unit] = {
    import c.universe._
    val Apply(TypeApply(Select(thisTree, _), List(u)), _) = c.macroApplication
    val thisName = newTermName(c.fresh("yangBoAwaitableSeq"))
    val thisVal = c.Expr(ValDef(Modifiers(), thisName, TypeTree(), thisTree))
    val TypeRef(_, _, List(_, tailRecResultType)) = thisTree.tpe.widen
    val functioExpr: c.Expr[Awaitable[Nothing, Nothing]] = f.tree match {
      case fTree @ Function(vparams, body) => {
        c.Expr(Apply(
          Select(Ident(thisName), newTermName("awaitableForeach")),
          List(Function(
            vparams,
            ANormalForm.applyMacroWithType(c)(
              c.Expr(body),
              u,
              Select(Ident(thisName), newTypeName("TailRecResult"))).tree))))
      }
      case fTree => {
        c.error(fTree.pos, "Expcet function literal.")
        reify { ??? }
      }
    }
    reify {
      thisVal.splice
      functioExpr.splice.await
    }
  }

}

final class AwaitableSeq[A, TRR](val underlying: LinearSeq[A]) {

  type TailRecResult = TRR

  private type Future[AwaitResult] = Awaitable[AwaitResult, TailRecResult]

  private type StatelessFuture[AwaitResult] = Awaitable[AwaitResult, TailRecResult]

  final def foldRight[B](right: B)(converter: (A, B) => Future[B]): StatelessFuture[B] = {
    if (underlying.nonEmpty) {
      Awaitable[B, TailRecResult] {
        converter(underlying.head, new AwaitableSeq(underlying.tail).foldRight(right)(converter).await).await
      }
    } else {
      Awaitable[B, TailRecResult] { right }
    }
  }

  final def foldLeft[B](left: B)(converter: (B, A) => Future[B]): StatelessFuture[B] = {
    if (underlying.nonEmpty) {
      Awaitable[B, TailRecResult] {
        new AwaitableSeq(underlying.tail).foldLeft(converter(left, underlying.head).await)(converter).await
      }
    } else {
      Awaitable[B, TailRecResult] { left }
    }
  }

  final def awaitableFlatMap[B, That](f: A => Future[AwaitableSeq[B, TailRecResult]]) = Awaitable[AwaitableSeq[B, TailRecResult], TailRecResult] {
    new AwaitableSeq[B, TailRecResult](
      foldLeft(Generator[B].Future {}) { (left, current) =>
        f(current).map { that =>
          Generator[B].Future {
            left.await
            Generator.linearSeqToFuture(that.underlying).await
          }
        }
      }.await)
  }

  final def awaitableMap[B, That](f: A => Future[B]) = Awaitable[AwaitableSeq[B, TailRecResult], TailRecResult] {
    new AwaitableSeq[B, TailRecResult](
      Generator.futureToGeneratorSeq(
        foldLeft(Generator[B].Future {}) { (left, current) =>
          f(current).map { that =>
            Generator[B].Future {
              left.await
              Generator[B].apply(that).await
            }
          }
        }.await))
  }

  final def awaitableForeach[U](f: A => Future[U]) = Awaitable[Unit, TailRecResult] {
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

  import scala.language.experimental.macros

  final def flatMap[B, That](f: A => AwaitableSeq[B, TailRecResult]): AwaitableSeq[B, TailRecResult] = macro AwaitableSeq.flatMapMacro
  final def map[B, That](f: A => B): AwaitableSeq[B, TailRecResult] = macro AwaitableSeq.mapMacro
  final def foreach[U](f: A => U): Unit = macro AwaitableSeq.foreachMacro
}
