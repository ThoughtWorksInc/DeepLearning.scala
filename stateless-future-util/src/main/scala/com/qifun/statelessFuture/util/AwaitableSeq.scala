/*
 * stateless-future-util
 * Copyright 2014 深圳岂凡网络有限公司 (Shenzhen QiFun Network Corp., LTD)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.qifun.statelessFuture
package util

import scala.collection.LinearSeq
import scala.reflect.macros.Context
import com.qifun.statelessFuture.ANormalForm
import com.qifun.statelessFuture.Awaitable
import com.qifun.statelessFuture.util.Generator.GeneratorSeq

object AwaitableSeq {

  final def apply[A, TailRecResult](underlying: LinearSeq[A]) =
    new AwaitableSeq[A, TailRecResult](underlying)

  final def apply[A, TailRecResult](underlying: TraversableOnce[A]) =
    new AwaitableSeq[A, TailRecResult](GeneratorSeq(underlying))

  private type FutureSeq[A] = AwaitableSeq[A, Unit]

  final def futureSeq[A](underlying: LinearSeq[A]) = new FutureSeq[A](underlying)

  final def futureSeq[A](underlying: TraversableOnce[A]) = new FutureSeq[A](GeneratorSeq(underlying))

  final def flatMapMacro(c: Context)(f: c.Expr[Nothing => Any]): c.Expr[Nothing] = {
    import c.universe._

    val Apply(TypeApply(Select(thisTree, _), List(b)), _) = c.macroApplication
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
    val Apply(TypeApply(Select(thisTree, _), List(b)), _) = c.macroApplication
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

/**
 * A wrapper that prevents compiler errors when you invoke [[Awaitable.await]] in a `for` block.
 *
 * For example:
 *
 * {{{
 * for (element <- seq) {
 *   // Compiler error: `await` must be enclosed in a `Future` block
 *   doSomething(element).await
 * }
 * }}}
 *
 * To suppress the error, wrap the original `seq` in a [[AwaitableSeq.futureSeq]]:
 *
 * {{{
 * for (element <- AwaitableSeq.futureSeq(seq)) {
 *   // No compiler error now
 *   doSomething(element).await
 * }
 * }}}
 *
 */
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

  final def awaitableFlatMap[B](f: A => Future[AwaitableSeq[B, TailRecResult]]) = {
    Awaitable[AwaitableSeq[B, TailRecResult], TailRecResult] {
      new AwaitableSeq[B, TailRecResult](
        foldLeft(Generator[B].Future {}) { (left, current) =>
          f(current).map { that =>
            Generator[B].Future {
              left.await
              Generator[B].apply(that.underlying: _*).await
            }
          }
        }.await)
    }
  }

  final def awaitableMap[B](f: A => Future[B]) = {
    Awaitable[AwaitableSeq[B, TailRecResult], TailRecResult] {
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
  }

  final def awaitableForeach[U](f: A => Future[U]) = Awaitable[Unit, TailRecResult] {
    val _ = foldLeft[Any](()) { (left, current) =>
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

  final def flatMap[B](f: A => AwaitableSeq[B, TailRecResult]): AwaitableSeq[B, TailRecResult] = macro AwaitableSeq.flatMapMacro
  final def map[B](f: A => B): AwaitableSeq[B, TailRecResult] = macro AwaitableSeq.mapMacro
  final def foreach[U](f: A => U): Unit = macro AwaitableSeq.foreachMacro
}
