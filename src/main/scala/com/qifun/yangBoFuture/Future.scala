package com.qifun.yangBoFuture

import scala.runtime.AbstractPartialFunction
import scala.reflect.macros.Context
import scala.util.control.Exception.Catcher
import scala.reflect.internal.annotations.compileTimeOnly
import scala.annotation.tailrec
import scala.util.control.TailCalls._
import java.util.concurrent.TimeUnit

trait Future[+A] { outer =>

  @compileTimeOnly("`await` must be enclosed in a `Future` block")
  final def await: A = ???

  def foreach(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit]

  final def map[B](f: A => B) = new Future[B] {
    def foreach(k: B => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      def apply(a: A): TailRec[Unit] = {
        val b = try {
          f(a)
        } catch {
          case e if catcher.isDefinedAt(e) => {
            return tailcall(catcher(e))
          }
        }
        tailcall(k(b))
      }
      outer.foreach(apply)
    }
  }

  final def withFilter(p: A => Boolean) = new Future[A] {
    def foreach(k: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      def apply(a: A): TailRec[Unit] = {
        val b = try {
          p(a)
        } catch {
          case e if catcher.isDefinedAt(e) => {
            return tailcall(catcher(e))
          }
        }
        if (b) {
          tailcall(k(a))
        } else {
          done(())
        }
      }
      outer.foreach(apply)
    }
  }

  final def flatMap[B](mapping: A => Future[B]) = new Future[B] {
    override final def foreach(body: B => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      def apply(a: A): TailRec[Unit] = {
        val futureB = try {
          mapping(a)
        } catch {
          case e if catcher.isDefinedAt(e) => {
            return tailcall(catcher(e))
          }
        }
        for (b <- futureB) {
          tailcall(body(b))
        }
      }
      outer.foreach(apply)
    }
  }

}

object Future {

  implicit final class FromConcurrentFuture[A](underlying: scala.concurrent.Future[A])(implicit executor: scala.concurrent.ExecutionContext) extends Future[A] {
    import scala.util._
    override def foreach(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      underlying.onComplete {
        case Success(successValue) => body(successValue).result
        case Failure(throwable) => {
          if (catcher.isDefinedAt(throwable)) {
            catcher(throwable).result
          } else {
            executor.prepare.reportFailure(throwable)
          }
        }
      }
      done(())
    }

  }

  implicit final class ToConcurrentFuture[A](underlying: Future[A]) extends scala.concurrent.Future[A] {

    import scala.concurrent._
    import scala.concurrent.duration.Duration
    import scala.util._

    final var value: Option[Try[A]] = None

    override final def isCompleted = value.isDefined

    override final def onComplete[U](func: (Try[A]) => U)(implicit executor: ExecutionContext) {
      def post(result: Try[A]) {
        synchronized {
          value = Some(result)
          executor.prepare.execute(new Runnable {
            override final def run() {
              func(result)
            }
          })
          notifyAll()
        }
      }
      implicit def catcher: Catcher[TailRec[Unit]] = {
        case throwable: Throwable => {
          post(Failure(throwable))
          done(())
        }
      }
      for (successValue <- underlying) {
        post(Success(successValue))
        done(())
      }
    }

    override final def result(atMost: Duration)(implicit permit: CanAwait): A = {
      ready(atMost)
      value.get match {
        case Success(successValue) => successValue
        case Failure(throwable) => throw throwable
      }
    }

    override final def ready(atMost: Duration)(implicit permit: CanAwait): this.type = {
      if (atMost eq Duration.Undefined) {
        throw new IllegalArgumentException
      }
      synchronized {
        val timeoutAt = atMost.toNanos + System.nanoTime
        val milliseconds = (atMost / 1000000).toNanos
        while (!isCompleted) {
          val restDuration = timeoutAt - System.nanoTime - atMost.toNanos
          if (restDuration < 0) {
            throw new TimeoutException
          }
          wait(restDuration / 1000000, (restDuration % 1000000).toInt)
        }
        this
      }
    }

  }

  implicit final class FromResponder[A](underlying: Responder[A]) extends Future[A] {
    override def foreach(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      try {
        underlying.respond { a =>
          body(a).result
        }
      } catch {
        case e if catcher.isDefinedAt(e) =>
          catcher(e).result
      }
      done(())
    }
  }

  implicit final class ToResponder[A](underlying: Future[A])(implicit catcher: Catcher[Unit]) extends Responder[A] {

    override final def respond(handler: A => Unit) {
      (underlying.foreach { a =>
        done(handler(a))
      } {
        case e if catcher.isDefinedAt(e) => {
          done(catcher(e))
        }
      }).result
    }

  }

  final class TryCatchFinally[A](tryFuture: Future[A], getCatcherFuture: Catcher[Future[A]], finallyBlock: => Unit) extends Future[A] {
    @inline
    override final def foreach(rest: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]) = {
      (for (a <- tryFuture) {
        // 成功执行 try
        try {
          finallyBlock
          tailcall(rest(a))
        } catch {
          case e if catcher.isDefinedAt(e) => {
            // 执行finally时出错
            tailcall(catcher(e))
          }
        }
      }) {
        case e if getCatcherFuture.isDefinedAt(e) => {
          // 执行 try 失败，用getCatcherFuture进行恢复
          val catcherFuture = getCatcherFuture(e)
          (for (a <- catcherFuture) {
            // 成功恢复
            try {
              finallyBlock
              tailcall(rest(a))
            } catch {
              case e if catcher.isDefinedAt(e) => {
                // 执行finally时出错
                tailcall(catcher(e))
              }
            }
          }) {
            case e if catcher.isDefinedAt(e) => {
              // 执行 try 失败，getCatcherFuture恢复失败，触发外层异常
              try {
                finallyBlock
                tailcall(catcher(e))
              } catch {
                case e if catcher.isDefinedAt(e) => {
                  // 执行finally时出错
                  tailcall(catcher(e))
                }
              }
            }
            case e: Throwable => {
              finallyBlock
              throw e
            }
          }
        }
        case e if catcher.isDefinedAt(e) => {
          // 执行 try 失败，getCatcherFuture不支持恢复，触发外层异常
          try {
            finallyBlock
            tailcall(catcher(e))
          } catch {
            case e if catcher.isDefinedAt(e) => {
              // 执行finally时出错
              tailcall(catcher(e))
            }
          }
        }
        case e: Throwable => {
          // 执行 try 失败，getCatcherFuture不支持恢复，外层异常
          finallyBlock
          throw e
        }
      }
    }
  }

  def applyMacro(c: Context)(futureBody: c.Expr[_]): c.Expr[Future[Nothing]] = {

    import c.universe.Flag._
    import c.universe._
    import c.mirror._
    import compat._

    val i = Ident(newTermName("xxx"))
    c.warning(c.enclosingPosition, raw"""symbol==NoSymbol ${i.symbol == NoSymbol}
    tpe==null ${i.tpe == null}
    pos${i.pos}""")

    val abstractPartialFunction = c.typeTag[AbstractPartialFunction[_, _]].tpe
    val futureType = c.typeTag[Future[_]].tpe
    val function1Type = c.typeTag[_ => _].tpe
    val function1Symbol = function1Type.typeSymbol
    val uncheckedSymbol = c.typeTag[scala.unchecked].tpe.typeSymbol
    val localSymbols = scala.collection.mutable.Set.empty[Symbol]
    for (subTree <- futureBody.tree) {
      subTree match {
        case _: DefTree => localSymbols += subTree.symbol
        case _ =>
      }
    }
    def multipleTransform(trees: List[Tree], relocatedSymbols: Set[Symbol], catcher: Tree, rest: (List[Tree], Set[Symbol]) => Tree)(implicit forceAwait: Set[Name]): Tree = {
      trees match {
        case Nil => rest(Nil, relocatedSymbols)
        case head :: tail => {
          transform(head, relocatedSymbols, catcher, { (transformedHead, relocatedSymbols) =>
            multipleTransform(tail, relocatedSymbols, catcher, { (transformedTail, relocatedSymbols) =>
              rest(transformedHead :: transformedTail, relocatedSymbols)
            })
          })
        }
      }
    }

    def newCatcher(cases: List[CaseDef], typeTree: Tree): Tree = {
      val catcherClassName = newTypeName(c.fresh("YangBoCatcher"))
      val isDefinedCases = ((for (CaseDef(pat, guard, _) <- cases.view) yield {
        CaseDef(c.resetLocalAttrs(pat), c.resetLocalAttrs(guard), Literal(Constant(true)))
      }) :+ CaseDef(Ident(nme.WILDCARD), EmptyTree, Literal(Constant(false)))).toList
      val defaultName = newTermName(c.fresh("default"))
      val throwableName = newTermName(c.fresh("throwable"))
      val applyOrElseCases = cases :+ CaseDef(Ident(nme.WILDCARD), EmptyTree, Apply(Ident(defaultName), List(Ident(throwableName))))
      Block(
        List(
          ClassDef(
            Modifiers(FINAL),
            catcherClassName,
            List(),
            Template(
              List(
                AppliedTypeTree(
                  Ident(abstractPartialFunction.typeSymbol),
                  List(
                    TypeTree(c.typeTag[Throwable].tpe),
                    typeTree))),
              emptyValDef,
              List(
                DefDef(
                  Modifiers(),
                  nme.CONSTRUCTOR,
                  List(),
                  List(List()),
                  TypeTree(),
                  Block(
                    List(
                      Apply(Select(Super(This(tpnme.EMPTY), tpnme.EMPTY), nme.CONSTRUCTOR), List())),
                    Literal(Constant(())))),

                DefDef(
                  Modifiers(OVERRIDE | FINAL),
                  newTermName("applyOrElse"),
                  List(
                    TypeDef(Modifiers(PARAM), newTypeName("A1"), List(), TypeBoundsTree(TypeTree(c.typeTag[Nothing].tpe), TypeTree(c.typeTag[Throwable].tpe))),
                    TypeDef(Modifiers(PARAM), newTypeName("B1"), List(), TypeBoundsTree(typeTree, TypeTree(c.typeTag[Any].tpe)))),
                  List(List(
                    ValDef(
                      Modifiers(PARAM),
                      throwableName,
                      Ident(newTypeName("A1")),
                      EmptyTree),
                    ValDef(
                      Modifiers(PARAM),
                      defaultName,
                      AppliedTypeTree(Ident(function1Symbol), List(Ident(newTypeName("A1")), Ident(newTypeName("B1")))),
                      EmptyTree))),
                  Ident(newTypeName("B1")),
                  Match(
                    Annotated(
                      Apply(Select(New(Ident(uncheckedSymbol)), nme.CONSTRUCTOR), List()),
                      Ident(throwableName)),
                    applyOrElseCases)),

                DefDef(
                  Modifiers(OVERRIDE | FINAL),
                  newTermName("isDefinedAt"),
                  List(),
                  List(List(ValDef(Modifiers(PARAM), throwableName, TypeTree(c.typeTag[Throwable].tpe), EmptyTree))),
                  TypeTree(c.typeTag[Boolean].tpe),
                  Match(
                    Annotated(
                      Apply(Select(New(Ident(uncheckedSymbol)), nme.CONSTRUCTOR), List()),
                      Ident(throwableName)),
                    isDefinedCases)))))),
        Apply(Select(New(Ident(catcherClassName)), nme.CONSTRUCTOR), List()))

    }

    def transformAwait(future: Tree, relocatedSymbols: Set[Symbol], awaitTypeTree: Tree, catcher: Tree, rest: (Tree, Set[Symbol]) => Tree)(implicit forceAwait: Set[Name]): Tree = {
      val futureExpr = c.Expr(future)
      val awaitValue = newTermName(c.fresh("awaitValue"))
      val catcherExpr = c.Expr[Catcher[TailRec[Unit]]](catcher)
      val restExpr = c.Expr(rest(Ident(awaitValue).setPos(future.pos), relocatedSymbols))

      // 这里调用 .setPos(tree.pos) 会崩溃
      Apply(
        Apply(
          Select(
            reify {
              // Erase
              futureExpr.splice.asInstanceOf[Future[Nothing]]
            }.tree,
            newTermName("foreach")).setPos(future.pos),
          List(Function(
            List(ValDef(Modifiers(PARAM), awaitValue, awaitTypeTree, EmptyTree)),
            reify {
              try {
                restExpr.splice
              } catch {
                case e if catcherExpr.splice.isDefinedAt(e) => {
                  _root_.scala.util.control.TailCalls.tailcall(catcherExpr.splice(e))
                }
              }
            }.tree))),
        List(catcher))
    }

    def transform(tree: Tree, relocatedSymbols: Set[Symbol], catcher: Tree, rest: (Tree, Set[Symbol]) => Tree)(implicit forceAwait: Set[Name]): Tree = {
      tree match {
        case Try(block, catches, finalizer) => {
          val futureName = newTermName(c.fresh("tryCatchFinallyFuture"))
          Block(
            List(
              ValDef(Modifiers(), futureName, TypeTree(),
                Apply(
                  Select(
                    New(
                      AppliedTypeTree(
                        Select(reify(_root_.com.qifun.yangBoFuture.Future).tree,
                          newTypeName("TryCatchFinally")),
                        List(TypeTree(block.tpe.widen)))),
                    nme.CONSTRUCTOR),
                  List(
                    newFuture(block).tree,
                    newCatcher(
                      for (cd @ CaseDef(pat, guard, body) <- catches) yield {
                        CaseDef(c.resetLocalAttrs(pat), c.resetLocalAttrs(guard), newFuture(body).tree).setPos(cd.pos).asInstanceOf[CaseDef]
                      },
                      AppliedTypeTree(
                        Ident(futureType.typeSymbol),
                        List(TypeTree(block.tpe.widen)))),
                    if (finalizer.isEmpty) {
                      Literal(Constant(()))
                    } else {
                      finalizer
                    })))),
            transformAwait(Ident(futureName), relocatedSymbols, TypeTree(tree.tpe), catcher, rest))

        }
        case _: ImplDef | _: DefDef | _: New | _: Ident | _: Literal | _: Super | _: This | _: TypTree | _: New | _: TypeDef | _: Function => {
          rest((new Transformer {
            override def transform(tree: Tree) = {
              tree match {
                case ident: Ident if localSymbols(ident.symbol) => {
                  tree.setSymbol(NoSymbol)
                  tree.setType(null)
                  tree
                  //
                  //                  Ident(ident.name).setPos(ident.pos)
                }
                case defTree: DefTree => {
                  //
                  ////                  tree.setType(null)
//                                    tree.setSymbol(NoSymbol)
//                                    tree
                  c.resetLocalAttrs(defTree)
                
                }
                case _ => super.transform(tree)
              }
            }
          }).transform(tree), relocatedSymbols)
        }
        case Select(future, await) if await.decoded == "await" && future.tpe <:< futureType => {
          transform(future, relocatedSymbols, catcher, { (transformedFuture, relocatedSymbols) =>
            transformAwait(transformedFuture, relocatedSymbols, TypeTree(tree.tpe), catcher, rest)
          })
        }
        case Select(instance, field) => {
          transform(instance, relocatedSymbols, catcher, { (transformedInstance, relocatedSymbols) =>
            rest(Select(transformedInstance, field).setPos(tree.pos), relocatedSymbols)
          })
        }
        case TypeApply(method, parameters) => {
          transform(method, relocatedSymbols, catcher, { (transformedMethod, relocatedSymbols) =>
            multipleTransform(parameters, relocatedSymbols, catcher, { (transformedParameters, relocatedSymbols) =>
              rest(TypeApply(transformedMethod, transformedParameters).setPos(tree.pos), relocatedSymbols)
            })
          })
        }
        case Apply(Ident(name), parameters) if forceAwait(name) => {
          multipleTransform(parameters, relocatedSymbols, catcher, { (transformedParameters, relocatedSymbols) =>
            transformAwait(Apply(Ident(name), transformedParameters).setPos(tree.pos), relocatedSymbols, TypeTree(tree.tpe), catcher, rest)
          })
        }
        case Apply(method, parameters) => {
          transform(method, relocatedSymbols, catcher, { (transformedMethod, relocatedSymbols) =>
            multipleTransform(parameters, relocatedSymbols, catcher, { (transformedParameters, relocatedSymbols) =>
              rest(Apply(transformedMethod, transformedParameters).setPos(tree.pos), relocatedSymbols)
            })
          })
        }
        case Block(stats, expr) => {
          def transformBlock(stats: List[Tree], relocatedSymbols: Set[Symbol]): Tree = {
            stats match {
              case Nil => {
                transform(expr, relocatedSymbols, catcher, { (transformedExpr, relocatedSymbols) =>
                  rest(transformedExpr, relocatedSymbols)
                })
              }
              case head :: tail => {
                transform(head, relocatedSymbols, catcher, { (transformedHead, relocatedSymbols) =>
                  transformedHead match {
                    case _: Ident | _: Literal => {
                      transformBlock(tail, relocatedSymbols)
                    }
                    case _ => {
                      Block(
                        List(transformedHead),
                        transformBlock(tail, relocatedSymbols))
                    }
                  }
                })
              }
            }
          }
          transformBlock(stats, relocatedSymbols)
        }
        case ValDef(mods, name, tpt, rhs) => {
          transform(rhs, relocatedSymbols, catcher, { (transformedRhs, relocatedSymbols) =>
            rest(ValDef(mods, name, c.resetLocalAttrs(tpt), transformedRhs), relocatedSymbols + tree.symbol)
          })
        }
        case Assign(left, right) => {
          transform(left, relocatedSymbols, catcher, { (transformedLeft, relocatedSymbols) =>
            transform(right, relocatedSymbols, catcher, { (transformedRight, relocatedSymbols) =>
              rest(Assign(transformedLeft, transformedRight).setPos(tree.pos), relocatedSymbols)
            })
          })
        }
        case Match(selector, cases) => {
          transform(selector, relocatedSymbols, catcher, { (transformedSelector, relocatedSymbols) =>
            transformAwait(
              Match(transformedSelector,
                for (CaseDef(pat, guard, body) <- cases) yield {
                  CaseDef(
                    c.resetLocalAttrs(pat),
                    c.resetLocalAttrs(guard),
                    Typed(newFuture(body).tree, AppliedTypeTree(Ident(futureType.typeSymbol), List(TypeTree(tree.tpe)))))
                }),
              relocatedSymbols,
              TypeTree(tree.tpe),
              catcher,
              rest)
          })
        }
        case If(cond, thenp, elsep) => {
          transform(cond, relocatedSymbols, catcher, { (transformedCond, relocatedSymbols) =>
            transformAwait(
              If(
                transformedCond,
                Typed(newFuture(thenp).tree, AppliedTypeTree(Ident(futureType.typeSymbol), List(TypeTree(tree.tpe)))),
                Typed(newFuture(elsep).tree, AppliedTypeTree(Ident(futureType.typeSymbol), List(TypeTree(tree.tpe))))),
              relocatedSymbols,
              TypeTree(tree.tpe),
              catcher,
              rest)
          })
        }
        case Throw(throwable) => {
          transform(throwable, relocatedSymbols, catcher, { (transformedThrowable, relocatedSymbols) =>
            rest(Throw(transformedThrowable).setPos(tree.pos), relocatedSymbols)
          })
        }
        case Typed(expr, tpt) => {
          transform(expr, relocatedSymbols, catcher, { (transformedExpr, relocatedSymbols) =>
            // 不确定会不会崩溃
            rest(Typed(transformedExpr, tpt).setPos(tree.pos), relocatedSymbols)
          })
        }
        case Annotated(annot, arg) => {
          transform(arg, relocatedSymbols, catcher, { (transformedArg, relocatedSymbols) =>
            // 不确定会不会崩溃
            rest(Annotated(annot, transformedArg).setPos(tree.pos), relocatedSymbols)
          })
        }
        case LabelDef(name, params, rhs) => {
          val breakName = newTermName(c.fresh("yangBoBreak"))
          Block(
            List(
              DefDef(Modifiers(),
                name,
                List(),
                List(
                  for (p <- params) yield {
                    ValDef(Modifiers(PARAM), p.name.toTermName, TypeTree(p.tpe), EmptyTree)
                  }),
                AppliedTypeTree(Ident(futureType.typeSymbol), List(TypeTree(tree.tpe))),
                newFuture(rhs)(forceAwait + name).tree)),
            transformAwait(
              Apply(
                Ident(name),
                params),
              relocatedSymbols,
              TypeTree(tree.tpe),
              catcher,
              rest))
        }
        case _: PackageDef | _: Import | _: ImportSelector | _: Template | _: CaseDef | _: Alternative | _: Star | _: Bind | _: UnApply | _: AssignOrNamedArg | _: Return | _: ReferenceToBoxed => {
          c.error(tree.pos, "Unexpected expression in a `Future` block")
          rest(tree, relocatedSymbols)
        }
      }
    }

    def newFutureAsType(tree: Tree, parameterTypeTree: Tree)(implicit forceAwait: Set[Name]): c.Expr[Future[Nothing]] = {

      val futureTypeTree = AppliedTypeTree(Ident(futureType.typeSymbol), List(parameterTypeTree))

      val futureName = newTypeName(c.fresh("YangBoFuture"))
      val returnName = newTermName(c.fresh("yangBoReturn"))
      val catcherName = newTermName(c.fresh("yangBoCatcher"))
      c.Expr(
        Block(
          List(
            ClassDef(
              Modifiers(FINAL),
              futureName,
              List(),
              Template(
                List(futureTypeTree),
                emptyValDef,
                List(
                  DefDef(
                    Modifiers(),
                    nme.CONSTRUCTOR,
                    List(),
                    List(List()),
                    TypeTree(),
                    Block(List(Apply(Select(Super(This(tpnme.EMPTY), tpnme.EMPTY), nme.CONSTRUCTOR), List())), Literal(Constant(())))),
                  DefDef(Modifiers(OVERRIDE | FINAL),
                    newTermName("foreach"),
                    List(),
                    List(
                      List(ValDef(
                        Modifiers(PARAM),
                        returnName,
                        AppliedTypeTree(Ident(function1Symbol), List(parameterTypeTree, TypeTree(typeTag[TailRec[Unit]].tpe))), EmptyTree)),
                      List(ValDef(
                        Modifiers(IMPLICIT | PARAM),
                        catcherName,
                        TypeTree(typeTag[Catcher[TailRec[Unit]]].tpe),
                        EmptyTree))),
                    TypeTree(typeTag[TailRec[Unit]].tpe),
                    {
                      val catcherExpr = c.Expr[Catcher[TailRec[Unit]]](Ident(catcherName))
                      val tryBodyExpr = c.Expr(transform(
                        tree,
                        Set.empty,
                        Ident(catcherName),
                        { (x, relocatedSymbols) =>
                          val resultExpr = c.Expr(x)
                          val returnExpr = c.Expr[Any => Nothing](Ident(returnName))
                          reify {
                            val result = resultExpr.splice
                            _root_.scala.util.control.TailCalls.tailcall(returnExpr.splice(result))
                          }.tree
                        }))
                      reify {
                        try {
                          tryBodyExpr.splice
                        } catch {
                          case e if catcherExpr.splice.isDefinedAt(e) => {
                            _root_.scala.util.control.TailCalls.tailcall(catcherExpr.splice(e))
                          }
                        }
                      }.tree
                    }))))),
          Typed(
            Apply(Select(New(Ident(futureName)), nme.CONSTRUCTOR), List()),
            futureTypeTree)))
    }
    def newFuture(tree: Tree)(implicit forceAwait: Set[Name]): c.Expr[Future[Nothing]] = {
      newFutureAsType(tree, TypeTree(tree.tpe.widen))
    }

    val Apply(TypeApply(_, List(t)), _) = c.macroApplication
    val result = newFutureAsType(futureBody.tree, t)(Set.empty)
    c.warning(c.enclosingPosition, show(result))
    result
    //    c.Expr(TypeApply(Select(result.tree, newTermName("asInstanceOf")), List(TypeTree(c.macroApplication.tpe))))
  }

  import scala.language.experimental.macros
  def apply[A](futureBody: => A): Future[A] = macro applyMacro

}