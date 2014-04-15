package com.qifun.yangBoFuture

import scala.runtime.AbstractPartialFunction
import scala.reflect.macros.Context
import scala.util.control.Exception.Catcher
import scala.reflect.internal.annotations.compileTimeOnly
import scala.annotation.elidable
import scala.annotation.tailrec
import scala.util.control.TailCalls._
import java.util.concurrent.TimeUnit
import scala.reflect.internal.annotations.uncheckedBounds

trait Future[+A] { outer =>

  @compileTimeOnly("`await` must be enclosed in a `Future` block")
  final def await: A = ???

  def onComplete(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit]

  final def foreach[U](f: A => U)(implicit catcher: Catcher[Unit]) {
    onComplete { a =>
      f(a)
      done(())
    } {
      case e if catcher.isDefinedAt(e) =>
        catcher(e)
        done(())
    }.result
  }

  final def map[B](f: A => B) = new Future[B] {
    def onComplete(k: B => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
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
      outer.onComplete(apply)
    }
  }

  final def withFilter(p: A => Boolean) = new Future[A] {
    def onComplete(k: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
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
      outer.onComplete(apply)
    }
  }

  final def flatMap[B](mapping: A => Future[B]) = new Future[B] {
    override final def onComplete(body: B => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
      def apply(a: A): TailRec[Unit] = {
        val futureB = try {
          mapping(a)
        } catch {
          case e if catcher.isDefinedAt(e) => {
            return tailcall(catcher(e))
          }
        }
        futureB.onComplete { b =>
          tailcall(body(b))
        }
      }
      outer.onComplete(apply)
    }
  }

}

object Future {

  implicit final class FromConcurrentFuture[A](underlying: scala.concurrent.Future[A])(implicit executor: scala.concurrent.ExecutionContext) extends Future[A] {
    import scala.util._
    override def onComplete(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
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
      implicit def catcher: Catcher[Unit] = {
        case throwable: Throwable => {
          post(Failure(throwable))
        }
      }
      for (successValue <- underlying) {
        post(Success(successValue))
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
    override def onComplete(body: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]): TailRec[Unit] = {
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
      (underlying.onComplete { a =>
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
    override final def onComplete(rest: A => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]) = {
      tryFuture.onComplete { a =>
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
      } {
        case e if getCatcherFuture.isDefinedAt(e) => {
          // 执行 try 失败，用getCatcherFuture进行恢复
          val catcherFuture = getCatcherFuture(e)
          catcherFuture.onComplete { a =>
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
          } {
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

  def applyMacro(c: Context)(futureBody: c.Expr[Any]): c.Expr[Future[Nothing]] = {

    import c.universe.Flag._
    import c.universe._
    import c.mirror._
    import compat._

    def uncheckedBounds(typeTree: Tree) = {
      Annotated(Apply(Select(New(TypeTree(typeOf[scala.reflect.internal.annotations.uncheckedBounds])), nme.CONSTRUCTOR), List()), typeTree)
    }

    val abstractPartialFunction = typeOf[AbstractPartialFunction[_, _]]
    val futureType = typeOf[Future[_]]
    val function1Type = typeOf[_ => _]
    val function1Symbol = function1Type.typeSymbol
    val uncheckedSymbol = typeOf[scala.unchecked].typeSymbol
    def multipleTransform(trees: List[Tree], catcher: Tree, rest: (List[Tree]) => Tree)(implicit forceAwait: Set[Name]): Tree = {
      trees match {
        case Nil => rest(Nil)
        case head :: tail => {
          transform(head, catcher, { (transformedHead) =>
            multipleTransform(tail, catcher, { (transformedTail) =>
              rest(transformedHead :: transformedTail)
            })
          })
        }
      }
    }

    def newCatcher(cases: List[CaseDef], typeTree: Tree): Tree = {
      val catcherClassName = newTypeName(c.fresh("YangBoCatcher"))
      val isDefinedCases = ((for (originCaseDef @ CaseDef(pat, guard, _) <- cases.view) yield {
        treeCopy.CaseDef(originCaseDef, pat, guard, Literal(Constant(true)))
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
                    TypeTree(typeOf[Throwable]),
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
                    TypeDef(Modifiers(PARAM), newTypeName("A1"), List(), TypeBoundsTree(TypeTree(typeOf[Nothing]), TypeTree(typeOf[Throwable]))),
                    TypeDef(Modifiers(PARAM), newTypeName("B1"), List(), TypeBoundsTree(typeTree, TypeTree(typeOf[Any])))),
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
                  List(List(ValDef(Modifiers(PARAM), throwableName, TypeTree(typeOf[Throwable]), EmptyTree))),
                  TypeTree(typeOf[Boolean]),
                  Match(
                    Annotated(
                      Apply(Select(New(Ident(uncheckedSymbol)), nme.CONSTRUCTOR), List()),
                      Ident(throwableName)),
                    isDefinedCases)))))),
        Apply(Select(New(Ident(catcherClassName)), nme.CONSTRUCTOR), List()))

    }

    def transformAwait(future: Tree, awaitTypeTree: TypTree, catcher: Tree, rest: (Tree) => Tree)(implicit forceAwait: Set[Name]): Tree = {
      c.warning(future.pos, show(awaitTypeTree))
      val futureExpr = c.Expr(future)
      val awaitValue = newTermName(c.fresh("awaitValue"))
      val catcherExpr = c.Expr[Catcher[TailRec[Unit]]](catcher)
//      val restExpr = c.Expr(rest(Ident(awaitValue)))
            val restExpr = c.Expr(rest(TypeApply(Select(Ident(awaitValue), newTermName("asInstanceOf")), List(uncheckedBounds(awaitTypeTree)))))

      Apply(
        Apply(
          Select(
            //            future,
            //            reify {
            //              futureExpr.splice.asInstanceOf[Future[Nothing]]
            //            }.tree,

            treeCopy.TypeApply(
              future,
              Select(future, newTermName("asInstanceOf")),
              List(uncheckedBounds(AppliedTypeTree(Ident(futureType.typeSymbol), List(uncheckedBounds(awaitTypeTree)))))),
            newTermName("onComplete")),
          List(Function(
            List(ValDef(Modifiers(PARAM), awaitValue, uncheckedBounds(awaitTypeTree), EmptyTree)),
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

    def transform(tree: Tree, catcher: Tree, rest: (Tree) => Tree)(implicit forceAwait: Set[Name]): Tree = {
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
                        List(TypeTree(block.tpe.widen).defineType(block.tpe.widen)))),
                    nme.CONSTRUCTOR),
                  List(
                    newFuture(block).tree,
                    newCatcher(
                      for (cd @ CaseDef(pat, guard, body) <- catches) yield {
                        treeCopy.CaseDef(cd, pat, guard, newFuture(body).tree)
                      },
                      AppliedTypeTree(
                        Ident(futureType.typeSymbol),
                        List(TypeTree(block.tpe.widen).defineType(block.tpe.widen)))),
                    if (finalizer.isEmpty) {
                      Literal(Constant(()))
                    } else {
                      finalizer
                    })))),
            transformAwait(Ident(futureName), TypeTree(tree.tpe).defineType(tree.tpe).asInstanceOf[TypTree], catcher, rest))

        }
        case EmptyTree | _: ImplDef | _: DefDef | _: New | _: Ident | _: Literal | _: Super | _: This | _: TypTree | _: New | _: TypeDef | _: Function => {
          rest(tree)
        }
        case Select(future, await) if await.decoded == "await" && future.tpe <:< futureType => {
          transform(future, catcher, { (transformedFuture) =>
            transformAwait(transformedFuture, TypeTree(tree.tpe).defineType(tree.tpe).asInstanceOf[TypTree], catcher, rest)
          })
        }
        case Select(instance, field) => {
          transform(instance, catcher, { (transformedInstance) =>
            rest(treeCopy.Select(tree, transformedInstance, field).setPos(tree.pos))
          })
        }
        case TypeApply(method, parameters) => {
          transform(method, catcher, { (transformedMethod) =>
            multipleTransform(parameters, catcher, { (transformedParameters) =>
              rest(treeCopy.TypeApply(tree, transformedMethod, transformedParameters).setPos(tree.pos))
            })
          })
        }
        case Apply(Ident(name), parameters) if forceAwait(name) => {
          multipleTransform(parameters, catcher, { (transformedParameters) =>
            transformAwait(treeCopy.Apply(tree, Ident(name), transformedParameters).setPos(tree.pos), TypeTree(tree.tpe).defineType(tree.tpe).asInstanceOf[TypTree], catcher, rest)
          })
        }
        case Apply(method, parameters) => {
          transform(method, catcher, { (transformedMethod) =>
            multipleTransform(parameters, catcher, { (transformedParameters) =>
              val uncheckedParameters = for (p <- transformedParameters) yield {
                p.tpe match {
                  case null => p
                  case TypeRef(NoPrefix, _, _) => {
                    TypeApply(
                      Select(p, newTermName("asInstanceOf")),
                      List(uncheckedBounds(TypeTree(p.tpe).defineType(p.tpe).asInstanceOf[TypTree])))
                  }
                  case _ => p
                }
              }
              rest(treeCopy.Apply(
                tree,
                transformedMethod,
                uncheckedParameters).setPos(tree.pos))
            })
          })
        }
        case Block(stats, expr) => {
          def addHead(head: Tree, tuple: (Tree, Boolean)): Tree = {
            val (tail, mergeable) = tuple
            tail match {
              case Block(stats, expr) if mergeable => {
                treeCopy.Block(tree, head :: stats, expr)
              }
              case _ => {
                treeCopy.Block(tree, List(head), tail)
              }
            }
          }
          def transformBlock(stats: List[Tree])(implicit forceAwait: Set[Name]): (Tree, Boolean) = {
            stats match {
              case Nil => {
                (transform(expr, catcher, { (transformedExpr) =>
                  Block(Nil, rest(transformedExpr))
                }), false)
              }
              case head :: tail => {
                (transform(head, catcher,
                  { (transformedHead) =>
                    transformedHead match {
                      case _: Ident | _: Literal => {
                        val (block, _) = transformBlock(tail)
                        block
                      }
                      case _ => {
                        addHead(transformedHead, transformBlock(tail))
                      }
                    }
                  }), true)
              }
            }
          }
          Block(Nil, transformBlock(stats)._1)
        }
        case ValDef(mods, name, tpt, rhs) => {
          transform(rhs, catcher, { (transformedRhs) =>
            rest(treeCopy.ValDef(tree, mods, name, tpt, transformedRhs))
          })
        }
        case Assign(left, right) => {
          transform(left, catcher, { (transformedLeft) =>
            transform(right, catcher, { (transformedRight) =>
              rest(Assign(transformedLeft, transformedRight).setPos(tree.pos))
            })
          })
        }
        case Match(selector, cases) => {
          transform(selector, catcher, { (transformedSelector) =>
            transformAwait(
              treeCopy.Match(
                tree,
                transformedSelector,
                for (originCaseDef @ CaseDef(pat, guard, body) <- cases) yield {
                  treeCopy.CaseDef(originCaseDef,
                    pat,
                    guard,
                    {
                      val caseFutureExpr = newFutureAsType(body, TypeTree(tree.tpe).defineType(tree.tpe))
                      reify {
                        caseFutureExpr.splice.asInstanceOf[Future[Nothing]]
                      }.tree
                    })
                }),
              TypeTree().defineType(tree.tpe).asInstanceOf[TypTree],
              catcher,
              rest)
          })
        }
        case If(cond, thenp, elsep) => {
          transform(cond, catcher, { (transformedCond) =>
            transformAwait(
              If(
                transformedCond,
                TypeApply(Select(newFuture(thenp).tree, newTermName("asInstanceOf")), List(uncheckedBounds(AppliedTypeTree(Ident(futureType.typeSymbol), List(TypeTree(tree.tpe).defineType(tree.tpe)))))),
                TypeApply(Select(newFuture(elsep).tree, newTermName("asInstanceOf")), List(uncheckedBounds(AppliedTypeTree(Ident(futureType.typeSymbol), List(TypeTree(tree.tpe).defineType(tree.tpe))))))),
              TypeTree(tree.tpe).defineType(tree.tpe).asInstanceOf[TypTree],
              catcher,
              rest)
          })
        }
        case Throw(throwable) => {
          transform(throwable, catcher, { (transformedThrowable) =>
            rest(Throw(transformedThrowable).setPos(tree.pos))
          })
        }
        case Typed(expr, tpt) => {
          transform(expr, catcher, { (transformedExpr) =>
            // 不确定会不会崩溃
            rest(Typed(transformedExpr, tpt).setPos(tree.pos))
          })
        }
        case Annotated(annot, arg) => {
          transform(arg, catcher, { (transformedArg) =>
            // 不确定会不会崩溃
            rest(Annotated(annot, transformedArg).setPos(tree.pos))
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
                    ValDef(Modifiers(PARAM), p.name.toTermName, TypeTree(p.tpe).defineType(p.tpe), EmptyTree)
                  }),
                AppliedTypeTree(Ident(futureType.typeSymbol), List(TypeTree(tree.tpe).defineType(tree.tpe))),
                newFuture(rhs)(forceAwait + name).tree)),
            transformAwait(
              Apply(
                Ident(name),
                params),

              TypeTree(tree.tpe).defineType(tree.tpe).asInstanceOf[TypTree],
              catcher,
              rest))
        }
        case _: PackageDef | _: Import | _: ImportSelector | _: Template | _: CaseDef | _: Alternative | _: Star | _: Bind | _: UnApply | _: AssignOrNamedArg | _: Return | _: ReferenceToBoxed => {
          c.error(tree.pos, "Unexpected expression in a `Future` block")
          rest(tree)
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
                    newTermName("onComplete"),
                    List(),
                    List(
                      List(ValDef(
                        Modifiers(PARAM),
                        returnName,
                        AppliedTypeTree(Ident(function1Symbol), List(parameterTypeTree, TypeTree(typeOf[TailRec[Unit]]))), EmptyTree)),
                      List(ValDef(
                        Modifiers(IMPLICIT | PARAM),
                        catcherName,
                        TypeTree(typeTag[Catcher[TailRec[Unit]]].tpe),
                        EmptyTree))),
                    TypeTree(typeOf[TailRec[Unit]]),
                    {
                      val catcherExpr = c.Expr[Catcher[TailRec[Unit]]](Ident(catcherName))
                      val tryBodyExpr = c.Expr(transform(
                        tree,
                        Ident(catcherName),
                        { (x) =>
                          val resultExpr = c.Expr(x)
                          val returnExpr = c.Expr[Any => Nothing](Ident(returnName))
                          reify {
                            val result = resultExpr.splice
                            _root_.scala.util.control.TailCalls.tailcall(returnExpr.splice.asInstanceOf[Any => _root_.scala.util.control.TailCalls.TailRec[Unit]](result))
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

          TypeApply(
            Select(Apply(Select(New(Ident(futureName)), nme.CONSTRUCTOR), List()), newTermName("asInstanceOf")),
            List(uncheckedBounds(futureTypeTree)))))
    }
    def newFuture(tree: Tree)(implicit forceAwait: Set[Name]): c.Expr[Future[Nothing]] = {
      newFutureAsType(tree, TypeTree(tree.tpe.widen).defineType(tree.tpe.widen))
    }
    val Apply(TypeApply(_, List(t)), _) = c.macroApplication
    val result = newFutureAsType(futureBody.tree, t)(Set.empty)
    c.warning(c.enclosingPosition, show(result.tree))
    val r = c.Expr(
      TypeApply(
        Select(
          c.resetLocalAttrs(result.tree),
          newTermName("asInstanceOf")),
        List(uncheckedBounds(AppliedTypeTree(Ident(futureType.typeSymbol), List(t))))))
    c.warning(c.enclosingPosition, "after/" +  show(r.tree))
    r
  }

  import scala.language.experimental.macros
  def apply[A](futureBody: => A): Future[A] = macro applyMacro

}