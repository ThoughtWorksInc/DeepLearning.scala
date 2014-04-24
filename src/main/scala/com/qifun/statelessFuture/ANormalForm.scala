/*
 * stateless-future
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

import scala.runtime.AbstractPartialFunction
import scala.reflect.macros.Context
import scala.util.control.Exception.Catcher
import scala.util.control.TailCalls._
import scala.concurrent.ExecutionContext
import scala.util.Success
import scala.util.Failure

/**
 * For internal using only.
 */
object ANormalForm {

  abstract class AbstractAwaitable[+AwaitResult, TailRecResult] extends Awaitable.Stateless[AwaitResult, TailRecResult]

  @inline
  final def forceOnComplete[TailRecResult](future: Awaitable[Any, TailRecResult], handler: Nothing => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]): TailRec[TailRecResult] = {
    future.onComplete(handler.asInstanceOf[Any => TailRec[TailRecResult]])
  }

  final class TryCatchFinally[AwaitResult, TailRecResult](tryFuture: Awaitable[AwaitResult, TailRecResult], getCatcherFuture: Catcher[Awaitable[AwaitResult, TailRecResult]], finallyBlock: => Unit) extends AbstractAwaitable[AwaitResult, TailRecResult] {
    @inline
    override final def onComplete(rest: AwaitResult => TailRec[TailRecResult])(implicit catcher: Catcher[TailRec[TailRecResult]]) = {
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

  def applyMacro(c: Context)(futureBody: c.Expr[Any]): c.Expr[Nothing] = {
    import c.universe._
    c.macroApplication match {
      case Apply(TypeApply(_, List(t)), _) => {
        applyMacroWithType(c)(futureBody, t, Ident(typeOf[Unit].typeSymbol))
      }
      case Apply(TypeApply(_, List(t, tailRecResultTypeTree)), _) => {
        applyMacroWithType(c)(futureBody, t, tailRecResultTypeTree)
      }
    }
  }

  def applyMacroWithType(c: Context)(futureBody: c.Expr[Any], macroAwaitResultTypeTree: c.Tree, tailRecResultTypeTree: c.Tree): c.Expr[Nothing] = {

    import c.universe.Flag._
    import c.universe._
    import compat._

    def unchecked(tree: Tree) = {
      Annotated(Apply(Select(New(TypeTree(typeOf[unchecked])), nme.CONSTRUCTOR), List()), tree)
    }

    val abstractPartialFunction = typeOf[AbstractPartialFunction[_, _]]
    val futureType = typeOf[Awaitable[_, _]]
    val statelessFutureType = typeOf[Awaitable.Stateless[_, _]]
    val futureClassType = typeOf[AbstractAwaitable[_, _]]
    val function1Type = typeOf[_ => _]
    val function1Symbol = function1Type.typeSymbol

    val tailRecType = typeOf[TailRec[_]]
    val tailRecSymbol = tailRecType.typeSymbol
    val uncheckedSymbol = typeOf[scala.unchecked].typeSymbol
    val AndSymbol = typeOf[Boolean].declaration(newTermName("&&").encodedName)
    val OrSymbol = typeOf[Boolean].declaration(newTermName("||").encodedName)

    val tailRecTypeTree = AppliedTypeTree(Ident(tailRecSymbol), List(tailRecResultTypeTree))
    val catcherTypeTree = AppliedTypeTree(Ident(typeOf[PartialFunction[_, _]].typeSymbol), List(Ident(typeOf[Throwable].typeSymbol), tailRecTypeTree))

    def checkNakedAwait(tree: Tree, errorMessage: String) {
      for (subTree @ Select(future, await) <- tree if await.decoded == "await" && future.tpe <:< futureType) {
        c.error(subTree.pos, errorMessage)
      }
    }

    def transformParameterList(isByNameParam: List[Boolean], trees: List[Tree], catcher: Tree, rest: (List[Tree]) => Tree)(implicit forceAwait: Set[Name]): Tree = {
      trees match {
        case Nil => rest(Nil)
        case head :: tail => {
          head match {
            case Typed(origin, typeTree) =>
              transform(origin, catcher, { transformedOrigin =>
                val parameterName = newTermName(c.fresh("yangBoParameter"))
                Block(
                  List(ValDef(Modifiers(), parameterName, TypeTree(), transformedOrigin).setPos(origin.pos)),
                  transformParameterList(if (isByNameParam.nonEmpty) isByNameParam.tail else Nil, tail, catcher, { transformedTail =>
                    rest(treeCopy.Typed(head, Ident(parameterName), typeTree) :: transformedTail)
                  }))
              })
            case _ if isByNameParam.nonEmpty && isByNameParam.head => {
              checkNakedAwait(head, "await must not be used under a by-name argument.")
              transformParameterList(isByNameParam.tail, tail, catcher, { (transformedTail) =>
                rest(head :: transformedTail)
              })
            }
            case _ =>
              transform(head, catcher, { transformedHead =>
                val parameterName = newTermName(c.fresh("yangBoParameter"))
                Block(
                  List(ValDef(Modifiers(), parameterName, TypeTree(), transformedHead).setPos(head.pos)),
                  transformParameterList(if (isByNameParam.nonEmpty) isByNameParam.tail else Nil, tail, catcher, { transformedTail =>
                    rest(Ident(parameterName) :: transformedTail)
                  }))
              })

          }
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
      val futureExpr = c.Expr(future)
      val AwaitResult = newTermName(c.fresh("awaitValue"))
      val catcherExpr = c.Expr[Catcher[Nothing]](catcher)
      val onCompleteCallExpr = c.Expr(
        Apply(
          Apply(
            Select(reify(_root_.com.qifun.statelessFuture.ANormalForm).tree, newTermName("forceOnComplete")),
            List(
              future,
              {
                val restTree = rest(Ident(AwaitResult))
                val tailcallSymbol = reify(scala.util.control.TailCalls).tree.symbol
                val TailcallName = newTermName("tailcall")
                val ApplyName = newTermName("apply")
                val AsInstanceOfName = newTermName("asInstanceOf")
                def function(awaitValDef: ValDef, restTree: Tree) = {
                  val restExpr = c.Expr(restTree)
                  Function(
                    List(awaitValDef),
                    reify {
                      try {
                        restExpr.splice
                      } catch {
                        case e if catcherExpr.splice.isDefinedAt(e) => {
                          _root_.scala.util.control.TailCalls.tailcall(catcherExpr.splice(e))
                        }
                      }
                    }.tree).setPos(future.pos)
                }
                restTree match {
                  case Block(
                    List(ValDef(_, capturedResultName1, _, Ident(AwaitResult))),
                    Apply(
                      Select(tailCallsTree, TailcallName),
                      List(
                        Apply(
                          Select(
                            capturedReturnTree,
                            ApplyName),
                          List(Ident(capturedResultName2)))))) if capturedResultName1 == capturedResultName2 && tailCallsTree.symbol == tailcallSymbol => {
                    // 尾调用优化
                    capturedReturnTree
                  }
                  case Block(List(oldVal @ ValDef(_, capturedResultName, _, Ident(AwaitResult))), expr) => {
                    // 参数名优化
                    function(treeCopy.ValDef(oldVal, Modifiers(PARAM), capturedResultName, awaitTypeTree, EmptyTree), expr)
                  }
                  case Block(List(oldVal @ ValDef(_, capturedResultName, _, Ident(AwaitResult)), restStates @ _*), expr) => {
                    // 参数名优化
                    function(treeCopy.ValDef(oldVal, Modifiers(PARAM), capturedResultName, awaitTypeTree, EmptyTree), Block(restStates.toList, expr))
                  }
                  case _ => {
                    function(ValDef(Modifiers(PARAM), AwaitResult, awaitTypeTree, EmptyTree), restTree)
                  }
                }
              })),
          List(catcher)))
      reify {
        val yangBoNextFuture = futureExpr.splice
        _root_.scala.util.control.TailCalls.tailcall { onCompleteCallExpr.splice }
      }.tree
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
                        Select(reify(_root_.com.qifun.statelessFuture.ANormalForm).tree,
                          newTypeName("TryCatchFinally")),
                        List(TypeTree(tree.tpe), tailRecResultTypeTree))),
                    nme.CONSTRUCTOR),
                  List(
                    newFuture(block).tree,
                    newCatcher(
                      for (cd @ CaseDef(pat, guard, body) <- catches) yield {
                        treeCopy.CaseDef(cd, pat, guard, newFuture(body).tree)
                      },
                      AppliedTypeTree(
                        Ident(futureType.typeSymbol),
                        List(TypeTree(tree.tpe), tailRecResultTypeTree))),
                    if (finalizer.isEmpty) {
                      Literal(Constant(()))
                    } else {
                      checkNakedAwait(finalizer, "await must not be used under a finally.")
                      finalizer
                    })))),
            transformAwait(Ident(futureName), TypeTree(tree.tpe), catcher, rest))

        }
        case ClassDef(mods, _, _, _) => {
          if (mods.hasFlag(TRAIT)) {
            checkNakedAwait(tree, "await must not be used under a nested trait.")
          } else {
            checkNakedAwait(tree, "await must not be used under a nested class.")
          }
          rest(tree)
        }
        case _: ModuleDef => {
          checkNakedAwait(tree, "await must not be used under a nested object.")
          rest(tree)
        }
        case DefDef(mods, _, _, _, _, _) => {
          if (mods.hasFlag(LAZY)) {
            checkNakedAwait(tree, "await must not be used under a lazy val initializer.")
          } else {
            checkNakedAwait(tree, "await must not be used under a nested method.")
          }
          rest(tree)
        }
        case _: Function => {
          checkNakedAwait(tree, "await must not be used under a nested function.")
          rest(tree)
        }
        case EmptyTree | _: New | _: Ident | _: Literal | _: Super | _: This | _: TypTree | _: New | _: TypeDef | _: Import | _: ImportSelector => {
          rest(tree)
        }
        case Select(future, await) if await.decoded == "await" && future.tpe <:< futureType => {
          transform(future, catcher, { (transformedFuture) =>
            transformAwait(transformedFuture, TypeTree(tree.tpe), catcher, rest)
          })
        }
        case Select(instance, field) => {
          transform(instance, catcher, { (transformedInstance) =>
            rest(treeCopy.Select(tree, transformedInstance, field))
          })
        }
        case TypeApply(method, parameters) => {
          transform(method, catcher, { (transformedMethod) =>
            rest(treeCopy.TypeApply(tree, transformedMethod, parameters))
          })
        }
        case Apply(method @ Ident(name), parameters) if forceAwait(name) => {
          transformParameterList(Nil, parameters, catcher, { (transformedParameters) =>
            transformAwait(treeCopy.Apply(tree, Ident(name), transformedParameters), TypeTree(tree.tpe).asInstanceOf[TypTree], catcher, rest)
          })
        }
        case Apply(method, parameters) => {
          transform(method, catcher, { (transformedMethod) =>
            val isByNameParam = method.symbol match {
              case AndSymbol | OrSymbol => {
                List(true)
              }
              case _ => {
                method.tpe match {
                  case MethodType(params, _) => {
                    for (param <- params) yield {
                      param.asTerm.isByNameParam
                    }
                  }
                  case _ => {
                    Nil
                  }
                }
              }
            }
            transformParameterList(isByNameParam, parameters, catcher, { (transformedParameters) =>
              rest(treeCopy.Apply(
                tree,
                transformedMethod,
                transformedParameters))
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
              rest(treeCopy.Assign(tree, transformedLeft, transformedRight))
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
                  checkNakedAwait(guard, "await must not be used under a pattern guard.")
                  treeCopy.CaseDef(originCaseDef,
                    pat,
                    guard,
                    newFutureAsType(body, TypeTree(tree.tpe)).tree)
                }),
              TypeTree(tree.tpe),
              catcher,
              rest)
          })
        }
        case If(cond, thenp, elsep) => {
          transform(cond, catcher, { (transformedCond) =>
            transformAwait(
              If(
                transformedCond,
                newFuture(thenp).tree,
                newFuture(elsep).tree),
              TypeTree(tree.tpe),
              catcher,
              rest)
          })
        }
        case Throw(throwable) => {
          transform(throwable, catcher, { (transformedThrowable) =>
            rest(treeCopy.Throw(tree, transformedThrowable))
          })
        }
        case Typed(expr, tpt) => {
          transform(expr, catcher, { (transformedExpr) =>
            rest(treeCopy.Typed(tree, transformedExpr, tpt))
          })
        }
        case Annotated(annot, arg) => {
          transform(arg, catcher, { (transformedArg) =>
            rest(treeCopy.Annotated(tree, annot, transformedArg))
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
                AppliedTypeTree(Ident(futureType.typeSymbol), List(TypeTree(tree.tpe), tailRecResultTypeTree)),
                newFuture(rhs)(forceAwait + name).tree)),
            transformAwait(
              Apply(
                Ident(name),
                params),
              TypeTree(tree.tpe),
              catcher,
              rest))
        }
        case _: Return => {
          c.error(tree.pos, "return is illegal.")
          rest(tree)
        }
        case _: PackageDef | _: Template | _: CaseDef | _: Alternative | _: Star | _: Bind | _: UnApply | _: AssignOrNamedArg | _: ReferenceToBoxed => {
          c.error(tree.pos, s"Unexpected expression in a `Future` block")
          rest(tree)
        }
      }
    }

    def newFutureAsType(tree: Tree, awaitValueTypeTree: Tree)(implicit forceAwait: Set[Name]): c.Expr[Awaitable.Stateless[Nothing, _]] = {

      val statelessFutureTypeTree = AppliedTypeTree(Ident(statelessFutureType.typeSymbol), List(awaitValueTypeTree, tailRecResultTypeTree))
      val futureClassTypeTree = AppliedTypeTree(Ident(futureClassType.typeSymbol), List(awaitValueTypeTree, tailRecResultTypeTree))

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
                List(futureClassTypeTree),
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
                        AppliedTypeTree(Ident(function1Symbol), List(awaitValueTypeTree, tailRecTypeTree)),
                        EmptyTree)),
                      List(ValDef(
                        Modifiers(IMPLICIT | PARAM),
                        catcherName,
                        catcherTypeTree,
                        EmptyTree))),
                    tailRecTypeTree,
                    {
                      val catcherExpr = c.Expr[Catcher[TailRec[Nothing]]](Ident(catcherName))
                      val tryBodyExpr = c.Expr(transform(
                        tree,
                        Ident(catcherName),
                        { (x) =>
                          val resultExpr = c.Expr(x)
                          val returnExpr = c.Expr[Any => Nothing](Ident(returnName))
                          reify {
                            val result = resultExpr.splice
                            // Workaround for some nested Future blocks.
                            _root_.scala.util.control.TailCalls.tailcall((returnExpr.splice).asInstanceOf[Any => _root_.scala.util.control.TailCalls.TailRec[Nothing]].apply(result))
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
            unchecked(statelessFutureTypeTree))))
    }
    def newFuture(tree: Tree)(implicit forceAwait: Set[Name]): c.Expr[Awaitable.Stateless[Nothing, _]] = {
      newFutureAsType(tree, TypeTree(tree.tpe.widen))
    }
    val result = newFutureAsType(futureBody.tree, macroAwaitResultTypeTree)(Set.empty)
    //      c.warning(c.enclosingPosition, show(result))
    c.Expr(
      TypeApply(
        Select(
          c.resetLocalAttrs(result.tree),
          newTermName("asInstanceOf")),
        List(unchecked(AppliedTypeTree(Ident(statelessFutureType.typeSymbol), List(macroAwaitResultTypeTree, tailRecResultTypeTree))))))
  }

}