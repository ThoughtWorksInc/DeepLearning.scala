package com.qifun.statelessFuture

import scala.reflect.macros.Context

object AwaitableFactory {

  /**
   * Used internally only.
   */
  final def applyMacro(c: Context)(block: c.Expr[Any]): c.Expr[Nothing] = {
    import c.universe._
    val Apply(TypeApply(Select(factory, _), List(t)), _) = c.macroApplication
    val factoryName = newTermName(c.fresh("yangBoAwaitableFactory"))
    val factoryVal = c.Expr(ValDef(Modifiers(), factoryName, TypeTree(), factory))
    val result = ANormalForm.applyMacroWithType(c)(block, t, Select(Ident(factoryName), newTypeName("TailRecResult")))
    reify {
      factoryVal.splice
      result.splice
    }
  }

  final def apply[TailRecResult] = new AwaitableFactory[TailRecResult] {}

}

trait AwaitableFactory[TRR] {

  type TailRecResult = TRR

  import scala.language.experimental.macros

  /**
   * Returns a stateless [[Awaitable]] that evaluates the `block`.
   * @param block The asynchronous operation that will perform later. Note that all [[Awaitable#await]] calls must be in the `block`.
   */
  final def apply[AwaitResult](block: AwaitResult): Awaitable[AwaitResult, TailRecResult] = macro AwaitableFactory.applyMacro

}