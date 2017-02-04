package com.thoughtworks.deeplearning

import com.thoughtworks.deeplearning.OpenCL.{DslFunction, DslType, TypedFunction}
import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class OpenCLSpec extends FreeSpec with Matchers {
  "Add" in {

    val f = DslFunction.Add(DslFunction.DoubleLiteral(1.5), DslFunction.DoubleLiteral(1.5), DslType.DslDouble)
    val tf = TypedFunction(f, DslType.DslHNil, DslType.DslDouble)
    val cl = OpenCL.compile(Map("f" -> tf)).toString
    cl should not be empty
    // TODO: compile it
  }
}
