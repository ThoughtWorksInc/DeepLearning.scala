package com.thoughtworks.deeplearning

import org.scalatest.{FreeSpec, Matchers}

object Foo {
  def call(implicit caller: Caller[_]): String = {
    caller.value.getClass.getName
  }
}

class CallerSpec extends FreeSpec with Matchers {
  "className" in {
    val className: String = Foo.call
    className should be(this.getClass.getName)
  }

}
