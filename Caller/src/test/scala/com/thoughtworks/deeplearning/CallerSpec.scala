package com.thoughtworks.deeplearning

import org.scalatest.{FreeSpec, Matchers}
object CallerSpec {
  object Foo {
    def call(implicit caller: Caller[_]): String = {
      caller.value.getClass.getName
    }
  }
}

final class CallerSpec extends FreeSpec with Matchers {
  import CallerSpec._
  "className" in {
    val className: String = Foo.call
    className should be(this.getClass.getName)
  }

}
