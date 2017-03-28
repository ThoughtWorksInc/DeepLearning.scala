package com.thoughtworks.deeplearning

import org.scalatest.{FreeSpec, Matchers}

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
final class ConstructorSpec extends FreeSpec with Matchers {
  import ConstructorSpec._

  "Constructor should create class instances" in {
    val a = Constructor[() => A].newInstance()
    a.getClass should be(classOf[A])
    a.i should be(0)
  }

  "Constructor should create abstract class instances" in {
    val x = Constructor[() => X].newInstance()
    x.getClass shouldNot be(classOf[X])
    x.i should be(42)
  }

  "Constructor should create class instances with mixins" in {
    val (ab: A with B, ac: A with C) = makePair[A, B, C]
    ab.i should be(1)
    ac.i should be(2)

    val (xb: X with B, xc: X with C) = makePair[X, B, C]
    xb.i should be(1)
    xc.i should be(2)
  }

}

private object ConstructorSpec {

  private[ConstructorSpec] class A {
    def i: Int = 0
  }

  private[ConstructorSpec] abstract class X extends A {
    override def i = 42
  }

  private[ConstructorSpec] trait B { this: A =>
    override def i = 1
  }

  private[ConstructorSpec] trait C { this: A =>
    override def i = 2
  }

  private[ConstructorSpec] def makePair[A, B, C](implicit constructor0: Constructor[() => A with B],
                                                 constructor1: Constructor[() => A with C]) = {
    (constructor0.newInstance(), constructor1.newInstance())
  }

}
