---
layout: doc
title: Export Scala.js APIs to JavaScript
---

By default, Scala.js classes, objects, methods and properties are not available
to JavaScript. Entities that have to be accessed from JavaScript must be
annotated explicitly as *exported*. The `@JSExport` annotation is the main way
to do this.

## A simple example

{% highlight scala %}
package example

import scala.scalajs.js
import js.annotation.JSExport

@JSExport
object HelloWorld {
  @JSExport
  def main(): Unit = {
    println("Hello world!")
  }
}
{% endhighlight %}

This allows to call the `main()` method of `HelloWorld` like this in JavaScript:

{% highlight javascript %}
HelloWorld().main();
{% endhighlight %}

Note the `()` when accessing the object, `HelloWorld` is a function.

You have probably already used an `@JSExport` without knowing it
through the `JSApp` trait in the `Main` class of the bootstrapping
skeleton (or any other template of Scala.js application). In fact, any
Scala.js application must export at least a class or an object and a
method in order to be invokable at all.

Most of the time, however, it is sufficient to just extend the `JSApp`
trait:

{% highlight scala %}
package example

import scala.scalajs.js
import js.annotation.JSExport

object HelloWorld extends js.JSApp {
  def main(): Unit = {
    println("Hello world!")
  }
}
{% endhighlight %}

And call like this (see documentation about
`@JSExportDescendentObjects` below for internal workings):

{% highlight javascript %}
example.HelloWorld().main();
{% endhighlight %}

## Exporting top-level objects

Put on a top-level object, the `@JSExport` annotation exports a zero-argument
function returning that object in JavaScript's global scope. By default, the
function has the same name as the object in Scala (unqualified).

{% highlight scala %}
@JSExport
object HelloWorld {
  ...
}
{% endhighlight %}

exports the `HelloWorld()` function in JavaScript.

`@JSExport` takes an optional string parameter to specify a non-default name
for JavaScript. For example,

{% highlight scala %}
@JSExport("MainObject")
object HelloWorld {
  ...
}
{% endhighlight %}

exports the `HelloWorld` object under the function `MainObject()` in JavaScript.

The name can contain dots, in which case the exported function is namespaced
in JavaScript.

{% highlight scala %}
@JSExport("myapp.foo.MainObject")
object HelloWorld {
  ...
}
{% endhighlight %}

will be accessible in JavaScript using `myapp.foo.MainObject()`.

## Exporting classes

The `@JSExport` annotation can also be used to export Scala.js classes to
JavaScript (but not traits), or, to be more precise, their constructors. This
allows JavaScript code to create instances of the class.

{% highlight scala %}
@JSExport
class Foo(val x: Int) {
  override def toString(): String = s"Foo($x)"
}
{% endhighlight %}

exposes `Foo` as a constructor function to JavaScript:

{% highlight javascript %}
var foo = new Foo(3);
console.log(foo.toString());
{% endhighlight %}

will log the string `"Foo(3)"` to the console. This particular example works
because it calls `toString()`, which is always exported to JavaScript. Other
methods must be exported explicitly as shown in the next section.

As is the case for top-level objects, classes can be exported under custom
names, including namespaced ones, by giving an explicit name to `@JSExport`.

## Exports with modules

When [emitting a module for Scala.js code](../project/module.html), top-level exports are not sent to the JavaScript global scope.
Instead, they are genuinely exported from the module.
In that case, a top-level `@JSExport` annotation has the semantics of an [ECMAScript 2015 export](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/export).
For example:

{% highlight scala %}
@JSExport("Bar")
class Foo(val x: Int)
{% endhighlight %}

is semantically equivalent to this JavaScript export:

{% highlight javascript %}
export { Foo as Bar };
{% endhighlight %}

## Exporting methods

Similarly to objects, methods of Scala classes, traits and objects can be
exported with `@JSExport`, with or without an explicit name.

{% highlight scala %}
class Foo(val x: Int) {
  @JSExport
  def square(): Int = x*x // note the (), omitting them has a different behavior
  @JSExport("foobar")
  def add(y: Int): Int = x+y
}
{% endhighlight %}

Given this definition, and some variable `foo` holding an instance of `Foo`,
you can call:

{% highlight javascript %}
console.log(foo.square());
console.log(foo.foobar(5));
// console.log(foo.add(3)); // TypeError, add is not a member of foo
{% endhighlight %}

### Overloading

Several methods can be exported with the same JavaScript name (either because
they have the same name in Scala, or because they have the same explicit
JavaScript name as parameter of `@JSExport`). In that case, run-time overload
resolution will decide which method to call depending on the number and run-time
types of arguments passed to the the method.

For example, given these definitions:

{% highlight scala %}
class Foo(val x: Int) {
  @JSExport
  def foobar(): Int = x
  @JSExport
  def foobar(y: Int): Int = x+y
  @JSExport("foobar")
  def bar(b: Boolean): Int = if (b) 0 else x
}
{% endhighlight %}

the following calls will dispatch to each of the three methods:

{% highlight javascript %}
console.log(foo.foobar());
console.log(foo.foobar(5));
console.log(foo.foobar(false));
{% endhighlight %}

If the Scala.js compiler cannot produce a dispatching code capable of reliably
disambiguating overloads, it will issue a compile error (with a somewhat cryptic
message):

{% highlight scala %}
class Foo(val x: Int) {
  @JSExport
  def foobar(): Int = x
  @JSExport
  def foobar(y: Int): Int = x+y
  @JSExport("foobar")
  def bar(i: Int): Int = if (i == 0) 0 else x
}
{% endhighlight %}

gives:

    [error] HelloWorld.scala:16: double definition:
    [error] method $js$exported$meth$foobar:(i: Int)Any and
    [error] method $js$exported$meth$foobar:(y: Int)Any at line 14
    [error] have same type
    [error]   @JSExport("foobar")
    [error]    ^
    [error] one error found

Hint to recognize this error: the methods are named `$js$exported$meth$`
followed by the JavaScript export name.

### <a name="JSExportNamed"></a> Exporting for call with named parameters (deprecated)

**Note:** Since Scala.js 0.6.11, `@JSExportNamed` is deprecated, and will be removed in the next major version.
Refer to [the Scaladoc]({{ site.production_url }}/api/scalajs-library/latest/#scala.scalajs.js.annotation.JSExportNamed) for migration tips.

It is customary in Scala to call methods with named parameters if this eases understanding of the code or if many arguments with default values are present:

{% highlight scala %}
def foo(x: Int = 1, y: Int = 2, z: Int = 3) = ???

foo(y = 3, x = 2)
{% endhighlight %}

A rough equivalent in JavaScript is to pass an object with the respective properties:
{% highlight javascript %}
foo({
  y: 3,
  x: 2
});
{% endhighlight %}

The `@JSExportNamed` annotation allows to export Scala methods for use in JavaScript with named parameters:

{% highlight scala %}
class A {
  @JSExportNamed
  def foo(x: Int, y: Int = 2, z: Int = 3) = ???
}
{% endhighlight %}

Note that default parameters are not required. `foo` can then be called like this:
{% highlight javascript %}
var a = // ...
a.foo({
  y: 3,
  x: 2
});
{% endhighlight %}

Not specifying `x` in this case will fail at runtime (since it does not have a default value).

Just like `@JSExport`, `@JSExportNamed` takes the name of the exported method as an optional argument.

## Exporting top-level methods

While an `@JSExport`ed method inside an `@JSExport`ed object allows JavaScript code to call a "static" method,
it does not feel like a top-level function from JavaScript's point of view.
`@JSExportTopLevel` allows to export a method of a top-level object as a truly top-level function:

{% highlight scala %}
object A {
  @JSExportTopLevel("foo")
  def foo(x: Int): Int = x + 1
}
{% endhighlight %}

can be called from JavaScript as:

{% highlight javascript %}
const y = foo(5);
{% endhighlight %}

Note that `@JSExportTopLevel` requires an explicit name under which to export the function.

## Exporting properties

`val`s, `var`s and `def`s without parentheses, as well as `def`s whose name
ends with `_=`, have a single argument and `Unit` result type, are
exported to JavaScript as properties with getters and/or setters
using, again, the `@JSExport` annotation.

Given this weird definition of a halfway mutable point:

{% highlight scala %}
@JSExport
class Point(_x: Double, _y: Double) {
  @JSExport
  val x: Double = _x
  @JSExport
  var y: Double = _y
  @JSExport
  def abs: Double = Math.sqrt(x*x + y*y)
  @JSExport
  def sum: Double = x + y
  @JSExport
  def sum_=(v: Double): Unit = y = v - x
}
{% endhighlight %}

JavaScript code can use the properties as follows:

{% highlight javascript %}
var point = new Point(4, 10)
console.log(point.x);   // 4
console.log(point.y);   // 10
point.y = 20;
console.log(point.y);   // 20
point.x = 1;            // does nothing, thanks JS semantics
console.log(point.x);   // still 4
console.log(point.abs); // 20.396078054371138
console.log(point.sum); // 24
point.sum = 30;
console.log(point.sum); // 30
console.log(point.y);   // 26
{% endhighlight %}

As usual, explicit names can be given to `@JSExport`. For `def` setters, the
JS name must be specified *without* the trailing `_=`.

`def` setters must have a result type of `Unit` and exactly one parameter. Note
that several `def` setters with different types for their argument can be
exported under a single, overloaded JavaScript name.

In case you overload properties in a way the compiler cannot
disambiguate, the methods in the error messages will be prefixed by
`$js$exported$prop$`.

### <a name="constructor-params"></a> Export fields directly declared in constructors
If you want to export fields that are directly declared in a class constructor, you'll have to use the `@field` meta annotation to avoid annotating the constructor arguments (exporting an argument is nonsensical and will fail):

{% highlight scala %}
import scala.annotation.meta.field

class Point(
    @(JSExport @field) val x: Double,
    @(JSExport @field) val y: Double)

// Also applies to case classes
case class Point(
    @(JSExport @field) x: Double,
    @(JSExport @field) y: Double)
{% endhighlight %}

## Automatically exporting descendent objects
Sometimes it is desirable to automatically export all descendent
objects of a given trait or class. You can use the
`@JSExportDescendentObjects` annotation. It will cause all descendent
objects to be exported to their fully qualified name.

This feature is especially useful in conjunction with exported
abstract methods and is used by the test libraries of Scala.js and the
`scala.scalajs.js.JSApp` trait. The following is just an example, how
the feature can be used:

{% highlight scala %}
package foo.test

@JSExportDescendentObjects
trait Test {
  @JSExport
  def test(param: String): Unit
}

// automatically exported as foo.test.Test1
object Test1 extends Test {
  // exported through inheritance
  def test(param: String) = {
    println(param)
  }
}
{% endhighlight %}

## <a name="JSExportAll"></a> Automatically export all members
Instead of writing `@JSExport` on every member of a class or object, you may use the `@JSExportAll` annotation. It is equivalent to adding `@JSExport` on every public (term) member directly declared in the class/object:

{% highlight scala %}
class A {
  def mul(x: Int, y: Int): Int = x * y
}

@JSExportAll
class B(val a: Int) extends A {
  def sum(x: Int, y: Int): Int = x + y
}
{% endhighlight %}

This is strictly equivalent to writing:

{% highlight scala %}
class A {
  def mul(x: Int, y: Int): Int = x * y
}

class B(@(JSExport @field) val a: Int) extends A {
  @JSExport
  def sum(x: Int, y: Int): Int = x + y
}
{% endhighlight %}

It is important to note that this does **not** export inherited members. If you wish to do so, you'll have to override them explicitly:

{% highlight scala %}
class A {
  def mul(x: Int, y: Int): Int = x * y
}

@JSExportAll
class B(val a: Int) extends A {
  override def mul(x: Int, y: Int): Int = super.mul(x,y)
  def sum(x: Int, y: Int): Int = x + y
}
{% endhighlight %}
