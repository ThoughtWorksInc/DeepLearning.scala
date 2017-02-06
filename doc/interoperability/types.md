---
layout: doc
title: JavaScript types
---

Understanding how different types are mapped between Scala.js and JavaScript is crucial for correct interoperability.
Some types map quite directly (like `String`) where others require some conversions. 
 
## <a name="type-correspondence"></a> Type Correspondence

Some Scala types are directly mapped to corresponding underlying JavaScript types. These correspondences can be used
when calling Scala.js code from JavaScript and when defining typed interfaces for JavaScript code.

<table class="table table-bordered">
  <thead>
    <tr><th>Scala type</th><th>JavaScript type</th><th>Restrictions</th></tr>
  </thead>
  <tbody>
    <tr><td>java.lang.String</td><td>string</td><td></td></tr>
    <tr><td>scala.Boolean</td><td>boolean</td><td></td></tr>
    <tr><td>scala.Char</td><td><i>opaque</i></td><td></td></tr>
    <tr><td>scala.Byte</td><td>number</td><td>integer, range (-128, 127)</td></tr>
    <tr><td>scala.Short</td><td>number</td><td>integer, range (-32768, 32767)</td></tr>
    <tr><td>scala.Int</td><td>number</td><td>integer, range (-2147483648, 2147483647)</td></tr>
    <tr><td>scala.Long</td><td><i>opaque</i></td><td></td></tr>
    <tr><td>scala.Float</td><td>number</td><td></td></tr>
    <tr><td>scala.Double</td><td>number</td><td></td></tr>
    <tr><td>scala.Unit</td><td>undefined</td><td></td></tr>
    <tr><td>scala.Null</td><td>null</td><td></td></tr>
    <tr>
      <td>subtypes of `js.Any`</td>
      <td><i>themselves</i></td>
      <td>see <a href="facade-types.html">the facade types guide</a></td>
    </tr>
    <tr>
      <td>
        other Scala classes<br />
        <small>including value classes</small>
      </td>
      <td>
        <i>opaque, except for exported methods</i><br />
        <small>Note: <code>toString()</code> is always exported</small>
      </td>
      <td>see <a href="export-to-javascript.html">exporting Scala.js APIs to JavaScript</a></td>
    </tr>
  </tbody>
</table>

On the other hand, some JavaScript (collection) types have similar types in Scala.
Instead of mapping them directly, Scala.js provides conversions between them.
We show with a couple of snippets how you can convert from JavaScript to Scala types and back.
Please refer to the [Scaladocs]({{ BASE_PATH }}/doc/index.html#api) for details.

#### `js.FunctionN` <--> `scala.FunctionN`

Functions from JavaScript and Scala are not exactly the same thing, therefore
they have different types. However, implicit conversions are available by
default to go from one to the other, which means the following snippets compile
out of the box:

{% highlight scala %}
import scala.scalajs.js

val scalaFun: Int => Int = (x: Int) => x * x
val jsFun: js.Function1[Int, Int] = scalaFun
val scalaFunAgain: Int => Int = jsFun
{% endhighlight %}

Most of the time, you don't even need to worry about those, except if you
[write facade types for JavaScript APIs](facade-types.html), in which case you
have to use the JS function types.

#### `js.Array[T]` <--> `mutable.Seq[T]`

{% highlight scala %}
import scala.scalajs.js

val jsArr = js.Array(1, 2, 3)

// Scala style operations on js.Array (returns a js.Array)
val x: js.Array[Int] = jsArr.takeWhile(_ < 3)

// Use a js.Array as a Scala mutable.Seq
val y: mutable.Seq[Int] = jsArr

// toArray (from js.ArrayOps) -- Copy into scala.Array
val z: scala.Array[Int] = jsArr.toArray

import js.JSConverters._

val scSeq = Seq(1, 2, 3)

// Seq to js.Array -- Copy to js.Array
val jsArray: js.Array[Int] = scSeq.toJSArray
{% endhighlight %}

#### `js.Dictionary[T]` <--> `mutable.Map[String, T]`

{% highlight scala %}
import scala.scalajs.js

val jsDict = js.Dictionary("a" -> 1, "b" -> 2)

// Scala style operations on js.Dictionary (returns mutable.Map)
val x: mutable.Map[String, Int] = jsDict.mapValues(_ * 2)

// Use a js.Dictionary as Scala mutable.Map
val y: mutable.Map[String, Int] = jsDict

import js.JSConverters._

val scMap = Map("a" -> 1, "b" -> 2)

// Map to js.Dictionary -- Copy to js.Dictionary
val jsDictionary: js.Dictionary[Int] = scMap.toJSDictionary
{% endhighlight %}

#### `js.UndefOr[T]` <--> `Option[T]`

{% highlight scala %}
import scala.scalajs.js

val jsUndefOr: js.UndefOr[Int] = 1

// Convert to scala.Option
val x: Option[Int] = jsUndefOr.toOption

import js.JSConverters._

val opt = Some(1)

// Convert to js.Undefined
val y: js.UndefOr[Int] = opt.orUndefined
{% endhighlight %}

## Pre-defined JavaScript types

Primitive JavaScript types (`number`, `boolean`, `string`, `null` and
`undefined`) are represented by their natural equivalent in Scala, as shown
[above](#type-correspondence).

For other pre-defined JavaScript types, such as arrays and functions, the package `scala.scalajs.js`
([ScalaDoc]({{ site.production_url }}/api/scalajs-library/{{ site.versions.scalaJS }}/#scala.scalajs.js.package))
provides dedicated definitions.

The class hierarchy for these standard types is as follows:

    js.Any
     +- js.Object
     |   +- js.Date
     |   +- js.RegExp
     |   +- js.Array[A]
     |   +- js.Function
     |       +- js.Function0[+R]
     |       +- js.Function1[-T1, +R]
     |       +- ...
     |       +- js.Function22[-T1, ..., -T22, +R]
     |       +- js.ThisFunction
     |           +- js.ThisFunction0[-T0, +R]
     |           +- js.ThisFunction1[-T0, -T1, +R]
     |           +- ...
     |           +- js.ThisFunction21[-T0, ..., -T21, +R]
     +- js.Dictionary[A]

Note that most of these types are similar to standard Scala types. For example,
`js.Array[A]` is similar to `scala.Array[A]`, and `js.FunctionN` is similar to
`scala.FunctionN`. However, they are not completely equivalent, and must not be confused.

With the exception of `js.Array[A]` and `js.Dictionary[A]`, these types have
all the fields and methods available in the JavaScript API.
The collection types feature the standard Scala collection API instead, so that
they can be used idiomatically in Scala code.

**0.5.x note**: In Scala.js 0.5.x, `js.Array[A]` and `js.Dictionary[A]` did not
really have the collection API. The methods defined in JavaScript took
precedence. This was changed in 0.6.x to avoid pitfalls when confusing the
APIs, avoiding common JavaScript warts, and improving performance.

## Function types

### `js.Function` and its subtypes

`js.FunctionN[T1, ..., TN, R]` is, as expected, the type of a JavaScript
function taking N parameters of types `T1` to `TN`, and returning a value of
type `R`.

There are implicit conversions from `scala.FunctionN` to `js.FunctionN` and
back, with the obvious meaning.
These conversions are the only way to create a `js.FunctionN` in Scala.js.
For example:

{% highlight scala %}
val f: js.Function1[Double, Double] = { (x: Double) => x*x }
{% endhighlight %}

defines a JavaScript `function` object which squares its argument.
This corresponds to the following JavaScript code:

{% highlight javascript %}
var f = function(x) {
  return x*x;
};
{% endhighlight %}

You can call a `js.FunctionN` in Scala.js with the usual syntax:

{% highlight scala %}
val y = f(5)
{% endhighlight %}

### `js.ThisFunction` and its subtypes

The series of `js.ThisFunctionN` solve the problem of modeling the `this`
value of JavaScript in Scala. Consider the following call to the `each` method
of a jQuery object:

{% highlight javascript %}
var lis = jQuery("ol > li");
lis.each(function() {
  jQuery(this).text(jQuery(this).text() + " - transformed")
});
{% endhighlight %}

Inside the closure, the value of `this` is the DOM element currently being
enumerated. This usage of `this`, which is nonsense from a Scala point of view,
is standard in JavaScript. `this` can actually be thought of as an additional
parameter to the closure.

In Scala.js, the `this` keyword always follows the same rules as in Scala,
i.e., it binds to the enclosing class, trait or object. It will never bind to
the equivalent of the JavaScript `this` in an anonymous function.

To access the JavaScript `this` in Scala.js, it can be made explicit using
`js.ThisFunctionN`. A `js.ThisFunctionN[T0, T1, ..., TN, R]` is the type of a
JavaScript function taking a `this` parameter of type `T0`, as well as N
normal parameters of types `T1` to `TN`, and returning a value of type `R`.
From Scala.js, the `this` parameter appears as any other parameter: it has a
non-keyword name, a type, and is listed first in the parameter list. Hence,
a `scala.FunctionN` is convertible to/from a `js.ThisFunction{N-1}`.

The previous example would be written as follows in Scala.js:

{% highlight scala %}
val lis = jQuery("ol > li")
lis.each({ (li: dom.HTMLElement) =>
  jQuery(li).text(jQuery(li).text() + " - transformed")
}: js.ThisFunction)
{% endhighlight %}

Skipping over the irrelevant details, note that the parameter `li` completely
corresponds to the JavaScript `this`. Note also that we have ascribed the
lambda with `: js.ThisFunction` explicitly to make sure that the right implicit
conversion is being used (by default it would convert it to a `js.Function1`).
If you call a statically typed API which expects a `js.ThisFunction0`, this is
not needed.

The mapping between JS `this` and first parameter of a `js.ThisFunction` also
works in the other direction, i.e., if calling the `apply` method of a
`js.ThisFunction`, the first actual argument is transferred to the called
function as its `this`. For example, the following snippet:

{% highlight scala %}
val f: js.ThisFunction1[js.Object, js.Number, js.Number] = ???
val o = new js.Object
val x = f(o, 4)
{% endhighlight %}

will map to

{% highlight javascript %}
var f = ...;
var o = new Object();
var x = f.call(o, 4);
{% endhighlight %}

## Dynamically typed interface: `js.Dynamic`

Because JavaScript is dynamically typed, it is not often practical, sometimes
impossible, to give sensible type definitions for JavaScript APIs.

Scala.js lets you call JavaScript in a dynamically typed fashion if you
want to. The basic entry point is to grab a dynamically typed reference to the
global scope, with `js.Dynamic.global`, which is of type `js.Dynamic`.

You can read and write any field of a `js.Dynamic`, as well as call any method
with any number of arguments, and you always receive back a `js.Dynamic`.

For example, this snippet taken from the Hello World example uses the
dynamically typed interface to manipulate the DOM model.

{% highlight scala %}
val document = js.Dynamic.global.document
val playground = document.getElementById("playground")

val newP = document.createElement("p")
newP.innerHTML = "Hello world! <i>-- DOM</i>"
playground.appendChild(newP)
{% endhighlight %}

In this example, `document`, `playground` and `newP` are all inferred to be of
type `js.Dynamic`.

### Literal object construction

Scala.js provides two syntaxes for creating JavaScript objects in a literal
way. The following JavaScript object

{% highlight javascript %}
{foo: 42, bar: "foobar"}
{% endhighlight %}

can be written in Scala.js either as

{% highlight scala %}
js.Dynamic.literal(foo = 42, bar = "foobar")
{% endhighlight %}

or as

{% highlight scala %}
js.Dynamic.literal("foo" -> 42, "bar" -> "foobar")
{% endhighlight %}

### Literal object construction using an Scala object interface
Sometimes for a nicer interface, literal objects can be implemented using
a trait interface. 
The above JavaScript code can be implemented using following code:

{% highlight scala %}
trait MyObject extends js.Object {
  val foo: Int = js.native
  val bar: String = js.native
}
{% endhighlight %}

A Scala object should be added for typesafe creation, it would help the readability 
of the code by removing lots of `js.Dynamic.literal` all over the code. 

{% highlight scala %}
object MyObject {
  def apply(foo: Int, bar: String): MyObject = 
    js.Dynamic.literal(foo = foo, bar = bar).asInstanceOf[MyObject]  
}
{% endhighlight %}

Alternatively, you can use anonymous classes extending `js.Object` or a
[Scala.js-defined JS trait](./sjs-defined-js-classes.html).
