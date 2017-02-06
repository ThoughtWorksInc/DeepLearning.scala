---
layout: doc
title: Semantics of Scala.js
tagline: and how they differ from Scala
---

In general, the semantics of the Scala.js language are the same as Scala on
the JVM.
However, a few differences exist, which we mention here.

## Primitive data types

All primitive data types work exactly as on the JVM, with the following three
exceptions.

### Floats can behave as Doubles by default

Scala.js underspecifies the behavior of `Float`s by default.
Any `Float` value can be stored as a `Double` instead, and any operation on
`Float`s can be computed with double precision.
The choice of whether or not to behave as such, when and where, is left to the
implementation.

If exact single precision operations are important to your application, you can
enable strict-floats semantics in Scala.js, with the following sbt setting:

{% highlight scala %}
scalaJSSemantics ~= { _.withStrictFloats(true) }
{% endhighlight %}

Note that this can have a major impact on performance of your application on
JS interpreters that do not support
[the `Math.fround` function](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/fround).

### `toString` of `Float`, `Double` and `Unit`

`x.toString()` returns slightly different results for floating point numbers
and `()` (`Unit`).

{% highlight scala %}
().toString   // "undefined", instead of "()"
1.0.toString  // "1", instead of "1.0"
1.4f.toString // "1.399999976158142" instead of "1.4"
{% endhighlight %}

In general, a trailing `.0` is omitted.
Floats print in a weird way because they are printed as if they were Doubles,
which means their lack of precision shows up.

To get sensible and portable string representation of floating point numbers,
use `String.format()` or related methods.

### Runtime type tests are based on values

Instance tests (and consequently pattern matching) on any of `Byte`,
`Short`, `Int`, `Float`, `Double` are based on the value and not the
type they were created with. The following are examples:

- 1 matches `Byte`, `Short`, `Int`, `Float`, `Double`
- 128 (`> Byte.MaxValue`) matches `Short`, `Int`, `Float`, `Double`
- 32768 (`> Short.MaxValue`) matches `Int`, `Float`, `Double`
- 2147483647 matches `Int`, `Double` if strict-floats are enabled
  (because that number cannot be represented in a strict 32-bit `Float`),
  otherwise `Int`, `Float` and `Double`
- 2147483648 (`> Int.MaxValue`) matches `Float`, `Double`
- 1.5 matches `Float`, `Double`
- 1.4 matches `Double` only if strict-floats are enabled,
  otherwise `Float` and `Double`
  (unlike 1.5, the value 1.4 cannot be represented in a strict 32-bit `Float`)
- `NaN`, `Infinity`, `-Infinity` and `-0.0` match `Float`, `Double`

As a consequence, the following apparent subtyping relationships hold:

    Byte <:< Short <:<  Int  <:< Double
                   <:< Float <:<

if strict-floats are enabled, or

    Byte <:< Short <:< Int <:< Float =:= Double

otherwise.

## Undefined behaviors

The JVM is a very well specified environment, which even specifies how some
bugs are reported as exceptions.
Currently known exhaustive list of exceptions are:

* `NullPointerException`
* `ArrayIndexOutOfBoundsException` and `StringIndexOutOfBoundsException`
* `ClassCastException`
* `ArithmeticException` (such as integer division by 0)
* `StackOverflowError` and other `VirtualMachineError`s

Because Scala.js does not receive VM support to detect such erroneous
conditions, checking them is typically too expensive.

Therefore, all of these are considered
[undefined behavior](http://en.wikipedia.org/wiki/Undefined_behavior).

Some of these, however, can be configured to be compliant with the JVM
specification using sbt settings.
Currently, only `ClassCastException`s (thrown by invalid `asInstanceOf` calls)
are configurable, but the list will probably expand in future versions.

Every configurable undefined behavior has 3 possible modes:

* `Compliant`: behaves as specified on a JVM
* `Unchecked`: completely unchecked and undefined
* `Fatal`: checked, but throws
  [`UndefinedBehaviorError`s]({{ site.production_url }}/api/scalajs-library/{{ site.versions.scalaJS }}/#scala.scalajs.runtime.UndefinedBehaviorError)
  instead of the specified exception.

By default, undefined behaviors are in `Fatal` mode for `fastOptJS` and in
`Unchecked` mode for `fullOptJS`.
This is so that bugs can be detected more easily during development, with
predictable exceptions and stack traces.
In production code (`fullOptJS`), the checks are removed for maximum
efficiency.

`UndefinedBehaviorError`s are *fatal* in the sense that they are not matched by
`case NonFatal(e)` handlers.
This makes sure that they always crash your program as early as possible, so
that you can detect and fix the bug.
It is *never* OK to catch an `UndefinedBehaviorError` (other than in a testing
framework), since that means your program will behave differently in `fullOpt`
stage than in `fastOpt`.

If you need a particular kind of exception to be thrown in compliance with the
JVM semantics, you can do so with an sbt setting.
For example, this setting enables compliant `asInstanceOf`s:

{% highlight scala %}
scalaJSSemantics ~= { _.withAsInstanceOfs(
  org.scalajs.core.tools.sem.CheckedBehavior.Compliant) }
{% endhighlight %}

Note that this will have (potentially major) performance impacts.

## JavaScript interoperability

The JavaScript interoperability feature is, in itself, a big semantic
difference. However, its details are discussed in a
[dedicated page](./interoperability).

## Reflection

Java reflection and, a fortiori, Scala reflection, are not supported. There is
limited support for `java.lang.Class`, e.g., `obj.getClass.getName` will work
for any Scala.js object (not for objects that come from JavaScript interop).

## Regular expressions

[JavaScript regular expressions](http://developer.mozilla.org/en/docs/Core_JavaScript_1.5_Guide:Regular_Expressions)
are slightly different from
[Java regular expressions](http://docs.oracle.com/javase/6/docs/api/java/util/regex/Pattern.html).
The support for regular expressions in Scala.js is implemented on top of
JavaScript regexes.

This sometimes has an impact on functions in the Scala library that
use regular expressions themselves. A list of known functions that are
affected is given here:

- `StringLike.split(x: Array[Char])` (see issue [#105](https://github.com/scala-js/scala-js/issues/105))

## Symbols

`scala.Symbol` is supported, but is a potential source of memory leaks
in applications that make heavy use of symbols. The main reason is that
JavaScript does not support weak references, causing all symbols created
by Scala.js to remain in memory throughout the lifetime of the application.

## Enumerations

The methods `Value()` and `Value(i: Int)` on `scala.Enumeration` use
reflection to retrieve a string representation of the member name and
are therefore -- in principle -- unsupported. However, since
Enumerations are an integral part of the Scala library, Scala.js adds
limited support for these two methods:

<ol>
<li>Calls to either of these two methods of the forms:

{% highlight scala %}
val <ident> = Value
val <ident> = Value(<num>)
{% endhighlight %}

are statically rewritten to (a slightly more complicated version of):

{% highlight scala %}
val <ident> = Value("<ident>")
val <ident> = Value(<num>, "<ident>")
{% endhighlight %}

Note that this also includes calls like
{% highlight scala %}
val A, B, C, D = Value
{% endhighlight %}
since they are desugared into separate <code>val</code> definitions.
</li>
<li>Calls to either of these two methods which could not be rewritten,
or calls to constructors of the protected <code>Val</code> class without an
explicit name as parameter, will issue a warning.</li>
</ol>

Note that the name rewriting honors the `nextName`
iterator. Therefore, the full rewrite is:

{% highlight scala %}
val <ident> = Value(
  if (nextName != null && nextName.hasNext)
    nextName.next()
  else
    "<ident>"
)
{% endhighlight %}

We believe that this covers most use cases of
`scala.Enumeration`. Please let us know if another (generalized)
rewrite would make your life easier.
