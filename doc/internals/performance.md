---
layout: doc
title: Performance
---

Scala.js compiler optimizes the generated JavaScript very well, so that you, as a developer, don't have to worry about
your application's performance. It does things like inlining and rewriting iterators as while-loops to get good
performance even with complex Scala code.

## Benchmarks

To test how fast code the Scala.js compiler generates, we ported a [benchmark suite](https://github.com/sjrd/scalajs-benchmarks) 
from Dart and rewrote the benchmark code in idiomatic Scala. It has three separate benchmarks, testing various aspects
of the compiler.

Tests were executed using V8 engine in May, 2015. The results below have been normalized against hand-written JavaScript.  
A lower bar height means better performance. 

![Benchmarks]({{ site.baseurl }}/assets/img/benchmarks.png)

As you can see the use of Google Closure compiler sometimes degrades performance, because GCC's focus is optimizing
the size of the produced JavaScript, not its speed. The DeltaBlue benchmark has two variants; one using Scala native 
collections and another one using native `js.Array`, which gives a clear performance boost in this test.

It is safe to say that Scala.js performance is very good, right there on par with hand-written JavaScript in comparable 
cases.
