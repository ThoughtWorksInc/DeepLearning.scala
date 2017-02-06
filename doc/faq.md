---
layout: doc
title: Frequently asked questions
---

### How to structure a .sbt build to cross-compile with Scala and Scala.js?

The best way to do this is to have two sbt projects, with two different base
directories that share a common source directory. This is easily done with the
`sourceDirectory` or the `unmanagedSourceDirectories` setting of sbt.

Please follow our [cross-building guide](./project/cross-build.html) for details.

### Can I use macros with Scala.js? What about compiler plugins?

Yes, you can. There is nothing specific to Scala.js here.

### Where can I find an example of non trivial Scala.js project?

Have a look at these projects [built with Scala.js](../community/#built_with_scalajs).

### Have you considered targeting [asm.js](http://asmjs.org/)? Would it help?

asm.js would not help in implementing Scala.js.

asm.js was designed as a target for C-like languages, that consider the memory
as a huge array of bytes, and with manual memory management. By their own
acknowledgment (see [their FAQ](http://asmjs.org/faq.html)), it is not a good
target for managed languages like Scala.
