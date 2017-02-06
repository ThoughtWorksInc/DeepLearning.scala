---
layout: doc
title: Cross-Building
---

It is often desirable to compile the same source code with Scala.js and Scala JVM.
In order to do this, you need two different projects, one for Scala.js and one for Scala JVM and a folder with the shared source code.
You can then tell sbt to use the shared source folder in addition to the normal source locations.

To do this, we provide a builder, `crossProject`, which constructs two related sbt projects, one for the JVM, and one for JS.
See [the ScalaDoc of `CrossProject`]({{ site.production_url }}/api/sbt-scalajs/{{ site.versions.scalaJS }}/#org.scalajs.sbtplugin.cross.CrossProject)
for examples and documentation.

We give a simple example of how such a project, we call it `foo`, could look. You can find this project on [GitHub](https://github.com/scala-js/scalajs-cross-compile-example).

## Directory Structure

    <project root>
     +- jvm
     |   +- src/main/scala
     +- js
     |   +- src/main/scala
     +- shared
         +- src/main/scala

In `shared/src/main/scala` are the shared source files.
In `{js|jvm}/src/main/scala` are the source files specific to the respective platform (these folders are optional).

## sbt Build File

This is an example how your `build.sbt` could look like:

{% highlight scala %}
name := "Foo root project"

scalaVersion in ThisBuild := "2.11.8"

lazy val root = project.in(file(".")).
  aggregate(fooJS, fooJVM).
  settings(
    publish := {},
    publishLocal := {}
  )

lazy val foo = crossProject.in(file(".")).
  settings(
    name := "foo",
    version := "0.1-SNAPSHOT"
  ).
  jvmSettings(
    // Add JVM-specific settings here
  ).
  jsSettings(
    // Add JS-specific settings here
  )

lazy val fooJVM = foo.jvm
lazy val fooJS = foo.js
{% endhighlight %}

You now have separate projects to compile towards Scala.js and Scala JVM. Note the same name given to both projects, this allows them to be published with corresponding artifact names:

- `foo_2.11-0.1-SNAPSHOT.jar`
- `foo_sjs{{ site.versions.scalaJSBinary }}_2.11-0.1-SNAPSHOT.jar`

If you do not publish the artifacts, you may choose different names for the projects.

## Dependencies

If your cross compiled source depends on libraries, you may use `%%%` for both projects. It will automatically determine whether you are in a Scala/JVM or a Scala.js project. For example, if your code uses [Scalatags](http://github.com/lihaoyi/scalatags), your project definitions look like this:

{% highlight scala %}
lazy val foo = crossProject.in(file(".")).
  settings(
    // other settings
    libraryDependencies += "com.lihaoyi" %%% "scalatags" % "0.4.3"
  )
{% endhighlight %}

instead of the more repetitive variant:

{% highlight scala %}
lazy val foo = crossProject.in(file(".")).
  settings(
    // other settings
  ).
  jvmSettings(
    libraryDependencies += "com.lihaoyi" %% "scalatags" % "0.4.3"
  ).
  jsSettings(
    libraryDependencies += "com.lihaoyi" %%% "scalatags" % "0.4.3"
  )
{% endhighlight %}

## Exporting shared classes to JavaScript

When working with shared classes, you may want to export some of them to JavaScript.
This is done with annotations as explained in [Export Scala.js APIs to JavaScript](../interoperability/export-to-javascript.html).

These annotations are part of the Scala.js library which is not available on the Scala JVM project.
In order for the annotated classes to compile on the JVM project, you should add the scalajs-stubs library to your JVM dependencies as "provided" (used only during compilation and not included at runtime):

	libraryDependencies += "org.scala-js" %% "scalajs-stubs" % scalaJSVersion % "provided"

[scalajs-stubs]({{ site.production_url }}/api/scalajs-stubs/latest/) is a tiny JVM (only) library containing Scala.js export annotations.
