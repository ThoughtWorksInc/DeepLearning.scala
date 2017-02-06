---
layout: doc
title: Quick Start
---

This Quick Start will help you set up your system for developing with Scala.js and guide you through your
 first Scala.js application.
 
# Installation

All you need to get started is 

* recent version of Java JDK [(download)](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
* Typesafe Activator(*) [(download)](https://www.typesafe.com/activator/download)

Once you have extracted Activator into a suitable folder, add the folder to your `PATH` to access it from the command line.

For editing Scala.js projects you'll probably want to install one of the IDEs that support Scala.js, but you can do this
also later. IDE is not required for compiling/running any of the tutorial projects.
 
* [Scala IDE](http://scala-ide.org/) (the official IDE for Scala, based on Eclipse)
* [IntelliJ IDEA](https://www.jetbrains.com/idea/download/) (a famous Java IDE with good support for Scala via a plugin)

(*) if you already have SBT installed, you can use that instead of Activator, but then you'll need to download examples
and tutorial projects manually.

# Hello World in Scala.js

Go to your command line shell and switch to the directory where you keep your development projects. Run `activator`
(which should be on your `PATH` now) with the following command line to create a new Scala.js project from a template.

{% highlight bash %}
activator new hello-scalajs scalajs_hello_world
{% endhighlight %}

Change into the project directory and run the application.

{% highlight bash %}
cd hello_scalajs
activator run
{% endhighlight %}

Note that this will take a while on the first time as `activator` downloads all required packages including the Scala
compiler. The next time you run the application, it will start *much* faster!

Once all packages have been download and the project compiled, you can navigate in your browser to <http://localhost:12345>
to access the application. You should now see a welcome screen in your browser
 
[ screenshot ]

## Making edits

While the application is running, you can edit the source code and it will be automatically recompiled and refreshed in
your browser.

# What next?

Check one of the [tutorials](../tutorial) to continue your journey!
