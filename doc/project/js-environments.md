---
layout: doc
title: JavaScript Environments
---

In order to decide how to run JavaScript code, the Scala.js sbt plugin uses the setting key `jsEnv`.
By default, `jsEnv` is set to use [Node.js](http://nodejs.org/), which you need to install separately.
If your application or one of its libraries requires a DOM (which can be specified with `jsDependencies += RuntimeDOM`), you will also need to install [`jsdom`](https://github.com/tmpvar/jsdom) with `npm install jsdom`.

## Alternative JavaScript environments

There are several alternative JavaScript environments that you can use:

* [Selenium](http://docs.seleniumhq.org/), provided by a separate project [scalajs-env-selenium](https://github.com/scala-js/scala-js-env-selenium)
* [PhantomJS](http://phantomjs.org/), with `jsEnv := PhantomJSEnv().value` (PhantomJS needs to be installed separately)
* Rhino (deprecated), with `scalaJSUseRhino in Global := true`

## <a name="phantomjs-no-auto-terminate"></a> Disabling auto-termination of PhantomJS

By default, the PhantomJS interpreter terminates itself as soon as the `main()` method returns.
This may not be what you want, if for example you register time-outs or use WebSockets.
You can disable this behavior with the following setting:

{% highlight scala %}
jsEnv := PhantomJSEnv(autoExit = false).value
{% endhighlight %}

You can terminate the interpreter from your Scala code with

{% highlight scala %}
System.exit(0)
{% endhighlight %}

## <a name="phantomjs-arguments"></a> Passing arguments to PhantomJS

You can pass command-line arguments to the PhantomJS interpreter like this:

{% highlight scala %}
jsEnv := PhantomJSEnv(args = Seq("arg1", "arg2")).value
{% endhighlight %}

For more options of the PhantomJS environment, see
[the ScalaDoc of `PhantomJSEnv`]({{ site.production_url }}/api/sbt-scalajs/{{ site.versions.scalaJS }}/#org.scalajs.sbtplugin.ScalaJSPlugin$$AutoImport$).

## <a name="node-on-ubuntu"></a> Node.js on Ubuntu

The easiest way to handle Node.js versions and installations on Ubuntu (and in Linux systems in general) is to use [nvm](https://github.com/creationix/nvm). All instructions are included.

Then run `nvm` to install the version of Node.js that you want:

    nvm install 5.0

For more options of the Node.js environment, see
[the ScalaDoc of `NodeJSEnv`]({{ site.production_url }}/api/sbt-scalajs/{{ site.versions.scalaJS }}/#org.scalajs.sbtplugin.ScalaJSPlugin$$AutoImport$).
