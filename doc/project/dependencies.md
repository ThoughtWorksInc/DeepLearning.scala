---
layout: doc
title: Dependencies
---


## Depending on Scala.js libraries

To be able to use a Scala library in Scala.js, it has to be separately compiled for Scala.js. You then can add it to your library dependencies as follows:

{% highlight scala %}
libraryDependencies += "org.scala-js" %%% "scalajs-dom" % "{{ site.versions.scalaJSDOM }}"
{% endhighlight %}

Note the `%%%` (instead of the usual `%%`) which will add the current Scala.js version to the artifact name. This allows to

- Cross-publish libraries to different Scala.js versions
- Disambiguate Scala.js artifacts from their JVM counterparts

Some Scala.js core libraries (such as the Scala.js library itself) do not need the `%%%` since their version number *is* the Scala.js version number itself.

Note that you can also use `%%%` in a Scala/JVM project, in which case it will be the same as `%%`. This allows you to use the same `libraryDependencies` settings when cross compiling Scala/JVM and Scala.js.

## Depending on JavaScript libraries

Thanks to [WebJars](http://www.webjars.org/), you can easily fetch a JavaScript library like so:

{% highlight scala %}
libraryDependencies += "org.webjars" % "jquery" % "2.1.4"
{% endhighlight %}

This will fetch the required JAR containing jQuery. However, it will not include it once you run your JavaScript code, since there is no class-loading process for JavaScript.

The Scala.js sbt plugin has `jsDependencies` for this purpose. You can write:

{% highlight scala %}
jsDependencies += "org.webjars" % "jquery" % "2.1.4" / "2.1.4/jquery.js"
{% endhighlight %}

This will make your project depend on the respective WebJar and include a file named `**/2.1.4/jquery.js` in the said WebJar when your project is run or tested. We are trying to make the semantics of "include" to be as close as possible to writing:

{% highlight html %}
<script type="text/javascript" src="..."></script>
{% endhighlight %}

All `jsDependencies` and associated metadata (e.g. for ordering) are persisted in a file (called `JS_DEPENDENCIES`) and shipped with the artifact your project publishes. For example, if you depend on the `scalajs-jquery` package for Scala.js, you do not need to explicitly depend or include `jquery.js`; this mechanism does it for you.

Note: This will **not** dump the JavaScript libraries in the file containing your compiled Scala.js code as this would not work across all JavaScript virtual machines. However, the Scala.js plugin can generate a separate file that contains all raw JavaScript dependencies (see [below](#packageJSDependencies)).

### Scoping to a Configuration

You may scope `jsDependencies` on a given configuration, just like for normal `libraryDependencies`:

{% highlight scala %}
jsDependencies += "org.webjars" % "jquery" % "2.1.4" / "jquery.js" % "test"
{% endhighlight %}

### CommonJS name

Some (most?) JavaScript libraries try to adapt the best they can to the environment in which they are being executed.
When they do so, you have to specify explicitly the name under which they are exported in a CommonJS environment (such as Node.js), otherwise they won't work when executed in Node.js.
This is the purpose of the `commonJSName` directive, to be used like this:

{% highlight scala %}
jsDependencies += "org.webjars" % "mustachejs" % "0.8.2" / "mustache.js" commonJSName "Mustache"
{% endhighlight %}

which essentially translates to a prelude

{% highlight javascript %}
var Mustache = require("mustache.js");
{% endhighlight %}

when running with Node.js from sbt (with `run`, `test`, etc.).

### Dependency Ordering

Since JavaScript does not have a class loading mechanism, the order in which libraries are loaded may matter. If this is the case, you can specify a library's dependencies like so:

{% highlight scala %}
jsDependencies += "org.webjars" % "jasmine" % "1.3.1" / "jasmine-html.js" dependsOn "jasmine.js"
{% endhighlight %}

Note that the dependee must be declared as explicit dependency elsewhere, but not necessarily in this project (for example in a project the current project depends on).

### Local JavaScript Files

If you need to include JavaScript files which are provided in the resources of your project, use:

{% highlight scala %}
jsDependencies += ProvidedJS / "myJSLibrary.js"
{% endhighlight %}

This will look for `myJSLibrary.js` in the resources and include it. It is an error if it doesn't exist. You may use ordering and scoping if you need.

### <a name="packageJSDependencies"></a> Write a Dependency File

If you want all JavaScript dependencies to be concatenated to a single file (for easy inclusion into a HTML file for example), you can set:

{% highlight scala %}
skip in packageJSDependencies := false
{% endhighlight %}

in your project settings. The resulting file in the target folder will have the suffix `-jsdeps.js`.
