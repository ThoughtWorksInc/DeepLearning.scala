Stateless Future
================

`stateless-future` is a set of Domain-specific language for asynchronous programming, in the pure functional favor.

Stateless Futures provide similar API to `scala.concurrent.Future` and [scala.async](http://docs.scala-lang.org/sips/pending/async.html), except Stateless Futures are simpler, cleaner, and more powerful than `scala.concurrent.Future` and `scala.async`.

There was a [continuation plugin](http://www.scala-lang.org/old/node/2096) for Scala. The continuation plugin also provided a DSL to define control flows like `stateless-future` or `scala.async`. I created the following table to compare the three DSL:

|               | stateless-future | scala.concurrent.Future and scala.async | scala.util.continuations |
| ------------- | ---------------- | --------------------------------------- | ------------------------ |
| Stateless | Yes | No | Yes |
| Threading-free | Yes | No | Yes |
| Exception handling in "A-Normal Form" | Yes | No | No |
| Tail call optimization in "A-Normal Form" | Yes | No | No |
| Pattern matching in "A-Normal Form" | Yes | Yes | Yes, but buggy |
| Lazy val in "A-Normal Form" | No, because of [some underlying scala.reflect bugs](https://issues.scala-lang.org/browse/SI-8499) | Only for those contain no `await` | Yes, but buggy |

## Usage

### Create a Stateless Future

    import com.qifun.statelessFuture.Future
    val randomDoubleFuture = Future {
      println("Generating a random Double...")
      math.random()
    }

A Stateless Future instance is lazy, only evaluating when you query it. Thus there is nothing printed when you create the Stateless Future.


### Read from a Stateless Future

    println("I am going to read a random Double.")
    for (randomDouble <- randomDoubleFuture) {
      println(s"Recevied $randomDouble.")
    }

Output:

    I am going to read a random Double.
    Generating a random Double...
    Recevied 0.19722960355012198.

### Another Stateless Future that invokes the former Stateless Future twice.

    val anotherFuture = Future {
      println("I am going to read the first random Double."
      val randomDouble1 = randomDoubleFuture.await
      println(s"The first random Double is $randomDouble1."
      
      println("I am going to read the second random Double."
      val randomDouble2 = randomDoubleFuture.await
      println(s"The second random Double is $randomDouble1."
    }
    
    println("Before running the Future.")
    for (unit <- anotherFuture) {
      println("After running the Future.")
    }

Output:

    Before running the Future.
    I am going to read the first random Double.
    Generating a random Double...
    The first random Double is 0.10768210465170625.
    I am going to read the second random Double.
    Generating a random Double...
    The second random Double is 0.6134780449033244.
    After running the Future.

Note the magic `await` postfix, which invokes the the former Stateless Future `randomDoubleFuture`. It looks like normal Scala method calls, but does not block any thread.

### A complex example with control structures

    import scala.concurrent.duration._
    import scala.util.control.Exception.Catcher
    import com.qifun.statelessFuture.Future
    
    val executor = java.util.concurrent.Executors.newSingleThreadScheduledExecutor
    
    // Manually implements a Stateless Future, which is the asynchronous version of `Thread.sleep()`
    def asyncSleep(duration: Duration) = new Future[Unit] {
      import scala.util.control.TailCalls._
      def onComplete(handler: Unit => TailRec[Unit])(implicit catcher: Catcher[TailRec[Unit]]) = {
        executor.schedule(new Runnable {
          def run() {
            handler().result
          }
        }, duration.length, duration.unit)
        done()
      }
    }
    
    // Without the keyword `new`, you have the magic version of `Future` constructor,
    // which enables the magic postfix `await`.
    val sleep10seconds = Future {
      var i = 0
      while (i < 10) {
        println(s"I have slept $i times.")
        // The magic postfix `await` invokes the asynchronous method `asyncSleep`.
        // It looks like normal `Thread.sleep()`, but does not block any thread.
        asyncSleep(1.seconds).await
        i += 1
      }
      i
    }
    
    // When `sleep10seconds` is running, it could report failures to this catcher
    implicit def catcher: Catcher[Unit] = {
      case e: Exception => {
        println("An exception occured when I was sleeping: " + e.getMessage)
      }
    }
    
    // A Stateless Future instance is lazy, only evaluating when you query it.
    println("Before the evaluation of the Stateless Future `sleep10seconds`.")
    for (total <- sleep10seconds) {
      println("After the evaluation of the Stateless Future `sleep10seconds`.")
      println(s"I slept $total times in total.")
      executor.shutdown()
    }


Run it and you will see the output:

    Before evaluation of the Stateless Future `sleep10seconds`.
    I have slept 0 times.
    I have slept 1 times.
    I have slept 2 times.
    I have slept 3 times.
    I have slept 4 times.
    I have slept 5 times.
    I have slept 6 times.
    I have slept 7 times.
    I have slept 8 times.
    I have slept 9 times.
    After evaluation of the Stateless Future `sleep10seconds`.
    I slept 10 times in total.

## Further Information

There are two sorts of API to use a Stateless Future, the for-comprehensions style API and "A-Normal Form" style API.

### For-Comprehensions

The for-comprehensions style API for `stateless-future` is like the [for-comprehensions for scala.concurrent.Future](http://docs.scala-lang.org/overviews/core/futures.html#functional_composition_and_forcomprehensions). 

    for (total <- sleep10seconds) {
      println("After evaluation of the Stateless Future `sleep10seconds`")
      println(s"I slept $total times in total.")
      executor.shutdown()
    }

A notable difference between the two for-comprehensions implementations is the required implicit parameter. A `scala.concurrent.Future` requires an `ExecutionContext`, while a Stateless Future requires a `Catcher`.

    import scala.util.control.Exception.Catcher
    implicit def catcher: Catcher[Unit] = {
      case e: Exception => {
        println("An exception occured when I was sleeping: " + e.getMessage)
      }
    }

### A-Normal Form

"A-Normal Form" style API for Stateless Futures is like the pending proposal [scala.async](http://docs.scala-lang.org/sips/pending/async.html), except Stateless Futures require less limitations than `scala.async`.

    val sleep10seconds = Future {
      var i = 0
      while (i < 10) {
        println(s"I have slept $i times")
        // The magic postfix `await` invokes asynchronous method like normal `Thread.sleep()`,
        // and does not block any thread.
        asyncSleep(1.seconds).await
        i += 1
      }
      i
    }

The `Future` function for Stateless Futures corresponds to `async` method in `Async`, and the `await` postfix to Stateless Futures corresponds to `await` method in `Async`.

## Design

Regardless of the familiar veneers between Stateless Futures and `scala.concurrent.Future`, I have made some different designed choices on Stateless Futures.

### Statelessness

The Stateless Futures are pure functional, thus they will never store result values or exceptions. Instead, Stateless Futures evaluate lazily, and they do the same job every time you invoke `foreach` or `onComplete`. The behavior of Stateless Futures is more like monads in Haskell than futures in Java.

Also, there is no `isComplete` method in Stateless Futures. As a result, the users of Stateless Futures are forced not to share futures between threads, not to check the states in futures. They have to care about control flows instead of threads, and build the control flows by defining Stateless Futures.

By the way, Stateless Futures can be easy adapted to other stateful future implementation, and then the users can use the other future's stateful API on the adapted futures. For example, you can perform `scala.concurrent.Await.result` on a Stateless Future which is implicitly adapted to a `Future.ToConcurrentFuture`. By this approach, I have [ported](https://github.com/Atry/stateless-future-test) the most of `scala.async` test cases for Stateless Futures.

### Threading-free Model

There are too many threading models and implimentations in the Java/Scala world, `java.util.concurrent.Executor`, `scala.concurrent.ExecutionContext`, `javax.swing.SwingUtilities.invokeLater`, `java.util.Timer`, ... It is very hard to communicate between threading models. When a developer is working with multiple threading models, he must very carefully pass messages between threading models, or he have to maintain bulks of `synchronized` methods to properly deal with the shared variables between threads.

Why does he need multiple threading models? Because the libraries that he uses depend on different threading modes. For example, you must update Swing components in the Swing's UI thread, you must specify `java.util.concurrent.ExecutionService`s for `java.nio.channels.CompletionHandler`, and, you must specify `scala.concurrent.ExecutionContext`s for `scala.concurrent.Future` and `scala.async.Async`. Oops!

Think about somebody who uses Swing to develop a text editor software. He wants to create a state machine to update UI. He have heard the cool `scala.async`, then he uses the cool "A-Normal Form" expression in `async` to build the state machine that updates UI, and he types `import scala.concurrent.ExecutionContext.Implicits._` to suppress the compiler errors. Everything looks pretty, except the software always crashes.

Fortunately, `stateless-future` depends on none of these threading model, and cooperates with all of these threading models. If the poor guy tries Stateless Future, replacing `async { }` to `stateless-future`'s `Future { }`, deleting the `import scala.concurrent.ExecutionContext.Implicits._`, he will find that everything looks pretty like before, and does not crash any more. That's why threading-free model is important.

### Exception Handling

There were two `Future` implementations in Scala standard library, `scala.actors.Future` and `scala.concurrent.Future`. `scala.actors.Future`s are not designed to handling exceptions, since exceptions are always handled by actors. There is no way to handle a particular exception in a particular subrange of an actor.

Unlike `scala.actors.Future`s, `scala.concurrent.Future`s are designed to handle exceptions. But, unfortunately, `scala.concurrent.Future`s provide too many mechanisms to handle an exception. For example:

    import scala.concurrent.Await
    import scala.concurrent.ExecutionContext
    import scala.concurrent.duration.Duration
    import scala.util.control.Exception.Catcher
    import scala.concurrent.forkjoin.ForkJoinPool
    val threadPool = new ForkJoinPool()
    val catcher1: Catcher[Unit] = { case e: Exception => println("catcher1") }
    val catcher2: Catcher[Unit] = {
      case e: java.io.IOException => println("catcher2")
      case other: Exception => throw new RuntimeException(other)
    }
    val catcher3: Catcher[Unit] = {
      case e: java.io.IOException => println("catcher4")
      case other: Exception => throw new RuntimeException(other)
    }
    val catcher4: Catcher[Unit] = { case e: Exception => println("catcher4") }
    val catcher5: Catcher[Unit] = { case e: Exception => println("catcher5") }
    val catcher6: Catcher[Unit] = { case e: Exception => println("catcher6") }
    val catcher7: Catcher[Unit] = { case e: Exception => println("catcher7") }
    def future1 = scala.concurrent.future { 1 }(ExecutionContext.fromExecutor(threadPool, catcher1))
    def future2 = scala.concurrent.Future.failed(new Exception)
    val composedFuture = future1.flatMap { _ => future2 }(ExecutionContext.fromExecutor(threadPool, catcher2))
    composedFuture.onFailure(catcher3)(ExecutionContext.fromExecutor(threadPool, catcher4))
    composedFuture.onFailure(catcher5)(ExecutionContext.fromExecutor(threadPool, catcher6))
    try { Await.result(composedFuture, Duration.Inf) } catch { case e if catcher7.isDefinedAt(e) => catcher7(e) }

Is any sane developer able to tell which catchers will receive the exceptions?

There are too many concepts about exceptions when you work with `scala.concurrent.Future`. You have to remember the different exception handling strategies between `flatMap`, `recover`, `recoverWith` and `onFailure`, and the difference between `scala.concurrent.Future.failed(new Exception)` and `scala.concurrent.future { throw new Exception }`.

`scala.async` does not make things better, because `scala.async` will [produce a compiler error](https://github.com/scala/async/blob/master/src/test/scala/scala/async/neg/NakedAwait.scala#L104) for every `await` in a `try` statement.

Fortunately, you can get rid of all those concepts if you switch to `stateless-future`. There is neither `catcher` implicit parameter in `flatMap` or `map` in Stateless Futures, nor `onFailure` nor `recover` method at all. You just simply `try`, and things get done. See [the examples](https://github.com/Atry/stateless-future-test/blob/2.10.x/test/src/test/scala/com/qifun/statelessFuture/test/run/exceptions/ExceptionsSpec.scala#L62) to learn that.

### Tail Call Optimization

Tail call optimization is an important feature for pure functional programming. Without tail call optimization, many recursive algorithm will fail at run-time, and you will get the well-known `StackOverflowError`.

The Scala language provides `scala.annotation.tailrec` to automatically optimize simple tail recursions, and `scala.util.control.TailCalls` to manually optimize complex tail calls.

`stateless-future` project is internally based on `scala.util.control.TailCalls`, and automatically performs tail call optimization in the magic `Future` blocks, without any additional special syntax.

See [this example](https://github.com/Atry/stateless-future-test/blob/2.10.x/test/src/test/scala/com/qifun/statelessFuture/test/run/tailcall/TailcallSpec.scala). The example creates 500,000,000 stack levels in recursion. And it just works, without any `StackOverflowError` or `OutOfMemoryError`. Note that if you port this example for `scala.async` it will throw an `OutOfMemoryError` or a `TimeoutException`.

## Installation

Put these lines in your `build.sbt` if you use [Sbt](http://www.scala-sbt.org/):

    libraryDependencies += "com.qifun" %% "stateless-future" % "0.1.1"

`stateless-future` should work with Scala 2.10.3, 2.10.4, or 2.11.0.

## Known issues

 * `lazy val`s in magic `Future` block are not supported.
 * In [some rare cases](https://github.com/Atry/stateless-future-test/blob/2.10.x/test/src/test/scala/com/qifun/statelessFuture/test/run/match0/Match0.scala#L85), if you create multiple `val` with same name in one `Future` block, the last `val` may be referred unexpectly.
 * [Some complex existential types](https://github.com/Atry/stateless-future-test/blob/2.10.x/test/src/test/scala/com/qifun/statelessFuture/test/run/uncheckedBounds/UncheckedBoundsSpec.scala) may cause compiler errors.

Clone [stateless-future-test](https://github.com/Atry/stateless-future-test) and run the test cases to check these limitations.

<hr/>

<table>
<tr>
<td>
<h5>明日歌</h5>
<p>
明日复明日，<br/>
明日何其多。<br/>
我生待明日，<br/>
万事成蹉跎。<br/>
</p>
<p align="right"><font size="-1"><i>
文嘉／錢鶴灘
</i><font></p>
</td>
<td>
<h5>Future Song</h5>
<p>
The Future flatMaps a Future.<br/>
The Future tailcalls forever.<br/>
My life to await the Future.<br/>
It comes OutOfMemoryError.
</p>
<p align="right"><font size="-1"><i>
Wen Jia / Qian Hetan
</i></font></p>
</td>
</tr>
</table>
